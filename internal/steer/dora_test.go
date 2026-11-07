package steer

import (
	"errors"
	"math"
	"sync"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestOracleDoRAExtendedRefusal(t *testing.T) {
	cfg := DefaultDoRAConfig()
	cfg.Rank = 2
	cfg.Alpha = 2.0
	cfg.Strength = 0.80
	cfg.RegAlpha = 0.0

	ft := NewExtendedRefusalFineTuner(cfg)

	// Base weight matrix W_0 (3 x 3) with exact integer row norms:
	// row 0: [3, 4, 0] -> norm = 5
	// row 1: [0, 6, 8] -> norm = 10
	// row 2: [1, 2, 2] -> norm = 3
	w0 := [][]float64{
		{3.0, 4.0, 0.0},
		{0.0, 6.0, 8.0},
		{1.0, 2.0, 2.0},
	}
	refusalDirs := [][]float64{
		{1.0, 0.0, 0.0},
	}
	benignDir := []float64{0.0, 0.0, 1.0}

	adapter, hardened, err := ft.HardenLayer(0, w0, refusalDirs, benignDir)
	if err != nil {
		t.Fatalf("HardenLayer failed: %v", err)
	}

	// Verify DoRA magnitude decomposition strictly preserves row L2 norms: ||W_DoRA[i]||_2 == ||W_0[i]||_2
	wantNorms := []float64{5.0, 10.0, 3.0}
	for i, wantN := range wantNorms {
		if math.Abs(adapter.Magnitude[i]-wantN) > 1e-12 {
			t.Errorf("adapter.Magnitude[%d] = %.15f, want %.15f", i, adapter.Magnitude[i], wantN)
		}
		gotN := floats.Norm(hardened[i], 2)
		if math.Abs(gotN-wantN) > 1e-12 {
			t.Errorf("||W_DoRA[%d]||_2 = %.15f, want %.15f", i, gotN, wantN)
		}
	}

	// Verify KL ceiling <= 0.20 and coherence >= 0.90
	if adapter.BaselineKL > cfg.MaxKLDivergence {
		t.Errorf("BaselineKL = %f, want <= %f", adapter.BaselineKL, cfg.MaxKLDivergence)
	}
	if adapter.CoherenceRatio < cfg.MinCoherenceRatio {
		t.Errorf("CoherenceRatio = %f, want >= %f", adapter.CoherenceRatio, cfg.MinCoherenceRatio)
	}
	if adapter.RefusalSuppression <= 0.0 {
		t.Errorf("expected positive refusal suppression, got %f", adapter.RefusalSuppression)
	}

	// Full checkpoint hardening & rollback determinism
	ckpt := &CheckpointWeights{
		ModelID:   "qwen2.5-7b",
		HighRisk:  true,
		RiskScore: 0.92,
		Layers: map[int][][]float64{
			0: cloneMatrix(w0),
		},
	}
	origHash := ComputeCheckpointSHA256(ckpt.Layers)

	res, err := ft.HardenCheckpoint(ckpt, map[int][][]float64{0: refusalDirs}, map[int][]float64{0: benignDir})
	if err != nil {
		t.Fatalf("HardenCheckpoint failed: %v", err)
	}
	if res.BaseCheckpointHash != origHash {
		t.Errorf("BaseCheckpointHash = %s, want %s", res.BaseCheckpointHash, origHash)
	}
	if res.HardenedCheckpointHash == origHash {
		t.Error("expected HardenedCheckpointHash to differ from BaseCheckpointHash")
	}

	// Rollback must restore exact original weights and SHA-256
	if err := ft.RollbackCheckpoint(ckpt); err != nil {
		t.Fatalf("RollbackCheckpoint failed: %v", err)
	}
	restoredHash := ComputeCheckpointSHA256(ckpt.Layers)
	if restoredHash != origHash {
		t.Errorf("RollbackCheckpoint hash = %s, want %s", restoredHash, origHash)
	}
}

func TestAccuracyDoRAKLAndNormInvariants(t *testing.T) {
	cfg := DefaultDoRAConfig()
	cfg.Rank = 3
	cfg.Alpha = 6.0
	cfg.MaxKLDivergence = 0.20
	ft := NewExtendedRefusalFineTuner(cfg)

	weights := [][]float64{
		{2.5, -1.2, 0.8, 1.4},
		{-0.9, 3.1, -0.4, 0.7},
		{1.1, 0.5, 2.8, -1.0},
		{0.3, -2.0, 1.5, 2.2},
	}
	refusalDirs := [][]float64{
		{1.0, 0.2, 0.0, 0.0},
		{0.0, 1.0, -0.1, 0.0},
	}
	benignDir := []float64{0.0, 0.0, 0.6, 0.8}

	adapter, hardened, err := ft.HardenLayer(3, weights, refusalDirs, benignDir)
	if err != nil {
		t.Fatalf("HardenLayer error: %v", err)
	}

	// Invariant 1: Every row preserves its exact L2 magnitude
	for i := range weights {
		origNorm := floats.Norm(weights[i], 2)
		hardNorm := floats.Norm(hardened[i], 2)
		if math.Abs(origNorm-hardNorm) > 1e-10 {
			t.Errorf("row %d norm changed: orig=%f, hardened=%f", i, origNorm, hardNorm)
		}
	}

	// Invariant 2: KL ceiling respected (<= 0.20)
	if adapter.BaselineKL > 0.20+1e-9 {
		t.Errorf("BaselineKL %f exceeded 0.20 ceiling", adapter.BaselineKL)
	}

	// Invariant 3: Benign capability direction is preserved with high cosine similarity
	if adapter.CoherenceRatio < 0.90 {
		t.Errorf("CoherenceRatio %f < 0.90", adapter.CoherenceRatio)
	}
}

func TestDoRAExtendedRefusalFineTuner_GatingDryRunAndConcurrency(t *testing.T) {
	ft := NewExtendedRefusalFineTuner(DoRAConfig{HighRiskOnly: true})

	w := [][]float64{{1.0, 2.0}, {3.0, 4.0}}
	subspace := map[int][][]float64{0: {{1.0, 0.0}}}

	// Low-risk checkpoint should be rejected by HighRiskOnly gating
	lowRiskCkpt := &CheckpointWeights{
		ModelID:   "safe-model",
		HighRisk:  false,
		RiskScore: 0.15,
		Layers:    map[int][][]float64{0: cloneMatrix(w)},
	}
	if _, err := ft.HardenCheckpoint(lowRiskCkpt, subspace, nil); !errors.Is(err, ErrCheckpointNotHighRisk) {
		t.Errorf("expected ErrCheckpointNotHighRisk, got %v", err)
	}

	// Dry-run mode should not mutate checkpoint weights
	dryCfg := DefaultDoRAConfig()
	dryCfg.DryRun = true
	dryCfg.TargetLayers = []int{0}
	ftDry := NewExtendedRefusalFineTuner(dryCfg)
	if !ftDry.Config().DryRun {
		t.Error("expected DryRun=true in Config()")
	}

	dryCkpt := &CheckpointWeights{
		ModelID:  "high-risk-model",
		HighRisk: true,
		Layers: map[int][][]float64{
			0: cloneMatrix(w),
			1: cloneMatrix(w),
		},
	}
	beforeHash := ComputeCheckpointSHA256(dryCkpt.Layers)
	res, err := ftDry.HardenCheckpoint(dryCkpt, subspace, nil)
	if err != nil {
		t.Fatalf("DryRun HardenCheckpoint error: %v", err)
	}
	afterHash := ComputeCheckpointSHA256(dryCkpt.Layers)
	if beforeHash != afterHash {
		t.Error("DryRun mutated checkpoint weights in-place")
	}
	if res.HardenedCheckpointHash == beforeHash {
		t.Error("expected HardenedCheckpointHash to reflect computed DoRA weights")
	}

	// Error paths
	if _, _, err := ft.HardenLayer(0, nil, [][]float64{{1, 0}}, nil); err == nil {
		t.Error("expected error for empty weights")
	}
	if _, _, err := ft.HardenLayer(0, [][]float64{{1, 2}, {1}}, [][]float64{{1, 0}}, nil); err == nil {
		t.Error("expected error for ragged weights")
	}
	if _, _, err := ft.HardenLayer(0, w, nil, nil); err == nil {
		t.Error("expected error for empty refusalDirs")
	}
	if _, _, err := ft.HardenLayer(0, w, [][]float64{{1}}, nil); err == nil {
		t.Error("expected error for refusalDir dim mismatch")
	}
	if _, _, err := ft.HardenLayer(0, w, [][]float64{{1, 0}}, []float64{1}); err == nil {
		t.Error("expected error for benignDir dim mismatch")
	}
	if _, _, err := ft.HardenLayer(0, w, [][]float64{{1, 0}}, []float64{1, 0}); err == nil {
		t.Error("expected error when refusalDir is collinear with benignDir (zero rank after biprojection)")
	}
	if _, err := ft.HardenCheckpoint(nil, subspace, nil); err == nil {
		t.Error("expected error for nil checkpoint")
	}
	if _, err := ft.HardenCheckpoint(dryCkpt, nil, nil); err == nil {
		t.Error("expected error for empty subspace map")
	}
	if _, err := ft.HardenCheckpoint(dryCkpt, map[int][][]float64{99: {{1, 0}}}, nil); err == nil {
		t.Error("expected error for no matching layers")
	}
	if err := ft.RollbackCheckpoint(nil); err == nil {
		t.Error("expected error for nil checkpoint in RollbackCheckpoint")
	}
	ftEmpty := NewExtendedRefusalFineTuner(DefaultDoRAConfig())
	if err := ftEmpty.RollbackCheckpoint(dryCkpt); err == nil {
		t.Error("expected error when rolling back with no snapshots")
	}

	// Concurrent stress test under -race
	var wg sync.WaitGroup
	for g := 0; g < 12; g++ {
		wg.Add(1)
		go func(layer int) {
			defer wg.Done()
			_, _, _ = ft.HardenLayer(layer, w, [][]float64{{1.0, 0.0}}, []float64{0.0, 1.0})
		}(g)
	}
	wg.Wait()
}
