package steer

import (
	"context"
	"math"
	"sync"
	"testing"
)

func TestOracleCrescendoReAnchoring(t *testing.T) {
	// Subspace basis in R^4: u0 = [1, 0, 0, 0], u1 = [0, 1, 0, 0]
	subspace := [][]float64{
		{2.0, 0.0, 0.0, 0.0},
		{1.0, 3.0, 0.0, 0.0}, // Gram-Schmidt orthonormalizes to [0, 1, 0, 0]
	}
	cfg := DefaultCrescendoConfig()
	cfg.ReAnchorCadence = 5
	cfg.WindowSize = 15

	det, err := NewCrescendoDetector(subspace, cfg)
	if err != nil {
		t.Fatalf("NewCrescendoDetector failed: %v", err)
	}

	// Verify SubspaceEnergy oracle values
	// [3, 4, 0, 0] lies completely in span(u0, u1) -> energy = 1.0
	eFull, err := det.SubspaceEnergy([]float64{3.0, 4.0, 0.0, 0.0})
	if err != nil || math.Abs(eFull-1.0) > 1e-12 {
		t.Errorf("SubspaceEnergy(in-span) = (%f, %v), want (1.0, nil)", eFull, err)
	}
	// [0, 0, 5, 0] is orthogonal to span(u0, u1) -> energy = 0.0
	eOrtho, err := det.SubspaceEnergy([]float64{0.0, 0.0, 5.0, 0.0})
	if err != nil || math.Abs(eOrtho-0.0) > 1e-12 {
		t.Errorf("SubspaceEnergy(ortho) = (%f, %v), want (0.0, nil)", eOrtho, err)
	}
	// [3, 0, 4, 0] has norm 5 and projection norm 3 -> energy = 0.6
	eMixed, err := det.SubspaceEnergy([]float64{3.0, 0.0, 4.0, 0.0})
	if err != nil || math.Abs(eMixed-0.6) > 1e-12 {
		t.Errorf("SubspaceEnergy([3,0,4,0]) = (%f, %v), want (0.6, nil)", eMixed, err)
	}

	// Simulate an 18-turn slow crescendo escalation where each turn rotates 2.5 degrees
	// toward the harmful subspace. Without re-anchoring, 18 * 2.5° = 45° > 15° budget.
	// With periodic re-anchoring every 5 turns, incremental drift stays <= 10° (< 15°).
	ctx := context.Background()
	var prefix [][]float64
	escalatedDetected := false

	for turn := 0; turn < 18; turn++ {
		// Harm angle grows from 8° (mostly benign along z-axis) to 50.5° (strongly in x-y harm subspace)
		harmAngleRad := (8.0 + float64(turn)*2.5) * (math.Pi / 180.0)
		act := []float64{
			math.Sin(harmAngleRad) * 0.8,
			math.Sin(harmAngleRad) * 0.6,
			math.Cos(harmAngleRad),
			0.0,
		}
		prefix = append(prefix, act)

		res, err := det.StepTurn(ctx, turn, act, act, prefix)
		if err != nil {
			t.Fatalf("StepTurn(%d) error: %v", turn, err)
		}
		if res.IncrementalDriftDeg > ThetaMaxDegrees {
			t.Errorf("turn %d exceeded 15° angular drift budget: got %.2f°", turn, res.IncrementalDriftDeg)
		}
		if res.DriftAlert {
			t.Errorf("turn %d triggered unexpected drift alert", turn)
		}
		if res.IsCrescendoEscalation {
			escalatedDetected = true
		}
	}

	if !escalatedDetected {
		t.Error("expected 18-turn slow crescendo escalation to be detected")
	}
	if det.ReAnchorCount() < 3 {
		t.Errorf("expected at least 3 full-prefix re-anchors across 18 turns, got %d", det.ReAnchorCount())
	}
	if det.EscalationCount() == 0 {
		t.Error("expected positive EscalationCount")
	}
}

func TestAccuracyCrescendoRecallAndDriftBound(t *testing.T) {
	subspace := [][]float64{
		{1.0, 0.0, 0.0, 0.0},
		{0.0, 1.0, 0.0, 0.0},
	}
	cfg := DefaultCrescendoConfig()
	det, err := NewCrescendoDetector(subspace, cfg)
	if err != nil {
		t.Fatalf("NewCrescendoDetector error: %v", err)
	}

	ctx := context.Background()

	// 1. Evaluate 20 synthetic 16-turn crescendo escalation dialogues (Recall >= 95%)
	detectedCrescendos := 0
	totalCrescendos := 20
	for d := 0; d < totalCrescendos; d++ {
		det.Reset()
		var prefix [][]float64
		flagged := false
		baseAngle := 5.0 + float64(d%5)*1.0
		stepDeg := 2.4 + float64(d%3)*0.2

		for turn := 0; turn < 16; turn++ {
			rad := (baseAngle + float64(turn)*stepDeg) * (math.Pi / 180.0)
			act := []float64{
				math.Sin(rad) * 0.85,
				math.Sin(rad) * 0.52678,
				math.Cos(rad),
				0.1,
			}
			prefix = append(prefix, act)
			res, err := det.StepTurn(ctx, turn, act, act, prefix)
			if err != nil {
				t.Fatalf("crescendo dialogue %d turn %d error: %v", d, turn, err)
			}
			if res.IncrementalDriftDeg > cfg.ThetaMaxDeg {
				t.Fatalf("dialogue %d turn %d violated drift budget: %.2f > %.2f",
					d, turn, res.IncrementalDriftDeg, cfg.ThetaMaxDeg)
			}
			if res.IsCrescendoEscalation {
				flagged = true
			}
		}
		if flagged {
			detectedCrescendos++
		}
	}

	recall := float64(detectedCrescendos) / float64(totalCrescendos)
	if recall < 0.95 {
		t.Errorf("crescendo recall = %.2f, want >= 0.95", recall)
	}

	// 2. Evaluate 100 synthetic 16-turn benign dialogues (FPR <= 1%)
	falsePositives := 0
	totalBenign := 100
	for d := 0; d < totalBenign; d++ {
		det.Reset()
		var prefix [][]float64
		flagged := false

		for turn := 0; turn < 16; turn++ {
			// Benign oscillations around a low-risk angle (8°..14°) without monotonic escalation
			osc := 10.0 + 3.0*math.Sin(float64(turn+d)*0.9)
			rad := osc * (math.Pi / 180.0)
			act := []float64{
				math.Sin(rad) * 0.7,
				math.Sin(rad) * 0.7,
				math.Cos(rad),
				0.2,
			}
			prefix = append(prefix, act)
			res, err := det.StepTurn(ctx, turn, act, act, prefix)
			if err != nil {
				t.Fatalf("benign dialogue %d turn %d error: %v", d, turn, err)
			}
			if res.IncrementalDriftDeg > cfg.ThetaMaxDeg {
				t.Fatalf("benign dialogue %d turn %d violated drift budget: %.2f", d, turn, res.IncrementalDriftDeg)
			}
			if res.IsCrescendoEscalation {
				flagged = true
			}
		}
		if flagged {
			falsePositives++
		}
	}

	fpr := float64(falsePositives) / float64(totalBenign)
	if fpr > 0.01 {
		t.Errorf("crescendo false positive rate = %.2f, want <= 0.01", fpr)
	}
}

func TestCrescendoDetector_EdgeCasesAndConcurrency(t *testing.T) {
	if _, err := NewCrescendoDetector(nil, DefaultCrescendoConfig()); err == nil {
		t.Error("expected error for empty subspace")
	}
	if _, err := NewCrescendoDetector([][]float64{{1, 0}, {1}}, DefaultCrescendoConfig()); err == nil {
		t.Error("expected error for mismatched subspace dimensions")
	}
	if _, err := NewCrescendoDetector([][]float64{{0, 0}}, DefaultCrescendoConfig()); err == nil {
		t.Error("expected error for zero-rank subspace")
	}

	det, err := NewCrescendoDetector([][]float64{{1, 0, 0}}, CrescendoConfig{})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if _, err := det.SubspaceEnergy([]float64{1, 0}); err == nil {
		t.Error("expected error for wrong activation dimension")
	}
	if e, err := det.SubspaceEnergy([]float64{0, 0, 0}); err != nil || e != 0 {
		t.Errorf("expected (0, nil) for zero-norm activation, got (%f, %v)", e, err)
	}

	if _, _, err := det.ReAnchor(context.Background(), nil); err == nil {
		t.Error("expected error for empty prefix in ReAnchor")
	}
	if _, _, err := det.ReAnchor(context.Background(), [][]float64{{1, 0}}); err == nil {
		t.Error("expected error for wrong prefix dimension in ReAnchor")
	}

	ctxCancel, cancel := context.WithCancel(context.Background())
	cancel()
	if _, _, err := det.ReAnchor(ctxCancel, [][]float64{{1, 0, 0}}); err == nil {
		t.Error("expected error for canceled context in ReAnchor")
	}
	if _, err := det.StepTurn(ctxCancel, 0, []float64{1, 0, 0}, nil, nil); err == nil {
		t.Error("expected error for canceled context in StepTurn")
	}
	if _, err := det.StepTurn(context.Background(), 0, []float64{1, 0}, nil, nil); err == nil {
		t.Error("expected error for wrong turnAct dimension")
	}
	if _, err := det.StepTurn(context.Background(), 0, []float64{1, 0, 0}, []float64{1, 0}, nil); err == nil {
		t.Error("expected error for wrong steerVec dimension")
	}

	// Test proactive re-anchoring when single turn jumps > 0.80 * ThetaMaxDeg
	_, _ = det.StepTurn(context.Background(), 0, []float64{0, 0, 1}, []float64{0, 0, 1}, nil)
	resJump, err := det.StepTurn(context.Background(), 1, []float64{0.5, 0, 0.866}, []float64{0.5, 0, 0.866}, nil)
	if err != nil {
		t.Fatalf("StepTurn jump error: %v", err)
	}
	if !resJump.ReAnchored || resJump.DriftAlert {
		t.Errorf("expected proactive re-anchor without drift alert on 30° jump, got ReAnchored=%v, DriftAlert=%v",
			resJump.ReAnchored, resJump.DriftAlert)
	}

	// Concurrent stress test under -race
	var wg sync.WaitGroup
	for g := 0; g < 16; g++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			ang := float64(id+1) * 0.05
			vec := []float64{math.Sin(ang), math.Cos(ang), 0.2}
			_, _ = det.StepTurn(context.Background(), id, vec, vec, nil)
			_, _, _ = det.ReAnchor(context.Background(), [][]float64{vec})
			_ = det.ReAnchorCount()
			_ = det.EscalationCount()
			_ = det.Tracker().ReAnchorCount()
		}(g)
	}
	wg.Wait()
}
