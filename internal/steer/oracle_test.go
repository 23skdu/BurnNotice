package steer

import (
	"math"
	"testing"
)

func TestOracleMPOA_And_Biprojection(t *testing.T) {
	// Golden test for NormPreservingAblation (MPOA)
	// h = [3, 4, 0] (norm = 5), v = [1, 0, 0]
	//With alpha = 0.5, regAlpha = 0.0:
	// h_ortho = [0, 4, 0], restored = [0, 5, 0]
	// interpolated = 0.5 * [3, 4, 0] + 0.5 * [0, 5, 0] = [1.5, 4.5, 0]
	act := []float64{3.0, 4.0, 0.0}
	dir := []float64{1.0, 0.0, 0.0}

	gotMPOA := NormPreservingAblation(act, dir, 0.5, 0.0)
	wantMPOA := []float64{1.5, 4.5, 0.0}
	for i := range wantMPOA {
		if math.Abs(gotMPOA[i]-wantMPOA[i]) > 1e-12 {
			t.Errorf("MPOA[%d] = %.15f, want %.15f", i, gotMPOA[i], wantMPOA[i])
		}
	}

	// Degenerate collinear case (orthoNorm == 0) returns original activation
	collinear := NormPreservingAblation([]float64{2.0, 0.0, 0.0}, dir, 1.0, 0.0)
	if collinear[0] != 2.0 {
		t.Errorf("expected collinear MPOA to return original activation, got %v", collinear)
	}

	// Golden test for BiprojectedAblation with alpha = 0.5 interpolation
	// h = [3, 4, 0], harmDir = [1, 0, 0], benignDir = [0, 1, 0]
	// hOrtho = [0, 4, 0], benign deficit = 0 -> restored = [0, 5, 0]
	// alpha = 0.5 -> [1.5, 4.5, 0.0]
	gotBi := BiprojectedAblation(act, dir, []float64{0.0, 1.0, 0.0}, 0.5, 0.0)
	for i := range wantMPOA {
		if math.Abs(gotBi[i]-wantMPOA[i]) > 1e-12 {
			t.Errorf("BiprojectedAblation[%d] = %.15f, want %.15f", i, gotBi[i], wantMPOA[i])
		}
	}

	// Degenerate collinear BiprojectedAblation (orthoNorm == 0)
	colBi := BiprojectedAblation([]float64{2.0, 0.0, 0.0}, dir, nil, 1.0, 0.0)
	if colBi[0] != 2.0 {
		t.Errorf("expected collinear BiprojectedAblation to return original, got %v", colBi)
	}

	// ResolveComponentStrengths golden values
	cmd := Command{Strength: 0} // defaults baseStrength to 1.0
	if got := ResolveComponentStrengths(cmd, ComponentResidual); got != 1.0 {
		t.Errorf("ResolveComponentStrengths(Residual) = %f, want 1.0", got)
	}

	// ApplyCommand default strength and unknown method
	resNP := ApplyCommand(act, Command{Vector: dir, Method: MethodNormPreserving, Strength: 0})
	if math.Abs(resNP[1]-5.0) > 1e-12 {
		t.Errorf("ApplyCommand(MethodNormPreserving, strength=0) = %v, want [0, 5, 0]", resNP)
	}
	resBi := ApplyCommand(act, Command{Vector: dir, SecondaryVector: []float64{0, 1, 0}, Method: MethodBiprojection, Strength: 0})
	if math.Abs(resBi[1]-5.0) > 1e-12 {
		t.Errorf("ApplyCommand(MethodBiprojection, strength=0) = %v, want [0, 5, 0]", resBi)
	}
	resUnk := ApplyCommand(act, Command{Method: Method(99)})
	if resUnk[0] != 3.0 {
		t.Errorf("ApplyCommand(unknown) = %v, want original", resUnk)
	}
}

func TestGoldenBiasSteering(t *testing.T) {
	// b' = b - strength * (b · v / (||v||^2 + regAlpha)) * v
	// b = [4.0, 2.0, -1.0], v = [2.0, 0.0, 0.0], ||v||^2 = 4.0, regAlpha = 1.0, strength = 0.5
	// dot = 8.0, projCoeff = (8.0 / 5.0) * 0.5 = 0.8
	// b' = [4.0 - 0.8*2.0, 2.0, -1.0] = [2.4, 2.0, -1.0]
	bias := []float64{4.0, 2.0, -1.0}
	dir := []float64{2.0, 0.0, 0.0}
	got, err := ComputeBiasSteering(bias, dir, 0.5, 1.0)
	if err != nil {
		t.Fatalf("ComputeBiasSteering failed: %v", err)
	}
	want := []float64{2.4, 2.0, -1.0}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("ComputeBiasSteering[%d] = %.15f, want %.15f", i, got[i], want[i])
		}
	}

	// Error and edge cases
	if _, err := ComputeBiasSteering(nil, dir, 1.0, 0.0); err == nil {
		t.Error("expected error for empty bias")
	}
	if _, err := ComputeBiasSteering(bias, nil, 1.0, 0.0); err == nil {
		t.Error("expected error for empty direction")
	}
	zeroDirOut, err := ComputeBiasSteering(bias, []float64{0, 0, 0}, 1.0, 0.0)
	if err != nil || zeroDirOut[0] != 4.0 {
		t.Errorf("expected unchanged copy for zero-norm direction, got %v, err=%v", zeroDirOut, err)
	}
}

func TestOracleDirectionInterpolation(t *testing.T) {
	d0 := []float64{1.0, 0.0}
	d1 := []float64{0.0, 1.0}
	dirs := [][]float64{d0, d1}

	// t = 0.5, sigma = 0 (defaults to 1.0): symmetric weights -> [1/sqrt(2), 1/sqrt(2)]
	got, err := FloatDirectionInterpolation(dirs, 0.5, 0.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	invSqrt2 := 1.0 / math.Sqrt(2.0)
	if math.Abs(got[0]-invSqrt2) > 1e-12 || math.Abs(got[1]-invSqrt2) > 1e-12 {
		t.Errorf("got %v, want [%f, %f]", got, invSqrt2, invSqrt2)
	}

	// Clamping t < 0 and t > n-1
	gotNeg, err := FloatDirectionInterpolation(dirs, -5.0, 1.0)
	if err != nil {
		t.Fatalf("unexpected error for t < 0: %v", err)
	}
	gotZero, _ := FloatDirectionInterpolation(dirs, 0.0, 1.0)
	if math.Abs(gotNeg[0]-gotZero[0]) > 1e-12 {
		t.Errorf("t=-5 should clamp to t=0: %v vs %v", gotNeg, gotZero)
	}

	gotHigh, err := FloatDirectionInterpolation(dirs, 10.0, 1.0)
	if err != nil {
		t.Fatalf("unexpected error for t > n-1: %v", err)
	}
	gotOne, _ := FloatDirectionInterpolation(dirs, 1.0, 1.0)
	if math.Abs(gotHigh[0]-gotOne[0]) > 1e-12 {
		t.Errorf("t=10 should clamp to t=1: %v vs %v", gotHigh, gotOne)
	}

	// Empty direction dimension error
	if _, err := FloatDirectionInterpolation([][]float64{{}}, 0.0, 1.0); err == nil {
		t.Error("expected error for empty direction dimension")
	}

	// InterpolateDirectionFromCommand with active mix > 0
	cmdMixed, err := InterpolateDirectionFromCommand(d0, [][]float64{d1}, 0.5)
	if err != nil {
		t.Fatalf("InterpolateDirectionFromCommand failed: %v", err)
	}
	if math.Abs(cmdMixed[0]-invSqrt2) > 1e-12 || math.Abs(cmdMixed[1]-invSqrt2) > 1e-12 {
		t.Errorf("InterpolateDirectionFromCommand(0.5) = %v, want [%f, %f]", cmdMixed, invSqrt2, invSqrt2)
	}

	// Mix > len(allDirs)-1 clamps
	cmdClamped, err := InterpolateDirectionFromCommand(d0, [][]float64{d1}, 99.0)
	if err != nil || len(cmdClamped) != 2 {
		t.Errorf("InterpolateDirectionFromCommand(99) failed: %v, err=%v", cmdClamped, err)
	}
}

func TestGoldenCAST_ConditionTree(t *testing.T) {
	cs := NewConditionalSteerer()
	act := []float64{0.8, 0.6, 0.0} // norm = 1.0

	// OpOR test
	orRule := ConditionNode{
		Operator: OpOR,
		Children: []ConditionNode{
			{Category: "miss", Vector: []float64{0, 0, 1}, Threshold: 0.5},
			{Category: "hit", Vector: []float64{1, 0, 0}, Threshold: 0.5, SmoothScaling: true},
		},
	}
	// sim with [1,0,0] is 0.8; threshold=0.5 -> scale = (0.8 - 0.5)/(1.0 - 0.5) = 0.6
	scale, matched, err := cs.Evaluate(act, orRule)
	if err != nil || !matched || math.Abs(scale-0.6) > 1e-12 {
		t.Errorf("OpOR Evaluate = (%f, %v, %v), want (0.6, true, nil)", scale, matched, err)
	}

	// OpOR all miss
	orMiss := ConditionNode{
		Operator: OpOR,
		Children: []ConditionNode{
			{Category: "miss", Vector: []float64{0, 0, 1}, Threshold: 0.5},
		},
	}
	if s, m, err := cs.Evaluate(act, orMiss); err != nil || m || s != 0 {
		t.Errorf("OpOR miss = (%f, %v, %v), want (0, false, nil)", s, m, err)
	}

	// Edge cases: empty activation, empty node vector, dim mismatch, zero norm, unknown op, empty NOT
	if _, _, err := cs.Evaluate(nil, orRule); err == nil {
		t.Error("expected error on empty activation")
	}
	if s, m, err := cs.Evaluate(act, ConditionNode{Operator: OpSingle}); err != nil || !m || s != 1.0 {
		t.Errorf("empty vector SINGLE should trivially pass: (%f, %v, %v)", s, m, err)
	}
	if _, _, err := cs.Evaluate(act, ConditionNode{Operator: OpSingle, Vector: []float64{1}}); err == nil {
		t.Error("expected error on dim mismatch")
	}
	if s, m, err := cs.Evaluate(act, ConditionNode{Operator: OpSingle, Vector: []float64{0, 0, 0}, Threshold: 0.1}); err != nil || m || s != 0 {
		t.Errorf("zero norm condition vector should return (0, false, nil), got (%f, %v, %v)", s, m, err)
	}
	if s, m, err := cs.Evaluate(act, ConditionNode{Operator: OpNOT}); err != nil || !m || s != 1.0 {
		t.Errorf("empty OpNOT should pass: (%f, %v, %v)", s, m, err)
	}
	if _, _, err := cs.Evaluate(act, ConditionNode{Operator: "XOR"}); err == nil {
		t.Error("expected error on unknown operator")
	}

	// ApplyConditional when not matched returns unmodified activation
	unsteered, sc, err := cs.ApplyConditional(act, Command{Vector: []float64{1, 0, 0}, Strength: 1.0}, orMiss)
	if err != nil || sc != 0.0 || unsteered[0] != act[0] {
		t.Errorf("ApplyConditional on miss failed: %v, %f, %v", unsteered, sc, err)
	}
}

func TestOracleCommitmentAndCoT(t *testing.T) {
	// Test all model families in DefaultCommitmentConfig
	for _, fam := range []string{"gemma", "qwen", "llama", "custom"} {
		cfg := DefaultCommitmentConfig(fam)
		if cfg.DefaultWindow <= 0 {
			t.Errorf("family %s has invalid DefaultWindow %d", fam, cfg.DefaultWindow)
		}
	}

	det := NewBehavioralCommitmentDetector(CommitmentWindowConfig{DefaultWindow: 0, PrefillDominant: false, PostCommitDecay: 0.6})
	if _, err := det.AnalyzeCommitmentWindow([]float64{0.5}, 1); err == nil {
		t.Error("expected error for < 2 scores")
	}

	scores := []float64{0.1, 0.2, 0.9, 0.92, 0.93}
	res, err := det.AnalyzeCommitmentWindow(scores, 4)
	if err != nil {
		t.Fatalf("AnalyzeCommitmentWindow error: %v", err)
	}
	if res.PeakPosition != 2 || math.Abs(res.MaxDerivative-0.7) > 1e-12 {
		t.Errorf("Commitment peak = (%d, %f), want (2, 0.7)", res.PeakPosition, res.MaxDerivative)
	}

	// BaseStrength <= 0 returns 0
	if m := det.ComputeSteeringMultiplier(2, 4, res, 0.0); m != 0.0 {
		t.Errorf("ComputeSteeringMultiplier(strength=0) = %f, want 0", m)
	}
	// Outside commitment window but inside prefill returns 0.75 * baseStrength
	narrowRes := CommitmentAnalysisResult{WindowStart: 2, WindowEnd: 2}
	if m := det.ComputeSteeringMultiplier(0, 4, narrowRes, 2.0); math.Abs(m-1.5) > 1e-12 {
		t.Errorf("ComputeSteeringMultiplier(outside window in prefill) = %f, want 1.5", m)
	}

	// CoTTracker CurrentPhase & Reset
	cot := NewCoTTracker(true)
	if cot.CurrentPhase() != PhaseDirect {
		t.Errorf("initial phase = %s, want DIRECT", cot.CurrentPhase())
	}
	cot.IngestToken("<think>", 0.7)
	if cot.CurrentPhase() != PhaseThinking {
		t.Errorf("phase after <think> = %s, want THINKING", cot.CurrentPhase())
	}
	cot.Reset()
	if cot.CurrentPhase() != PhaseDirect || cot.Summary().ThinkingTokens != 0 {
		t.Errorf("CoTTracker Reset did not clear state")
	}
}

func TestOracleCoherenceAndKL(t *testing.T) {
	ct := NewCoherenceTracker(0, 0) // uses defaults (15.0, 20)
	if ppl := ct.ComputePerplexity(); ppl != 1.0 {
		t.Errorf("empty perplexity = %f, want 1.0", ppl)
	}
	if mult := ct.AdaptiveSteeringMultiplier(); mult != 1.0 {
		t.Errorf("empty AdaptiveSteeringMultiplier = %f, want 1.0", mult)
	}

	// Window eviction check
	ctSmall := NewCoherenceTracker(10.0, 2)
	ctSmall.RecordTokenLogProb(-1.0)
	ctSmall.RecordTokenLogProb(-2.0)
	ctSmall.RecordTokenLogProb(-2.0) // evicts -1.0 -> avgNegLogProb = 2.0 -> exp(2)
	wantPPL := math.Exp(2.0)
	if math.Abs(ctSmall.ComputePerplexity()-wantPPL) > 1e-12 {
		t.Errorf("ComputePerplexity after window eviction = %.15f, want %.15f", ctSmall.ComputePerplexity(), wantPPL)
	}

	// AdjustAlphaForKL oracle values
	if got := AdjustAlphaForKL(0.0, 0.5, 0.2); got != 0.0 {
		t.Errorf("AdjustAlphaForKL(0) = %f, want 0", got)
	}
	// Default maxTargetKL = 0.20 when <= 0
	if got := AdjustAlphaForKL(1.0, 0.10, 0.0); got != 1.0 {
		t.Errorf("AdjustAlphaForKL within limit = %f, want 1.0", got)
	}
	// Proportional scaling: alpha=1.0, measuredKL=0.40, maxTargetKL=0.20 -> 0.5
	if got := AdjustAlphaForKL(1.0, 0.40, 0.20); math.Abs(got-0.5) > 1e-12 {
		t.Errorf("AdjustAlphaForKL proportional = %.15f, want 0.5", got)
	}
	// Floor at 5% when measuredKL is huge
	if got := AdjustAlphaForKL(2.0, 100.0, 0.20); math.Abs(got-0.10) > 1e-12 {
		t.Errorf("AdjustAlphaForKL floor = %.15f, want 0.10", got)
	}
}

func TestGoldenGCAD_And_KVCacheTracker(t *testing.T) {
	defCfg := DefaultGCADConfig()
	if defCfg.MaxDeltaNorm != 1.5 || defCfg.GateDecay != 0.95 {
		t.Errorf("unexpected DefaultGCADConfig: %+v", defCfg)
	}

	unsteered := []float64{1.0, 2.0}
	steered := []float64{4.0, 6.0} // delta = [3, 4], norm = 5.0
	cfg := GCADConfig{MaxDeltaNorm: 2.5, GateDecay: 0.8}

	// Scale = (2.5 / 5.0) * (0.8^2) = 0.5 * 0.64 = 0.32
	// Expected output = [1.0 + 0.32*3.0, 2.0 + 0.32*4.0] = [1.96, 3.28]
	got, cropped := GCADApply(steered, unsteered, 2, cfg)
	if !cropped {
		t.Error("expected cropped = true")
	}
	if math.Abs(got[0]-1.96) > 1e-12 || math.Abs(got[1]-3.28) > 1e-12 {
		t.Errorf("GCADApply = %v, want [1.96, 3.28]", got)
	}

	// Verify GCADApplyInPlace produces identical results
	inPlaceVec := []float64{4.0, 6.0}
	if !GCADApplyInPlace(inPlaceVec, unsteered, 2, cfg) {
		t.Error("expected GCADApplyInPlace cropped = true")
	}
	if math.Abs(inPlaceVec[0]-1.96) > 1e-12 || math.Abs(inPlaceVec[1]-3.28) > 1e-12 {
		t.Errorf("GCADApplyInPlace = %v, want [1.96, 3.28]", inPlaceVec)
	}

	// Edge cases for GCAD
	if _, c := GCADApply(nil, unsteered, 0, cfg); c {
		t.Error("expected false on empty steered")
	}
	if GCADApplyInPlace(nil, unsteered, 0, cfg) {
		t.Error("expected false on empty steered in-place")
	}

	// KVCacheAccumulationTracker edge cases & Reset
	tracker := NewKVCacheAccumulationTracker(0) // defaults to 15.0
	if _, _, err := tracker.RecordTurn(0, nil); err == nil {
		t.Error("expected error on empty steerVec")
	}
	if _, _, err := tracker.RecordTurn(0, []float64{0, 0}); err == nil {
		t.Error("expected error on zero-norm steerVec")
	}
	if _, _, err := tracker.RecordTurn(0, []float64{1, 0}); err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if _, _, err := tracker.RecordTurn(1, []float64{1, 0, 0}); err == nil {
		t.Error("expected error on dimension mismatch")
	}
	tracker.Reset()
	if len(tracker.History()) != 0 || tracker.AlarmCount() != 0 {
		t.Error("expected Reset to clear history and alarms")
	}
}

func TestOracleLoRA_ExactRank1Update(t *testing.T) {
	l := NewLoRASteerer(2, 1)
	// direction = [3, 4], norm = 5, strength = 4.0 -> sqrt(strength) = 2.0
	// A = [[6], [8]], B = [[0.6, 0.8]]
	// A * B = [[3.6, 4.8], [4.8, 6.4]]
	if err := l.ComputeLoRAWeights([]float64{3.0, 4.0}, 4.0); err != nil {
		t.Fatalf("ComputeLoRAWeights failed: %v", err)
	}

	W := [][]float64{
		{1.0, 2.0},
		{3.0, 4.0},
	}
	updated, err := l.ApplyLoRA(W)
	if err != nil {
		t.Fatalf("ApplyLoRA failed: %v", err)
	}

	want := [][]float64{
		{4.6, 6.8},
		{7.8, 10.4},
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(updated[i][j]-want[i][j]) > 1e-12 {
				t.Errorf("ApplyLoRA[%d][%d] = %.15f, want %.15f", i, j, updated[i][j], want[i][j])
			}
		}
	}

	restored, err := l.ReverseLoRA(updated)
	if err != nil {
		t.Fatalf("ReverseLoRA failed: %v", err)
	}
	for i := 0; i < 2; i++ {
		for j := 0; j < 2; j++ {
			if math.Abs(restored[i][j]-W[i][j]) > 1e-12 {
				t.Errorf("ReverseLoRA[%d][%d] = %.15f, want %.15f", i, j, restored[i][j], W[i][j])
			}
		}
	}
}
