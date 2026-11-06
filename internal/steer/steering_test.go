package steer

import (
	"math"
	"testing"
)

func TestActAdd(t *testing.T) {
	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := ActAdd(activation, direction, 2.0)

	expected := []float64{1, 4, 3}
	for i := range result {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("ActAdd[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestActAdd_ZeroAlpha(t *testing.T) {
	activation := []float64{1, 2, 3}
	direction := []float64{1, 0, 0}

	result := ActAdd(activation, direction, 0)

	for i := range result {
		if math.Abs(result[i]-activation[i]) > 1e-9 {
			t.Errorf("ActAdd with alpha=0[%d] = %f, want %f", i, result[i], activation[i])
		}
	}
}

func TestAngular_NormPreserving(t *testing.T) {
	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := Angular(activation, direction, 0.5)

	inputNorm := vecNorm(activation)
	outputNorm := vecNorm(result)

	if math.Abs(outputNorm-inputNorm) > 1e-6 {
		t.Errorf("Angular norm = %f, want %f", outputNorm, inputNorm)
	}
}

func TestAblate_Orthogonal(t *testing.T) {
	activation := []float64{3, 4, 5}
	direction := []float64{1, 0, 0}

	result := Ablate(activation, direction)

	dot := vecDot(result, direction)
	if math.Abs(dot) > 1e-9 {
		t.Errorf("Ablate result not orthogonal to direction: dot = %f", dot)
	}
}

func TestAblate_ZeroDirection(t *testing.T) {
	activation := []float64{1, 2, 3}
	direction := []float64{0, 0, 0}

	result := Ablate(activation, direction)

	for i := range result {
		if math.Abs(result[i]-activation[i]) > 1e-9 {
			t.Errorf("Ablate with zero direction[%d] = %f, want %f", i, result[i], activation[i])
		}
	}
}

func TestSelectiveSteerer_IsDiscriminative(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	activation := []float64{1, 0, 0}
	if !s.IsDiscriminative(0, activation) {
		t.Error("expected discriminative for truth-aligned activation")
	}

	activation = []float64{-1, 0, 0}
	if !s.IsDiscriminative(0, activation) {
		t.Error("expected discriminative for deception-aligned activation")
	}

	activation = []float64{0, 1, 0}
	if s.IsDiscriminative(0, activation) {
		t.Error("expected non-discriminative for orthogonal activation")
	}
}

func TestSelectiveSteerer_ComputeSteering_NonTarget(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0}},
		TargetLayers: map[int]bool{1: true},
	}

	cmd := s.ComputeSteering(0, []float64{-1, 0}, 0.5)
	if cmd != nil {
		t.Error("expected nil for non-target layer")
	}
}

func TestSelectiveSteerer_ComputeSteering_TruthLeaning(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	cmd := s.ComputeSteering(0, []float64{1, 0}, 0.5)
	if cmd != nil {
		t.Error("expected nil for truth-leaning activation (no steering needed)")
	}
}

func TestSelectiveSteerer_ComputeSteering_DeceptionLeaning(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	cmd := s.ComputeSteering(0, []float64{-1, 0}, 0.5)
	if cmd == nil {
		t.Fatal("expected steering command for deception-leaning activation")
	}
	if cmd.Method != MethodSelective {
		t.Errorf("Method = %v, want MethodSelective", cmd.Method)
	}
}

func TestNewSelectiveSteerer(t *testing.T) {
	truthDir := map[int][]float64{0: {1, 0}, 1: {0, 1}}
	deceptionDir := map[int][]float64{0: {-1, 0}, 1: {0, -1}}
	layers := []int{0, 1, 2}

	s := NewSelectiveSteerer(truthDir, deceptionDir, layers)

	if s == nil {
		t.Fatal("NewSelectiveSteerer returned nil")
	}
	if !s.TargetLayers[0] || !s.TargetLayers[1] || !s.TargetLayers[2] {
		t.Error("TargetLayers should contain 0, 1, 2")
	}
	if s.TargetLayers[3] {
		t.Error("TargetLayers should not contain 3")
	}
}

func TestSelectiveSteerer_ComputeSteering_NonDiscriminative(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {1, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	cmd := s.ComputeSteering(0, []float64{1, 0}, 0.5)
	if cmd != nil {
		t.Error("expected nil for non-discriminative layer (same direction)")
	}
}

func TestRotate2D_OrthogonalInput(t *testing.T) {
	h := []float64{0, 1, 0}
	v := []float64{1, 0, 0}

	result := rotate2D(h, v, 0.5)

	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
}

func TestRotate2D_ParallelInput(t *testing.T) {
	h := []float64{1, 0, 0}
	v := []float64{1, 0, 0}

	result := rotate2D(h, v, 0.5)

	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
}

func TestNormalizeVec_ZeroVector(t *testing.T) {
	v := []float64{0, 0, 0}
	result := normalizeVec(v)

	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
	for i := range result {
		if result[i] != 0 {
			t.Errorf("result[%d] = %f, want 0", i, result[i])
		}
	}
}

func TestNormalizeVec_UnitVector(t *testing.T) {
	v := []float64{1, 0, 0}
	result := normalizeVec(v)

	if result[0] != 1 || result[1] != 0 || result[2] != 0 {
		t.Errorf("normalizeVec(%v) = %v, want [1 0 0]", v, result)
	}
}

func TestAngular_ZeroActivation(t *testing.T) {
	result := Angular([]float64{0, 0, 0}, []float64{1, 0, 0}, 0.5)
	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
}

func TestAngular_ZeroDirection(t *testing.T) {
	result := Angular([]float64{1, 2, 3}, []float64{0, 0, 0}, 0.5)
	if len(result) != 3 {
		t.Fatalf("len(result) = %d, want 3", len(result))
	}
}

func TestSelectiveSteerer_ComputeSteering_MissingLayer(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	cmd := s.ComputeSteering(99, []float64{-1, 0}, 0.5)
	if cmd != nil {
		t.Error("expected nil for missing layer")
	}
}

func TestSelectiveSteerer_ComputeSteering_ClampedProjection(t *testing.T) {
	s := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0}},
		TargetLayers: map[int]bool{0: true},
	}

	cmd := s.ComputeSteering(0, []float64{-100, 0}, 0.5)
	if cmd == nil {
		t.Fatal("expected steering command for strongly deception-leaning activation")
	}
}

func TestSelectiveSteerer_NilsAndDimensionMismatches(t *testing.T) {
	// Nil maps
	sNil := &SelectiveSteerer{
		TruthDir:     nil,
		DeceptionDir: nil,
		TargetLayers: map[int]bool{0: true},
	}
	if sNil.IsDiscriminative(0, []float64{1}) {
		t.Error("IsDiscriminative should return false for nil maps")
	}
	if sNil.ComputeSteering(0, []float64{1}, 0.5) != nil {
		t.Error("ComputeSteering should return nil for nil maps")
	}

	// Mismatched vector dimensions
	sMismatch := &SelectiveSteerer{
		TruthDir:     map[int][]float64{0: {1, 0}},
		DeceptionDir: map[int][]float64{0: {-1, 0, 0}}, // mismatch
		TargetLayers: map[int]bool{0: true},
	}
	if sMismatch.IsDiscriminative(0, []float64{1, 0}) {
		t.Error("IsDiscriminative should return false for dimension mismatch")
	}
	if sMismatch.ComputeSteering(0, []float64{1, 0}, 0.5) != nil {
		t.Error("ComputeSteering should return nil for dimension mismatch")
	}
}

func TestAngular_ZeroOrthoNorm(t *testing.T) {
	// If orthoNorm < 1e-10 (activation and direction perfectly collinear)
	act := []float64{1, 2, 3}
	dir := []float64{1, 2, 3}
	res := Angular(act, dir, 0.5)
	if len(res) != 3 {
		t.Fatalf("len(res) = %d, want 3", len(res))
	}
}

func TestNormPreservingAblation_PreservesNormAndRemovesDirection(t *testing.T) {
	act := []float64{3.0, 4.0, 0.0}
	dir := []float64{1.0, 0.0, 0.0}

	origNorm := vecNorm(act) // sqrt(9 + 16) = 5.0
	res := NormPreservingAblation(act, dir, 1.0, 0.0)

	// 1. Result norm must equal original norm exactly (norm preservation)
	resNorm := vecNorm(res)
	if math.Abs(resNorm-origNorm) > 1e-6 {
		t.Errorf("MPOA norm = %f, want %f", resNorm, origNorm)
	}

	// 2. Result must be orthogonal to direction
	dot := vecDot(res, dir)
	if math.Abs(dot) > 1e-6 {
		t.Errorf("MPOA result not orthogonal: dot = %f", dot)
	}

	// 3. Expected vector: [0, 5, 0]
	if math.Abs(res[0]) > 1e-6 || math.Abs(res[1]-5.0) > 1e-6 {
		t.Errorf("unexpected MPOA vector: %v", res)
	}
}

func TestNormPreservingAblation_RegularizationAndAlpha(t *testing.T) {
	act := []float64{3.0, 4.0, 0.0}
	dir := []float64{1.0, 0.0, 0.0}

	// Alpha = 0 returns original
	res0 := NormPreservingAblation(act, dir, 0.0, 0.0)
	if math.Abs(res0[0]-act[0]) > 1e-6 {
		t.Errorf("alpha=0 did not return original: %v", res0)
	}

	// With regularization > 0, projection is softer
	resReg := NormPreservingAblation(act, dir, 1.0, 1.0)
	if math.Abs(vecNorm(resReg)-vecNorm(act)) > 1e-6 {
		t.Errorf("regularized MPOA should still preserve norm")
	}

	// Edge cases: zero activation and zero direction
	zeroAct := []float64{0.0, 0.0}
	if r := NormPreservingAblation(zeroAct, dir, 1.0, 0.0); len(r) != 2 {
		t.Errorf("failed on zero activation")
	}
	zeroDir := []float64{0.0, 0.0, 0.0}
	if r := NormPreservingAblation(act, zeroDir, 1.0, 0.0); math.Abs(r[0]-act[0]) > 1e-6 {
		t.Errorf("zero direction should return original activation")
	}
}

func TestApplyCommand(t *testing.T) {
	act := []float64{1.0, 2.0, 3.0}
	dir := []float64{0.0, 1.0, 0.0}

	// 1. MethodAddition
	cmdAdd := Command{Vector: dir, Strength: 2.0, Method: MethodAddition}
	resAdd := ApplyCommand(act, cmdAdd)
	if resAdd[1] != 4.0 {
		t.Errorf("MethodAddition failed: %v", resAdd)
	}

	// 2. MethodAngular
	cmdAng := Command{Vector: dir, Strength: 0.1, Method: MethodAngular}
	resAng := ApplyCommand(act, cmdAng)
	if math.Abs(vecNorm(resAng)-vecNorm(act)) > 1e-6 {
		t.Errorf("MethodAngular norm not preserved")
	}

	// 3. MethodAblation
	cmdAbl := Command{Vector: dir, Method: MethodAblation}
	resAbl := ApplyCommand(act, cmdAbl)
	if math.Abs(resAbl[1]) > 1e-6 {
		t.Errorf("MethodAblation failed: %v", resAbl)
	}

	// 4. MethodNormPreserving
	cmdMPOA := Command{
		Vector:              dir,
		Strength:            1.0,
		Method:              MethodNormPreserving,
		RegularizationAlpha: 0.01,
		AttentionStrength:   1.0,
		MLPStrength:         0.5,
	}
	resMPOA := ApplyCommand(act, cmdMPOA)
	if math.Abs(vecNorm(resMPOA)-vecNorm(act)) > 1e-6 {
		t.Errorf("MethodNormPreserving norm not preserved")
	}
}
