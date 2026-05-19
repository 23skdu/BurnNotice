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

