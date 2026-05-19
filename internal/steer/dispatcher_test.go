package steer

import (
	"math"
	"testing"
)

func TestDispatcher_Apply_TargetLayer(t *testing.T) {
	d := NewDispatcher([]int{0, 1, 2}, map[int]Method{0: MethodAddition, 1: MethodAngular, 2: MethodAblation})

	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := d.Apply(0, activation, direction, 2.0)
	expected := []float64{1, 4, 3}
	for i := range result {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("Apply(Addition)[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestDispatcher_Apply_NonTargetLayer(t *testing.T) {
	d := NewDispatcher([]int{0, 1}, nil)

	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := d.Apply(99, activation, direction, 2.0)
	for i := range result {
		if math.Abs(result[i]-activation[i]) > 1e-9 {
			t.Errorf("Apply(non-target)[%d] = %f, want %f", i, result[i], activation[i])
		}
	}
}

func TestDispatcher_Apply_AngularMethod(t *testing.T) {
	d := NewDispatcher([]int{0}, map[int]Method{0: MethodAngular})

	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := d.Apply(0, activation, direction, 0.5)

	inputNorm := vecNorm(activation)
	outputNorm := vecNorm(result)
	if math.Abs(outputNorm-inputNorm) > 1e-6 {
		t.Errorf("Angular via dispatcher: norm = %f, want %f", outputNorm, inputNorm)
	}
}

func TestDispatcher_Apply_AblationMethod(t *testing.T) {
	d := NewDispatcher([]int{0}, map[int]Method{0: MethodAblation})

	activation := []float64{3, 4, 5}
	direction := []float64{1, 0, 0}

	result := d.Apply(0, activation, direction, 1.0)

	dot := vecDot(result, direction)
	if math.Abs(dot) > 1e-9 {
		t.Errorf("Ablation via dispatcher: not orthogonal, dot = %f", dot)
	}
}

func TestDispatcher_Apply_DefaultMethod(t *testing.T) {
	d := NewDispatcher([]int{0}, nil)

	activation := []float64{1, 2, 3}
	direction := []float64{1, 0, 0}

	result := d.Apply(0, activation, direction, 1.0)

	expected := []float64{2, 2, 3}
	for i := range result {
		if math.Abs(result[i]-expected[i]) > 1e-9 {
			t.Errorf("Default(Addition)[%d] = %f, want %f", i, result[i], expected[i])
		}
	}
}

func TestNewDispatcher(t *testing.T) {
	layers := []int{0, 1, 2, 3}
	methods := map[int]Method{0: MethodAddition, 2: MethodAngular}

	d := NewDispatcher(layers, methods)

	if !d.TargetLayers[0] || !d.TargetLayers[3] {
		t.Error("TargetLayers should contain all specified layers")
	}
	if d.TargetLayers[99] {
		t.Error("TargetLayers should not contain unspecified layers")
	}
	if d.Methods[0] != MethodAddition || d.Methods[2] != MethodAngular {
		t.Error("Methods should match provided map")
	}
}

func TestDispatcher_Apply_UnknownMethod(t *testing.T) {
	d := NewDispatcher([]int{0}, map[int]Method{0: Method(999)})

	activation := []float64{1, 2, 3}
	direction := []float64{0, 1, 0}

	result := d.Apply(0, activation, direction, 1.0)
	for i := range result {
		if math.Abs(result[i]-activation[i]) > 1e-9 {
			t.Errorf("Unknown method should return original activation[%d] = %f, want %f", i, result[i], activation[i])
		}
	}
}

func TestDispatcher_Apply_SelectiveMethod(t *testing.T) {
	d := NewDispatcher([]int{0}, map[int]Method{0: MethodSelective})

	// Vector in same direction (compliant): no steering
	activation := []float64{1, 2, 3}
	direction := []float64{1, 2, 3}

	result := d.Apply(0, activation, direction, 1.0)
	for i := range result {
		if math.Abs(result[i]-activation[i]) > 1e-9 {
			t.Errorf("Apply(Selective) steered compliant activation[%d] = %f, want %f", i, result[i], activation[i])
		}
	}

	// Vector in opposite direction (deceptive): should steer
	deceptiveActivation := []float64{-1, 0, -3}
	steeredResult := d.Apply(0, deceptiveActivation, direction, 0.5)

	diffCount := 0
	for i := range steeredResult {
		if math.Abs(steeredResult[i]-deceptiveActivation[i]) > 1e-4 {
			diffCount++
		}
	}
	if diffCount == 0 {
		t.Error("Apply(Selective) failed to steer unaligned activation")
	}
}
