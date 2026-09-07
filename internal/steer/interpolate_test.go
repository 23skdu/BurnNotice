package steer

import (
	"math"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestFloatDirectionInterpolation_Basic(t *testing.T) {
	d0 := []float64{1.0, 0.0, 0.0}
	d1 := []float64{0.0, 1.0, 0.0}
	dirs := [][]float64{d0, d1}

	// At t = 0.5, equal mix of d0 and d1
	blend, err := FloatDirectionInterpolation(dirs, 0.5, 0.5)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	norm := floats.Norm(blend, 2)
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("expected unit norm, got %f", norm)
	}

	// Blend should have equal components along x and y, 0 along z
	if math.Abs(blend[0]-blend[1]) > 1e-6 {
		t.Errorf("expected blend[0] == blend[1], got %f vs %f", blend[0], blend[1])
	}
	if math.Abs(blend[2]) > 1e-6 {
		t.Errorf("expected blend[2] == 0, got %f", blend[2])
	}

	// At t = 0 with small sigma, blend[0] should dominate
	blend0, err := FloatDirectionInterpolation(dirs, 0.0, 0.2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if blend0[0] < 0.95 {
		t.Errorf("expected blend0[0] > 0.95, got %f", blend0[0])
	}

	// At t = 1 with small sigma, blend[1] should dominate
	blend1, err := FloatDirectionInterpolation(dirs, 1.0, 0.2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if blend1[1] < 0.95 {
		t.Errorf("expected blend1[1] > 0.95, got %f", blend1[1])
	}
}

func TestFloatDirectionInterpolation_SmoothMonotonicity(t *testing.T) {
	d0 := []float64{1.0, 0.0}
	d1 := []float64{0.0, 1.0}
	dirs := [][]float64{d0, d1}

	prevDot := 2.0
	steps := 10
	for i := 0; i <= steps; i++ {
		tVal := float64(i) / float64(steps)
		blend, err := FloatDirectionInterpolation(dirs, tVal, 1.0)
		if err != nil {
			t.Fatalf("unexpected error at t=%f: %v", tVal, err)
		}
		dot := floats.Dot(blend, d0)
		if dot > prevDot+1e-6 {
			t.Errorf("dot product with d0 should decrease monotonically: step %d t=%f dot=%f prev=%f", i, tVal, dot, prevDot)
		}
		prevDot = dot
	}
}

func TestFloatDirectionInterpolation_EdgeCases(t *testing.T) {
	// Empty
	_, err := FloatDirectionInterpolation(nil, 0, 1.0)
	if err == nil {
		t.Errorf("expected error for empty directions")
	}

	// Single direction
	single := [][]float64{{0.5, 0.5}}
	res, err := FloatDirectionInterpolation(single, 0, 1.0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res[0] != 0.5 || res[1] != 0.5 {
		t.Errorf("expected exact copy of single direction, got %v", res)
	}

	// Dimension mismatch
	mismatched := [][]float64{{1.0, 0.0}, {1.0, 0.0, 0.0}}
	_, err = FloatDirectionInterpolation(mismatched, 0.5, 1.0)
	if err == nil {
		t.Errorf("expected error for mismatched dimensions")
	}

	// Command integration
	primary := []float64{1.0, 0.0}
	basis := [][]float64{{0.0, 1.0}}
	cmdBlend, err := InterpolateDirectionFromCommand(primary, basis, 0.0)
	if err != nil || cmdBlend[0] != 1.0 {
		t.Errorf("mix=0 should return primary direction, got %v, err=%v", cmdBlend, err)
	}
}
