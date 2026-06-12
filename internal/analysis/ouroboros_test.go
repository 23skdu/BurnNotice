package analysis

import (
	"testing"
)

func TestOuroborosDetector(t *testing.T) {
	d := NewOuroborosDetector()

	if d.NetReduction() != 0 {
		t.Errorf("empty detector NetReduction = %f, want 0", d.NetReduction())
	}
	if d.RequiresRefinement(0.5) {
		t.Error("empty detector should not require refinement")
	}
	if d.RefinementPasses(0.5, 3) != 0 {
		t.Error("empty detector should return 0 refinement passes")
	}

	d.RecordPre(0, 1.0)
	d.RecordPost(0, 0.8)

	d.RecordPre(1, 0.5)
	d.RecordPost(1, 0.6)

	comp := d.CompensatingLayers()
	if len(comp) != 1 || comp[0] != 1 {
		t.Errorf("expected compensating layers [1], got %v", comp)
	}

	expectedRed := (1.5 - 1.4) / 1.5
	if mathAbs(d.NetReduction()-expectedRed) > 1e-6 {
		t.Errorf("NetReduction = %f, want %f", d.NetReduction(), expectedRed)
	}

	if !d.RequiresRefinement(0.5) {
		t.Error("expected requires refinement for 0.066 < 0.5")
	}

	passes := d.RefinementPasses(0.5, 3)
	if passes != 1 {
		t.Errorf("RefinementPasses = %d, want 1", passes)
	}

	// Test passes bounds
	if d.RefinementPasses(0.5, 0) != 1 { // clamp passes < 1 makes it 1
		t.Errorf("RefinementPasses(0.5, 0) = %d, want 1", d.RefinementPasses(0.5, 0))
	}

	// Test with no requires refinement
	if d.RefinementPasses(0.01, 3) != 0 {
		t.Errorf("RefinementPasses with no refinement required = %d, want 0", d.RefinementPasses(0.01, 3))
	}

	// Test with no compensating layers but requires refinement
	d2 := NewOuroborosDetector()
	d2.RecordPre(0, 1.0)
	d2.RecordPost(0, 0.9) // reduction = 10%
	if d2.RefinementPasses(0.5, 3) != 0 {
		t.Errorf("expected 0 passes when no compensating layers, got %d", d2.RefinementPasses(0.5, 3))
	}

	d.Reset()
	if len(d.CompensatingLayers()) != 0 {
		t.Error("expected 0 compensating layers after Reset")
	}
}

func mathAbs(x float64) float64 {
	if x < 0 {
		return -x
	}
	return x
}
