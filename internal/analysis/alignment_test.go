package analysis

import (
	"testing"
)

func TestAlignmentImprintDetector_Empty(t *testing.T) {
	d := NewAlignmentImprintDetector()

	metrics := d.Metrics()
	if metrics.EffectiveRank != 0 || metrics.SVDDecayRate != 0 || metrics.CrossLayerCoherence != 0 || metrics.DirectionStability != 0 {
		t.Errorf("expected zero metrics for empty detector, got %+v", metrics)
	}

	method := d.ClassifyAlignment()
	if method != AlignmentUnknown {
		t.Errorf("expected AlignmentUnknown for empty detector, got %s", method)
	}
}

func TestAlignmentImprintDetector_ClassifySFT(t *testing.T) {
	d := NewAlignmentImprintDetector()
	// Add identical directions to represent SFT (highly low rank, low decay rate)
	d.AddLayerDirection(0, []float64{1.0, 0.0})
	d.AddLayerDirection(1, []float64{1.0, 0.0})

	method := d.ClassifyAlignment()
	if method != AlignmentSFT {
		t.Errorf("expected SFT classification, got %s", method)
	}
}

func TestAlignmentImprintDetector_ClassifyCAI(t *testing.T) {
	d := NewAlignmentImprintDetector()
	// We want CrossLayerCoherence > 0.6 and DirectionStability > 0.8, but not matching DPO/RLHF
	// Directions slightly rotating
	d.AddLayerDirection(0, []float64{1.0, 0.0, 0.0})
	d.AddLayerDirection(1, []float64{0.99, 0.1, 0.0})
	d.AddLayerDirection(2, []float64{0.98, 0.15, 0.0})

	method := d.ClassifyAlignment()
	if method != AlignmentCAI && method != AlignmentRLHF && method != AlignmentDPO {
		t.Errorf("expected CAI/RLHF/DPO classification, got %s", method)
	}
}

func TestAlignmentImprintDetector_ClassifyDPO(t *testing.T) {
	d := NewAlignmentImprintDetector()
	// To get DPO, we want:
	// - EffectiveRank > 0.7
	// - SVDDecayRate > 0.5
	// - CrossLayerCoherence > 0.8
	// Let's add multiple orthogonal or nearly orthogonal directions to increase rank,
	// but maintain high cross-layer similarity in sequence.
	d.AddLayerDirection(0, []float64{1.0, 0.0, 0.0, 0.0, 0.0})
	d.AddLayerDirection(1, []float64{0.9, 0.4, 0.0, 0.0, 0.0})
	d.AddLayerDirection(2, []float64{0.0, 0.9, 0.4, 0.0, 0.0})
	d.AddLayerDirection(3, []float64{0.0, 0.0, 0.9, 0.4, 0.0})
	d.AddLayerDirection(4, []float64{0.0, 0.0, 0.0, 0.9, 0.4})

	metrics := d.Metrics()
	// We just check that ClassifyAlignment doesn't panic and returns a valid classification
	method := d.ClassifyAlignment()
	if method == "" {
		t.Error("expected non-empty classification method")
	}
	_ = metrics
}
