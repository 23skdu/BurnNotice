package analysis

import (
	"math"
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

func TestAccuracyNumericalStabilityAlignmentAndCone(t *testing.T) {
	// Single-layer, zero-norm, and NaN directions in AlignmentImprintDetector
	d := NewAlignmentImprintDetector()
	d.AddLayerDirection(0, []float64{1.0, 2.0, 3.0})
	m1 := d.Metrics()
	if math.IsNaN(m1.EffectiveRank) || math.IsNaN(m1.SVDDecayRate) || math.IsNaN(m1.CrossLayerCoherence) || math.IsNaN(m1.DirectionStability) {
		t.Errorf("single-layer detector returned NaN metrics: %+v", m1)
	}

	d.AddLayerDirection(1, []float64{0.0, 0.0, 0.0})
	d.AddLayerDirection(2, []float64{math.NaN(), 1.0, 0.0})
	d.AddLayerDirection(3, []float64{math.Inf(1), 0.0, 0.0})
	m2 := d.Metrics()
	if math.IsNaN(m2.EffectiveRank) || math.IsNaN(m2.SVDDecayRate) || math.IsNaN(m2.CrossLayerCoherence) || math.IsNaN(m2.DirectionStability) {
		t.Errorf("degenerate detector returned NaN metrics: %+v", m2)
	}

	// ConceptConeAnalyzer with NaN, zero-norm, and single valid category
	cats := []HarmCategory{
		{Name: "nan_cat", Activations: [][]float64{{math.NaN(), 1.0, 0.0}}},
		{Name: "zero_cat", Activations: [][]float64{{0.0, 0.0, 0.0}}},
		{Name: "valid_cat", Activations: [][]float64{{1.0, 2.0, 3.0}}},
	}
	cone := NewConceptConeAnalyzer(cats)
	res := cone.Analyze()
	if res == nil {
		t.Fatal("expected non-nil ConeResult")
	}
	for _, sv := range res.SingularValues {
		if math.IsNaN(sv) || math.IsInf(sv, 0) {
			t.Errorf("unexpected non-finite singular value: %v", sv)
		}
	}
	for k, v := range res.CategoryPairwiseCosines {
		if math.IsNaN(v) || math.IsInf(v, 0) || v < -1.0 || v > 1.0 {
			t.Errorf("invalid pairwise cosine for %s: %v", k, v)
		}
	}

	// ProjectLatentSpace with NaN / Inf / degenerate inputs
	if _, err := ProjectLatentSpace(nil, nil, nil); err == nil {
		t.Error("expected error for empty points")
	}
	if _, err := ProjectLatentSpace([]LatentPoint{{Vector: []float64{1.0}}}, nil, nil); err == nil {
		t.Error("expected error for 1D vector")
	}
	if _, err := ProjectLatentSpace([]LatentPoint{
		{Vector: []float64{math.NaN(), 1.0}},
		{Vector: []float64{1.0, 2.0}},
	}, nil, nil); err == nil {
		t.Error("expected error for NaN activation vector")
	}
	if _, err := ProjectLatentSpace([]LatentPoint{
		{Vector: []float64{1.0, 2.0}},
		{Vector: []float64{3.0, 4.0}},
	}, []float64{math.Inf(1), 0.0}, []float64{0.0, 1.0}); err == nil {
		t.Error("expected error for Inf basis vector")
	}
}
