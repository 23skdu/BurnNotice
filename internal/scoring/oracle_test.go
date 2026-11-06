package scoring

import (
	"math"
	"testing"
)

func TestOracle_ScorerGoldenFixtures(t *testing.T) {
	s := NewScorer(nil)
	s.RegisterDirection("truth", []float64{1.0, 0.0, 0.0, 0.0})
	s.RegisterDirection("deception", []float64{-1.0, 0.0, 0.0, 0.0})
	s.RegisterDirection("diagonal", []float64{1.0, 1.0, 0.0, 0.0})

	fixtures := []struct {
		name      string
		act       []float64
		wantTruth float64
		wantDecep float64
		wantDiag  float64
	}{
		{
			name:      "aligned_truth",
			act:       []float64{2.0, 0.0, 0.0, 0.0},
			wantTruth: 1.0 / (1.0 + math.Exp(-10.0)),
			wantDecep: 1.0 / (1.0 + math.Exp(10.0)),
			wantDiag:  1.0 / (1.0 + math.Exp(-10.0/math.Sqrt2)),
		},
		{
			name:      "aligned_deception",
			act:       []float64{-3.0, 0.0, 0.0, 0.0},
			wantTruth: 1.0 / (1.0 + math.Exp(10.0)),
			wantDecep: 1.0 / (1.0 + math.Exp(-10.0)),
			wantDiag:  1.0 / (1.0 + math.Exp(10.0/math.Sqrt2)),
		},
		{
			name:      "orthogonal",
			act:       []float64{0.0, 0.0, 5.0, 0.0},
			wantTruth: 0.5,
			wantDecep: 0.5,
			wantDiag:  0.5,
		},
	}

	for _, tc := range fixtures {
		t.Run(tc.name, func(t *testing.T) {
			scores := s.Score(tc.act)
			if scores == nil {
				t.Fatalf("expected non-nil scores")
			}
			if math.Abs(scores["truth"]-tc.wantTruth) > 1e-12 {
				t.Errorf("truth = %.15f, want %.15f", scores["truth"], tc.wantTruth)
			}
			if math.Abs(scores["deception"]-tc.wantDecep) > 1e-12 {
				t.Errorf("deception = %.15f, want %.15f", scores["deception"], tc.wantDecep)
			}
			if math.Abs(scores["diagonal"]-tc.wantDiag) > 1e-12 {
				t.Errorf("diagonal = %.15f, want %.15f", scores["diagonal"], tc.wantDiag)
			}
		})
	}
}

func TestOracle_SubspaceAndAggregatorDeterminism(t *testing.T) {
	s := NewScorer(nil)
	basis := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
	}
	act := []float64{3.0, 4.0, 12.0, 0.0}
	energy1 := s.ScoreSubspace(act, basis)
	energy2 := s.ScoreSubspace(act, basis)
	if energy1 != energy2 || math.Abs(energy1-5.0) > 1e-12 {
		t.Fatalf("ScoreSubspace oracle mismatch: %v vs %v (want 5.0)", energy1, energy2)
	}

	agg := NewAggregator()
	agg.AddToken("tok1", map[string]float64{"deception": 0.1, "refusal": 0.4})
	agg.AddToken("tok2", map[string]float64{"deception": 0.5, "refusal": 0.6})
	agg.AddToken("tok3", map[string]float64{"deception": 0.9, "refusal": 0.8})

	m1 := agg.Compute()
	m2 := agg.Compute()
	if m1.MeanScores["deception"] != m2.MeanScores["deception"] || math.Abs(m1.MeanScores["deception"]-0.5) > 1e-12 {
		t.Errorf("deterministic MeanScores mismatch: %v vs %v", m1.MeanScores["deception"], m2.MeanScores["deception"])
	}
	if m1.AUC["deception"] != m2.AUC["deception"] {
		t.Errorf("deterministic AUC mismatch: %v vs %v", m1.AUC["deception"], m2.AUC["deception"])
	}
}
