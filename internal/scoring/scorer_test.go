package scoring

import (
	"math"
	"testing"
)

func TestScorer_RegisterAndScore(t *testing.T) {
	s := NewScorer()
	s.RegisterDirection("truth", []float64{1, 0, 0})
	s.RegisterDirection("deception", []float64{-1, 0, 0})

	scores := s.Score([]float64{1, 0, 0})

	if scores == nil {
		t.Fatal("Score returned nil")
	}

	truthScore := scores["truth"]
	deceptionScore := scores["deception"]

	if truthScore <= deceptionScore {
		t.Errorf("truth score (%f) should be > deception score (%f) for truth-aligned activation", truthScore, deceptionScore)
	}

	if truthScore < 0.5 {
		t.Errorf("truth score = %f, want > 0.5", truthScore)
	}
}

func TestScorer_ZeroActivation(t *testing.T) {
	s := NewScorer()
	s.RegisterDirection("truth", []float64{1, 0, 0})

	scores := s.Score([]float64{0, 0, 0})

	if scores != nil {
		t.Errorf("Score(zero) = %v, want nil", scores)
	}
}

func TestScorer_NoDirections(t *testing.T) {
	s := NewScorer()
	scores := s.Score([]float64{1, 2, 3})

	if scores == nil {
		t.Fatal("Score returned nil with no directions")
	}
	if len(scores) != 0 {
		t.Errorf("len(scores) = %d, want 0", len(scores))
	}
}

func TestScorer_GetScore(t *testing.T) {
	s := NewScorer()
	s.RegisterDirection("test", []float64{1, 0})

	scores := s.Score([]float64{1, 0})

	// GetScore returns the value from the scores map returned by Score
	if scores["test"] == 0 {
		t.Error("Score returned 0 for aligned direction")
	}

	missing := s.GetScore("nonexistent")
	if missing != 0 {
		t.Errorf("GetScore(nonexistent) = %f, want 0", missing)
	}
}

func TestScorer_SubspaceScoring(t *testing.T) {
	s := NewScorer()

	basis := [][]float64{
		{1, 0, 0},
		{0, 1, 0},
	}

	activation := []float64{3, 4, 0}
	energy := s.ScoreSubspace(activation, basis)

	expected := math.Sqrt(3*3 + 4*4)
	if math.Abs(energy-expected) > 1e-9 {
		t.Errorf("ScoreSubspace = %f, want %f", energy, expected)
	}
}

func TestScorer_SubspaceOrthogonal(t *testing.T) {
	s := NewScorer()

	basis := [][]float64{{1, 0, 0}}
	activation := []float64{0, 5, 0}

	energy := s.ScoreSubspace(activation, basis)

	if energy > 1e-9 {
		t.Errorf("ScoreSubspace(orthogonal) = %f, want ~0", energy)
	}
}

func TestScorer_ConcurrentAccess(t *testing.T) {
	s := NewScorer()
	s.RegisterDirection("test", []float64{1, 0, 0})

	done := make(chan bool, 10)
	for i := 0; i < 10; i++ {
		go func() {
			for j := 0; j < 100; j++ {
				s.Score([]float64{float64(j), 0, 0})
			}
			done <- true
		}()
	}

	for i := 0; i < 10; i++ {
		<-done
	}
}

func TestScorer_ZeroNormDirection(t *testing.T) {
	s := NewScorer()
	s.RegisterDirection("zero", []float64{0, 0, 0})
	
	scores := s.Score([]float64{1, 0, 0})
	if scores == nil {
		t.Fatal("expected scores, got nil")
	}
	
	if val, ok := scores["zero"]; !ok || val != 0.0 {
		t.Errorf("expected score for zero direction to be 0.0, got %v (ok=%v)", val, ok)
	}
}

func TestAggregator_Basic(t *testing.T) {
	agg := NewAggregator()
	
	if agg.TokenCount() != 0 {
		t.Errorf("expected TokenCount 0, got %d", agg.TokenCount())
	}
	
	metrics := agg.Compute()
	if metrics.TokenCount != 0 {
		t.Errorf("expected ResponseMetrics.TokenCount 0, got %d", metrics.TokenCount)
	}
	
	agg.AddToken("token1", map[string]float64{"deception": 0.5, "truth": 0.8})
	agg.AddToken("token2", map[string]float64{"deception": 0.7, "truth": 0.6})
	
	if agg.TokenCount() != 2 {
		t.Errorf("expected TokenCount 2, got %d", agg.TokenCount())
	}
	
	metrics = agg.Compute()
	if metrics.TokenCount != 2 {
		t.Errorf("expected ResponseMetrics.TokenCount 2, got %d", metrics.TokenCount)
	}
	
	// Check means
	if math.Abs(metrics.MeanScores["deception"]-0.6) > 1e-9 {
		t.Errorf("expected mean deception 0.6, got %f", metrics.MeanScores["deception"])
	}
	if math.Abs(metrics.MeanScores["truth"]-0.7) > 1e-9 {
		t.Errorf("expected mean truth 0.7, got %f", metrics.MeanScores["truth"])
	}
	
	// Check maxes
	if metrics.MaxScores["deception"] != 0.7 {
		t.Errorf("expected max deception 0.7, got %f", metrics.MaxScores["deception"])
	}
	if metrics.MaxScores["truth"] != 0.8 {
		t.Errorf("expected max truth 0.8, got %f", metrics.MaxScores["truth"])
	}
	
	// Test Reset
	agg.Reset()
	if agg.TokenCount() != 0 {
		t.Errorf("expected TokenCount 0 after Reset, got %d", agg.TokenCount())
	}
}

func TestAggregator_ComputeAUC(t *testing.T) {
	agg := NewAggregator()
	
	// AUC with < 2 elements should return 0.0
	agg.AddToken("t1", map[string]float64{"m1": 0.5})
	m := agg.Compute()
	if m.AUC["m1"] != 0.0 {
		t.Errorf("expected AUC 0.0 for 1 token, got %f", m.AUC["m1"])
	}
	
	agg.Reset()
	agg.AddToken("t1", map[string]float64{"m1": 0.2})
	agg.AddToken("t2", map[string]float64{"m1": 0.4})
	agg.AddToken("t3", map[string]float64{"m1": 0.8})
	
	m = agg.Compute()
	auc := m.AUC["m1"]
	// Sorted: 0.2, 0.4, 0.8
	// Interval 1: [0.2, 0.4] -> width = 0.2, height = 0.3 -> Area = 0.06
	// Interval 2: [0.4, 0.8] -> width = 0.4, height = 0.6 -> Area = 0.24
	// Total Area = 0.30
	// Normalized = 0.30 / 3 = 0.10
	if math.Abs(auc-0.1) > 1e-9 {
		t.Errorf("expected AUC 0.1, got %f", auc)
	}
}
