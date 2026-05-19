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
