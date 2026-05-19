package scoring

import (
	"math"
	"sync"
)

// Scorer computes activation scores against registered direction vectors.
type Scorer struct {
	mu         sync.RWMutex
	directions map[string][]float64
	scores     map[string]float64
}

// NewScorer creates a new activation scorer.
func NewScorer() *Scorer {
	return &Scorer{
		directions: make(map[string][]float64),
		scores:     make(map[string]float64),
	}
}

// RegisterDirection stores a direction vector for scoring.
func (s *Scorer) RegisterDirection(name string, vec []float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.directions[name] = vec
}

// Score computes cosine similarity scores for all registered directions.
func (s *Scorer) Score(activation []float64) map[string]float64 {
	s.mu.RLock()
	dirs := make(map[string][]float64, len(s.directions))
	for k, v := range s.directions {
		dirs[k] = v
	}
	s.mu.RUnlock()

	actNorm := vecNorm(activation)
	if actNorm < 1e-10 {
		return nil
	}

	scores := make(map[string]float64, len(dirs))
	for name, dir := range dirs {
		dot := vecDot(activation, dir)
		dirNorm := vecNorm(dir)
		if dirNorm < 1e-10 {
			scores[name] = 0.0
			continue
		}
		cosine := dot / (actNorm * dirNorm)
		scores[name] = 1.0 / (1.0 + math.Exp(-cosine*10)) // Sigmoid scaled
	}

	s.mu.Lock()
	s.scores = scores
	s.mu.Unlock()

	return scores
}

// ScoreSubspace computes the aggregate projection energy onto a subspace basis.
func (s *Scorer) ScoreSubspace(activation []float64, basis [][]float64) float64 {
	var energy float64
	for _, v := range basis {
		proj := vecDot(activation, v)
		energy += proj * proj
	}
	return math.Sqrt(energy)
}

// GetScore returns the last computed score for a direction.
func (s *Scorer) GetScore(name string) float64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.scores[name]
}

func vecNorm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func vecDot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}
