package scoring

import (
	"math"
	"math/rand/v2"
	"gonum.org/v1/gonum/floats"
	"sync"
)

// AsyncLedger represents a zero-copy stream or memory-mapped file containing asynchronous metrics from the CUDA plugins.
type AsyncLedger interface {
	PollActivation(layerIdx int) ([]float64, error)
}

// Scorer computes activation scores against registered direction vectors asynchronously.
type Scorer struct {
	mu         sync.RWMutex
	directions map[string][]float64
	scores     map[string]float64
	ledger     AsyncLedger

	// Robustness parameters
	NoiseScale float64 // Sigma for Gaussian noise injection S(a + eta)
	SkipRatio  float64 // Probability of skipping a layer per step (gradient masking)
}

// NewScorer creates a new asynchronous activation scorer.
func NewScorer(ledger AsyncLedger) *Scorer {
	return &Scorer{
		directions: make(map[string][]float64),
		scores:     make(map[string]float64),
		ledger:     ledger,
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
	noiseScale := s.NoiseScale
	skipRatio := s.SkipRatio
	s.mu.RUnlock()

	var workingAct []float64
	if noiseScale > 0 {
		workingAct = make([]float64, len(activation))
		for i, val := range activation {
			/* #nosec G404 */
			noise := rand.NormFloat64() * noiseScale
			workingAct[i] = val + noise
		}
	} else {
		workingAct = activation
	}

	actNorm := vecNorm(workingAct)
	if actNorm < 1e-10 {
		return nil
	}

	scores := make(map[string]float64, len(dirs))
	for name, dir := range dirs {
		/* #nosec G404 */
		if skipRatio > 0 && rand.Float64() < skipRatio {
			scores[name] = 0.5 // Masked baseline score
			continue
		}

		dot := vecDot(workingAct, dir)
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
	return floats.Norm(v, 2)
}

func vecDot(a, b []float64) float64 {
	return floats.Dot(a, b)
}
