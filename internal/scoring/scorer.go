package scoring

import (
	"gonum.org/v1/gonum/floats"
	"math"
	"math/rand/v2"
	"sync"
)

// AsyncLedger represents a zero-copy stream or memory-mapped file containing asynchronous metrics from the CUDA plugins.
type AsyncLedger interface {
	PollActivation(layerIdx int) ([]float64, error)
}

type registeredDirection struct {
	name string
	vec  []float64
	norm float64
}

// Scorer computes activation scores against registered direction vectors asynchronously.
type Scorer struct {
	mu         sync.RWMutex
	directions map[string][]float64
	dirList    []registeredDirection
	scores     map[string]float64
	ledger     AsyncLedger
	workPool   sync.Pool

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
	s.dirList = make([]registeredDirection, 0, len(s.directions))
	for k, v := range s.directions {
		s.dirList = append(s.dirList, registeredDirection{
			name: k,
			vec:  v,
			norm: vecNorm(v),
		})
	}
}

// Score computes cosine similarity scores for all registered directions.
func (s *Scorer) Score(activation []float64) map[string]float64 {
	s.mu.RLock()
	dirList := s.dirList
	noiseScale := s.NoiseScale
	skipRatio := s.SkipRatio
	s.mu.RUnlock()

	var workingAct []float64
	var pooledPtr *[]float64
	if noiseScale > 0 {
		if raw := s.workPool.Get(); raw != nil {
			ptr := raw.(*[]float64)
			if cap(*ptr) >= len(activation) {
				workingAct = (*ptr)[:len(activation)]
				pooledPtr = ptr
			}
		}
		if workingAct == nil {
			buf := make([]float64, len(activation))
			workingAct = buf
			pooledPtr = &buf
		}
		for i, val := range activation {
			/* #nosec G404 */
			noise := rand.NormFloat64() * noiseScale
			workingAct[i] = val + noise
		}
		defer func() {
			if pooledPtr != nil {
				s.workPool.Put(pooledPtr)
			}
		}()
	} else {
		workingAct = activation
	}

	actNorm := vecNorm(workingAct)
	if actNorm < 1e-10 {
		return nil
	}

	scores := make(map[string]float64, len(dirList))
	for _, d := range dirList {
		/* #nosec G404 */
		if skipRatio > 0 && rand.Float64() < skipRatio {
			scores[d.name] = 0.5 // Masked baseline score
			continue
		}

		if d.norm < 1e-10 {
			scores[d.name] = 0.0
			continue
		}
		dot := vecDot(workingAct, d.vec)
		cosine := dot / (actNorm * d.norm)
		scores[d.name] = 1.0 / (1.0 + math.Exp(-cosine*10)) // Sigmoid scaled
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
