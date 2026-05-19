package engine

// MoEEngine wraps a base inference engine with per-expert activation extraction.
type MoEEngine struct {
	numExperts   int
	topK         int
	expertDims   []int
	routingCache map[int][]float64 // layer -> routing weights
}

// NewMoEEngine creates a MoE engine.
func NewMoEEngine(numExperts, topK int, expertDims []int) *MoEEngine {
	if len(expertDims) != numExperts {
		panic("expertDims length must match numExperts")
	}
	return &MoEEngine{
		numExperts:   numExperts,
		topK:         topK,
		expertDims:   expertDims,
		routingCache: make(map[int][]float64),
	}
}

// DetectRouting computes which experts are active for a given activation.
// Returns expert indices and their routing weights.
func (e *MoEEngine) DetectRouting(layerIdx int, activation []float64) ([]int, []float64) {
	// Simulate routing: use activation norm to determine expert selection
	routing := e.routingCache[layerIdx]
	if routing == nil {
		routing = make([]float64, e.numExperts)
		for i := 0; i < e.numExperts; i++ {
			routing[i] = 1.0 / float64(e.numExperts) // uniform
		}
		e.routingCache[layerIdx] = routing
	}

	// Select top-K experts
	type expertScore struct {
		idx   int
		score float64
	}
	scores := make([]expertScore, e.numExperts)
	for i := 0; i < e.numExperts; i++ {
		scores[i] = expertScore{i, routing[i]}
	}

	// Simple bubble sort for top-K
	for i := 0; i < e.numExperts-1; i++ {
		for j := 0; j < e.numExperts-i-1; j++ {
			if scores[j].score < scores[j+1].score {
				scores[j], scores[j+1] = scores[j+1], scores[j]
			}
		}
	}

	topK := e.topK
	if topK > e.numExperts {
		topK = e.numExperts
	}

	indices := make([]int, topK)
	weights := make([]float64, topK)
	for i := 0; i < topK; i++ {
		indices[i] = scores[i].idx
		weights[i] = scores[i].score
	}

	return indices, weights
}

// ExpertActivation extracts the activation for a specific expert.
func (e *MoEEngine) ExpertActivation(activation []float64, expertIdx int) []float64 {
	if expertIdx < 0 || expertIdx >= e.numExperts {
		return nil
	}
	dim := e.expertDims[expertIdx]
	if dim > len(activation) {
		dim = len(activation)
	}
	result := make([]float64, dim)
	copy(result, activation[:dim])
	return result
}

// NumExperts returns the number of experts.
func (e *MoEEngine) NumExperts() int {
	return e.numExperts
}

// ExpertDim returns the dimension of a specific expert.
func (e *MoEEngine) ExpertDim(expertIdx int) int {
	if expertIdx < 0 || expertIdx >= e.numExperts {
		return 0
	}
	return e.expertDims[expertIdx]
}

// UpdateRouting updates the routing weights for a layer.
func (e *MoEEngine) UpdateRouting(layerIdx int, weights []float64) {
	if len(weights) != e.numExperts {
		return
	}
	e.routingCache[layerIdx] = make([]float64, len(weights))
	copy(e.routingCache[layerIdx], weights)
}
