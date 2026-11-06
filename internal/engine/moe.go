package engine

import (
	"fmt"
	"math"
	"sort"
)

// AsyncLedger represents a zero-copy stream or memory-mapped file containing asynchronous metrics from the CUDA plugins.
type AsyncLedger interface {
	PollActivation(layerIdx int) ([]float64, error)
}

// MoEEngine wraps a base inference engine with per-expert activation extraction asynchronously.
type MoEEngine struct {
	numExperts   int
	topK         int
	expertDims   []int
	routingCache map[int][]float64 // layer -> routing weights
	ledger       AsyncLedger
}

// NewMoEEngine creates an async MoE engine.
func NewMoEEngine(numExperts, topK int, expertDims []int, ledger AsyncLedger) *MoEEngine {
	if len(expertDims) != numExperts {
		panic("expertDims length must match numExperts")
	}
	return &MoEEngine{
		numExperts:   numExperts,
		topK:         topK,
		expertDims:   expertDims,
		routingCache: make(map[int][]float64),
		ledger:       ledger,
	}
}

// PollLedger queries the underlying asynchronous ledger for activation data at layerIdx.
func (e *MoEEngine) PollLedger(layerIdx int) ([]float64, error) {
	if e.ledger == nil {
		return nil, fmt.Errorf("no ledger configured")
	}
	return e.ledger.PollActivation(layerIdx)
}

// DetectRouting computes which experts are active for a given activation.
// If activation is empty and a ledger is configured, it attempts to poll the ledger.
// Returns expert indices and their routing weights.
func (e *MoEEngine) DetectRouting(layerIdx int, activation []float64) ([]int, []float64) {
	if len(activation) == 0 && e.ledger != nil {
		if polled, err := e.ledger.PollActivation(layerIdx); err == nil && len(polled) > 0 {
			activation = polled
		}
	}

	var totalSumSq float64
	expertSumSq := make([]float64, e.numExperts)

	start := 0
	for idx := 0; idx < e.numExperts; idx++ {
		dim := e.expertDims[idx]
		end := start + dim
		if end > len(activation) {
			end = len(activation)
		}

		var sumSq float64
		for i := start; i < end; i++ {
			sumSq += activation[i] * activation[i]
		}
		expertSumSq[idx] = sumSq
		totalSumSq += sumSq

		start = end
	}

	topK := e.topK
	if topK > e.numExperts {
		topK = e.numExperts
	}

	indices := make([]int, topK)
	weights := make([]float64, topK)

	if totalSumSq < 1e-10 {
		// Fallback to uniform 1/N distribution
		uniformWeight := 1.0 / float64(e.numExperts)
		for i := 0; i < topK; i++ {
			indices[i] = i
			weights[i] = uniformWeight
		}
		return indices, weights
	}

	type expertScore struct {
		index int
		score float64
	}
	scores := make([]expertScore, e.numExperts)
	for i := 0; i < e.numExperts; i++ {
		scores[i] = expertScore{index: i, score: expertSumSq[i]}
	}

	// Simple stable sort
	for i := 0; i < e.numExperts-1; i++ {
		for j := i + 1; j < e.numExperts; j++ {
			if scores[j].score > scores[i].score {
				scores[i], scores[j] = scores[j], scores[i]
			}
		}
	}

	for i := 0; i < topK; i++ {
		indices[i] = scores[i].index
		weights[i] = scores[i].score / totalSumSq
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

// DetectRoutingFromLogits computes top-k active experts and their normalized softmax weights from router logits.
func (e *MoEEngine) DetectRoutingFromLogits(logits []float64) ([]int, []float64) {
	if len(logits) != e.numExperts || e.numExperts == 0 {
		return nil, nil
	}
	topK := e.topK
	if topK > e.numExperts {
		topK = e.numExperts
	}
	type expertScore struct {
		index int
		logit float64
	}
	scores := make([]expertScore, e.numExperts)
	for i := 0; i < e.numExperts; i++ {
		scores[i] = expertScore{index: i, logit: logits[i]}
	}
	sort.Slice(scores, func(i, j int) bool {
		return scores[i].logit > scores[j].logit
	})

	maxLogit := scores[0].logit
	var sumExp float64
	exps := make([]float64, topK)
	for i := 0; i < topK; i++ {
		exps[i] = math.Exp(scores[i].logit - maxLogit)
		sumExp += exps[i]
	}

	indices := make([]int, topK)
	weights := make([]float64, topK)
	for i := 0; i < topK; i++ {
		indices[i] = scores[i].index
		if sumExp > 0 {
			weights[i] = exps[i] / sumExp
		} else {
			weights[i] = 1.0 / float64(topK)
		}
	}
	return indices, weights
}
