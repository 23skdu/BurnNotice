package steer

import (
	"math"
	"sync"
)

// CoherenceTracker monitors token perplexities to prevent activation-space lobotomy or output collapse.
type CoherenceTracker struct {
	mu                sync.RWMutex
	logProbs          []float64
	perplexityLimit   float64
	windowSize        int
}

// NewCoherenceTracker creates a tracker with target parameters.
func NewCoherenceTracker(perplexityLimit float64, windowSize int) *CoherenceTracker {
	if perplexityLimit <= 0 {
		perplexityLimit = 15.0 // Default critical perplexity threshold
	}
	if windowSize <= 0 {
		windowSize = 20 // Default sliding token history window
	}
	return &CoherenceTracker{
		logProbs:        make([]float64, 0),
		perplexityLimit: perplexityLimit,
		windowSize:      windowSize,
	}
}

// RecordTokenLogProb stores the latest generated token's log-probability.
func (ct *CoherenceTracker) RecordTokenLogProb(logProb float64) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	ct.logProbs = append(ct.logProbs, logProb)
	if len(ct.logProbs) > ct.windowSize {
		ct.logProbs = ct.logProbs[1:]
	}
}

// ComputePerplexity calculates the exponential of the average negative log probability.
func (ct *CoherenceTracker) ComputePerplexity() float64 {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	if len(ct.logProbs) == 0 {
		return 1.0
	}

	var sum float64
	for _, lp := range ct.logProbs {
		sum += lp
	}
	avgNegLogProb := -sum / float64(len(ct.logProbs))
	return math.Exp(avgNegLogProb)
}

// AdaptiveSteeringMultiplier calculates the scalar feedback factor to adjust steering strength alpha.
// If perplexity exceeds the safety limit, the steering strength is dynamically scaled down.
func (ct *CoherenceTracker) AdaptiveSteeringMultiplier() float64 {
	ppl := ct.ComputePerplexity()
	if ppl <= 1.0 {
		return 1.0
	}

	ct.mu.RLock()
	limit := ct.perplexityLimit
	ct.mu.RUnlock()

	if ppl >= limit {
		return 0.1 // Critical degradation: drop steering strength to a minimum safety mode
	}

	// Dynamic scaling: smooth linear decay from limit/2 to limit
	trigger := limit / 2.0
	if ppl < trigger {
		return 1.0
	}

	// Scaling factor decays smoothly from 1.0 to 0.1
	ratio := (ppl - trigger) / (limit - trigger)
	return 1.0 - (ratio * 0.9)
}

// Reset clears the stored token sequence history.
func (ct *CoherenceTracker) Reset() {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.logProbs = ct.logProbs[:0]
}

// AdjustAlphaForKL scales steering strength alpha dynamically based on harmless prompt KL divergence.
// If measuredKL exceeds maxTargetKL (e.g. 0.20), alpha is scaled down proportionally to protect capability.
func AdjustAlphaForKL(currentAlpha, measuredKL, maxTargetKL float64) float64 {
	if currentAlpha <= 0 {
		return 0
	}
	if maxTargetKL <= 0 {
		maxTargetKL = 0.20 // Default harmless target ceiling
	}
	if measuredKL <= maxTargetKL {
		return currentAlpha
	}
	// Scale down proportionally
	scaled := currentAlpha * (maxTargetKL / measuredKL)
	// Floor at 5% of currentAlpha to avoid total deactivation
	floor := currentAlpha * 0.05
	if scaled < floor {
		scaled = floor
	}
	return scaled
}
