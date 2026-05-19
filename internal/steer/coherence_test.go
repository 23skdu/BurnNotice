package steer

import (
	"math"
	"testing"
)

func TestCoherenceTracker_PerplexityCalculation(t *testing.T) {
	ct := NewCoherenceTracker(15.0, 5)

	// In Go: log(0.5) is approx -0.693147
	ct.RecordTokenLogProb(-0.693147)
	ct.RecordTokenLogProb(-0.693147)

	ppl := ct.ComputePerplexity()
	// Exp(-(-0.693147)) should be close to 2.0
	if math.Abs(ppl-2.0) > 1e-3 {
		t.Errorf("expected perplexity around 2.0, got %f", ppl)
	}
}

func TestCoherenceTracker_AdaptiveSteeringMultiplier(t *testing.T) {
	// Let's set a low limit of 10.0 for easier math
	ct := NewCoherenceTracker(10.0, 10)

	// 1. Safe region: perplexity <= 5.0 (which is limit/2)
	// LogProb of -1.6094379 corresponds to probability 0.2 (perplexity 5.0)
	ct.RecordTokenLogProb(-1.6094379)
	mult := ct.AdaptiveSteeringMultiplier()
	if mult != 1.0 {
		t.Errorf("expected safe multiplier 1.0, got %f", mult)
	}

	// 2. Critical region: perplexity >= 10.0
	// LogProb of -2.302585 corresponds to probability 0.1 (perplexity 10.0)
	ct.Reset()
	ct.RecordTokenLogProb(-2.302585)
	mult = ct.AdaptiveSteeringMultiplier()
	if math.Abs(mult-0.1) > 1e-3 {
		t.Errorf("expected critical multiplier 0.1, got %f", mult)
	}

	// 3. Dynamic scaling region: 5.0 < perplexity < 10.0
	// LogProb of -1.94591 corresponds to probability 0.143 (perplexity 7.0)
	ct.Reset()
	ct.RecordTokenLogProb(-1.94591)
	mult = ct.AdaptiveSteeringMultiplier()
	if mult <= 0.1 || mult >= 1.0 {
		t.Errorf("expected dynamic multiplier between 0.1 and 1.0, got %f", mult)
	}
}
