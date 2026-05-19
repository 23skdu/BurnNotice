package scoring

import (
	"math"
	"testing"
)

func TestAggregator_TemporalEMAAndDrift(t *testing.T) {
	a := NewAggregator()

	// Add sequence of scores
	a.AddToken("token1", map[string]float64{"deception": 0.3})
	a.AddToken("token2", map[string]float64{"deception": 0.4})
	a.AddToken("token3", map[string]float64{"deception": 0.8})
	a.AddToken("token4", map[string]float64{"deception": 0.9})
	a.AddToken("token5", map[string]float64{"deception": 0.95})

	t.Run("Compute EMA Sequence", func(t *testing.T) {
		ema := a.ComputeEMA("deception", 0.5)
		if len(ema) != 5 {
			t.Fatalf("expected 5 EMA values, got %d", len(ema))
		}

		// First EMA element should be initial score (0.3)
		if math.Abs(ema[0]-0.3) > 1e-6 {
			t.Errorf("expected first EMA score 0.3, got %f", ema[0])
		}

		// Second: 0.5 * 0.4 + 0.5 * 0.3 = 0.35
		if math.Abs(ema[1]-0.35) > 1e-6 {
			t.Errorf("expected second EMA score 0.35, got %f", ema[1])
		}
	})

	t.Run("Detect Drift Trajectory", func(t *testing.T) {
		// Should not detect drift for threshold 0.8 with 4 consecutive steps (we only have 3 steps above 0.8: token3, token4, token5)
		if a.DetectDriftTrajectory("deception", 0.7, 4) {
			t.Errorf("expected no drift detection for 4 consecutive steps above 0.7")
		}

		// Should detect drift for threshold 0.7 with 3 consecutive steps
		if !a.DetectDriftTrajectory("deception", 0.7, 3) {
			t.Errorf("expected drift detection for 3 consecutive steps above 0.7")
		}
	})
}
