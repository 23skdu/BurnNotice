package compliance

import (
	"math"
	"testing"
)

func TestCapabilityEvaluator_Evaluate(t *testing.T) {
	ce := NewCapabilityEvaluator()

	t.Run("Safe Degradation", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.76}, // 5% degradation
			{Name: "GSM8K", Baseline: 0.90, Current: 0.85}, // 5.5% degradation
		}

		deg, safe := ce.Evaluate(benchmarks)
		if !safe {
			t.Errorf("expected benchmarks to be evaluated as safe, got safe = %t", safe)
		}
		if deg > 0.06 {
			t.Errorf("unexpected degradation calculation: %f", deg)
		}
	})

	t.Run("Unsafe Degradation", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.60}, // 25% degradation
			{Name: "GSM8K", Baseline: 0.90, Current: 0.80}, // 11% degradation
		}

		_, safe := ce.Evaluate(benchmarks)
		if safe {
			t.Errorf("expected benchmarks to be flagged as unsafe")
		}
	})
}

func TestSteeringController_UpdateSteer(t *testing.T) {
	sc := NewSteeringController(2.0)

	t.Run("Safe Path", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.78},
		}

		strength, rolledBack := sc.UpdateSteer(benchmarks)
		if rolledBack {
			t.Errorf("expected no rollback on safe path")
		}
		if math.Abs(strength-2.0) > 1e-6 {
			t.Errorf("expected strength to remain at base 2.0, got %f", strength)
		}
	})

	t.Run("Trigger Rollback Path", func(t *testing.T) {
		benchmarks := []CapabilityBenchmark{
			{Name: "MMLU", Baseline: 0.80, Current: 0.40}, // 50% degradation
		}

		strength, rolledBack := sc.UpdateSteer(benchmarks)
		if !rolledBack {
			t.Errorf("expected automated rollback to trigger on unsafe degradation")
		}
		expectedStrength := 2.0 * (1.0 - 0.50) // 1.0
		if math.Abs(strength-expectedStrength) > 1e-6 {
			t.Errorf("expected strength rolled back to %f, got %f", expectedStrength, strength)
		}
	})
}
