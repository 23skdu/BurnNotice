package compliance

// CapabilityBenchmark represents a single capability evaluation metric (e.g. MMLU, GSM8K).
type CapabilityBenchmark struct {
	Name     string
	Baseline float64 // Original performance before steering
	Current  float64 // Performance after applying steering
}

// CapabilityEvaluator validates model capability degradation bounds (max 15%).
type CapabilityEvaluator struct {
	MaxDegradation float64 // e.g. 0.15 (15%)
}

// NewCapabilityEvaluator creates a new evaluator.
func NewCapabilityEvaluator() *CapabilityEvaluator {
	return &CapabilityEvaluator{
		MaxDegradation: 0.15,
	}
}

// Evaluate checks the benchmarks and returns average degradation and safety status.
func (ce *CapabilityEvaluator) Evaluate(benchmarks []CapabilityBenchmark) (float64, bool) {
	if len(benchmarks) == 0 {
		return 0.0, true
	}

	var totalDegradation float64
	for _, b := range benchmarks {
		if b.Baseline <= 0 {
			continue
		}
		// Compute degradation ratio
		deg := (b.Baseline - b.Current) / b.Baseline
		if deg < 0 {
			deg = 0.0 // performance improved or stayed same
		}
		totalDegradation += deg
	}

	avgDegradation := totalDegradation / float64(len(benchmarks))
	safe := avgDegradation <= ce.MaxDegradation

	return avgDegradation, safe
}

// SteeringController dynamically manages steering strength to respect capability bounds.
type SteeringController struct {
	BaseStrength   float64
	ActiveStrength float64
	Evaluator      *CapabilityEvaluator
}

// NewSteeringController creates a controller with the given base steering strength.
func NewSteeringController(base float64) *SteeringController {
	return &SteeringController{
		BaseStrength:   base,
		ActiveStrength: base,
		Evaluator:      NewCapabilityEvaluator(),
	}
}

// UpdateSteering assesses capability health and rolls back active strength if degradation exceeds safety bounds.
func (sc *SteeringController) UpdateSteer(benchmarks []CapabilityBenchmark) (float64, bool) {
	avgDeg, safe := sc.Evaluator.Evaluate(benchmarks)
	if !safe {
		// Roll back active steering strength proportionally to degradation
		factor := 1.0 - avgDeg
		if factor < 0 {
			factor = 0
		}
		sc.ActiveStrength = sc.BaseStrength * factor
		return sc.ActiveStrength, true // Alert triggered and rollback applied
	}

	// Keep base strength if safe
	sc.ActiveStrength = sc.BaseStrength
	return sc.ActiveStrength, false
}
