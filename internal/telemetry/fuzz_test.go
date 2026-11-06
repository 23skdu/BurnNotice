package telemetry

import (
	"math"
	"testing"
)

func FuzzTelemetryAggregateAndSubmit(f *testing.F) {
	f.Add("llama", "8B", "whitened_svd", 0.05, 0.12, 0.99, "cuda")
	f.Add("qwen", "7B", "dim", 0.20, 0.05, 0.95, "rocm")
	f.Add("", "", "", 0.0, 0.0, 0.0, "")

	f.Fuzz(func(t *testing.T, arch, size, method string, refusal, kl, capRet float64, hw string) {
		if math.IsNaN(refusal) || math.IsInf(refusal, 0) ||
			math.IsNaN(kl) || math.IsInf(kl, 0) ||
			math.IsNaN(capRet) || math.IsInf(capRet, 0) {
			return
		}

		rep1 := Report{
			ModelArchitecture:   arch,
			ModelSize:           size,
			SteeringMethod:      method,
			RefusalRate:         refusal,
			KLDivergence:        kl,
			CapabilityRetention: capRet,
			Hardware:            hw,
		}
		rep2 := Report{
			ModelArchitecture:   arch + "-v2",
			ModelSize:           size,
			SteeringMethod:      method,
			RefusalRate:         refusal * 0.5,
			KLDivergence:        kl * 0.5,
			CapabilityRetention: capRet,
			Hardware:            hw,
		}

		agg := Aggregate([]Report{rep1, rep2})
		if agg.TotalReports != 2 {
			t.Fatalf("TotalReports = %d, want 2", agg.TotalReports)
		}
	})
}
