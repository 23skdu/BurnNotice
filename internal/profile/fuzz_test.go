package profile

import (
	"math"
	"testing"
)

func FuzzDetectModelFamilyAndPrompt(f *testing.F) {
	f.Add("meta-llama/Llama-3-8B", []byte(`{"model_type":"llama","num_hidden_layers":32}`), "sys", "user")
	f.Add("Qwen/Qwen2.5-7B", []byte(`{"architectures":["Qwen2ForCausalLM"]}`), "", "hello")
	f.Add("unknown", []byte(`invalid`), "system", "prompt")

	f.Fuzz(func(t *testing.T, modelID string, configJSON []byte, sys, user string) {
		spec := DetectModelFamily(modelID, configJSON, []string{modelID})
		formatted := FormatPromptForFamily(spec.Family, sys, user)
		if formatted == "" {
			t.Fatal("FormatPromptForFamily returned empty string")
		}
	})
}

func FuzzImportProfileAndDrift(f *testing.F) {
	f.Add([]byte(`{"model_id":"test"}`), "json", 0.5, 4)
	f.Add([]byte{1, 2, 3, 4}, "gob", 1.2, 8)
	f.Add([]byte{}, "xml", -0.5, 1)

	f.Fuzz(func(t *testing.T, data []byte, format string, driftVal float64, coneDim int) {
		_, _ = ImportProfile(data, format)
		if !math.IsNaN(driftVal) && !math.IsInf(driftVal, 0) {
			dd := NewDriftDetector(5, 1.5)
			_ = dd.Record(driftVal)
			_ = dd.Record(driftVal * 2)
			_ = AdjustThresholdForCone(0.5, coneDim)
			_ = RecommendedINLPIterations(coneDim)
		}
	})
}
