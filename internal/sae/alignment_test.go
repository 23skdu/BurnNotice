package sae

import (
	"math"
	"os"
	"path/filepath"
	"testing"
)

func TestSAELoader_LlamaScopeAndQwenSAE(t *testing.T) {
	// 1. Test DetectSAEArchitecture & MapSAEToModelFamily
	if fam := DetectSAEArchitecture("meta-llama/Llama-3.1-8B-Scope", nil, nil); fam != SAELlamaScope {
		t.Errorf("DetectSAEArchitecture(llama) = %s, want %s", fam, SAELlamaScope)
	}
	if fam := DetectSAEArchitecture("Qwen/Qwen2.5-7B-SAE", nil, nil); fam != SAEQwenSAE {
		t.Errorf("DetectSAEArchitecture(qwen) = %s, want %s", fam, SAEQwenSAE)
	}
	if fam := DetectSAEArchitecture("google/gemma-2-2b-scope", nil, nil); fam != SAEGemmaScope {
		t.Errorf("DetectSAEArchitecture(gemma) = %s, want %s", fam, SAEGemmaScope)
	}
	if fam := DetectSAEArchitecture("unknown", []byte(`{"sae_family":"llama-scope"}`), nil); fam != SAELlamaScope {
		t.Errorf("DetectSAEArchitecture(config llama) = %s, want %s", fam, SAELlamaScope)
	}
	if fam := DetectSAEArchitecture("unknown", []byte(`{"architecture":"qwen2"}`), nil); fam != SAEQwenSAE {
		t.Errorf("DetectSAEArchitecture(config qwen) = %s, want %s", fam, SAEQwenSAE)
	}
	if fam := DetectSAEArchitecture("unknown", []byte(`{"model_id":"gemma"}`), nil); fam != SAEGemmaScope {
		t.Errorf("DetectSAEArchitecture(config gemma) = %s, want %s", fam, SAEGemmaScope)
	}
	if fam := DetectSAEArchitecture("unknown", nil, []string{"encoder.weight", "b_pre"}); fam != SAELlamaScope {
		t.Errorf("DetectSAEArchitecture(tensor llama) = %s, want %s", fam, SAELlamaScope)
	}
	if fam := DetectSAEArchitecture("unknown", nil, []string{"w_enc", "pre_bias"}); fam != SAEQwenSAE {
		t.Errorf("DetectSAEArchitecture(tensor qwen) = %s, want %s", fam, SAEQwenSAE)
	}
	if fam := DetectSAEArchitecture("unknown", nil, []string{"W_enc", "threshold"}); fam != SAEGemmaScope {
		t.Errorf("DetectSAEArchitecture(tensor gemma) = %s, want %s", fam, SAEGemmaScope)
	}
	if fam := DetectSAEArchitecture("unknown", nil, nil); fam != SAEGeneric {
		t.Errorf("DetectSAEArchitecture(generic) = %s, want %s", fam, SAEGeneric)
	}

	if m := MapSAEToModelFamily(SAEGemmaScope); m != "gemma" {
		t.Errorf("MapSAEToModelFamily(gemma) = %s", m)
	}
	if m := MapSAEToModelFamily(SAELlamaScope); m != "llama" {
		t.Errorf("MapSAEToModelFamily(llama) = %s", m)
	}
	if m := MapSAEToModelFamily(SAEQwenSAE); m != "qwen" {
		t.Errorf("MapSAEToModelFamily(qwen) = %s", m)
	}
	if m := MapSAEToModelFamily(SAEGeneric); m != "unknown" {
		t.Errorf("MapSAEToModelFamily(generic) = %s", m)
	}

	// 2. Build mock Llama-Scope NPZ with encoder.weight, decoder.weight, encoder.bias, decoder.bias, b_pre, threshold
	encW := createMockNpy([]float32{1, 0, 0, 1}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }`)
	decW := createMockNpy([]float32{1, 0, 0, 1}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2, 2), }`)
	encB := createMockNpy([]float32{0, 0}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)
	decB := createMockNpy([]float32{0, 0}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)
	bPre := createMockNpy([]float32{0.5, 0.5}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)
	thresh := createMockNpy([]float32{0.2, 0.2}, `{'descr': '<f4', 'fortran_order': False, 'shape': (2,), }`)

	llamaNpz := filepath.Join(t.TempDir(), "llama_scope.npz")
	_ = os.WriteFile(llamaNpz, createMockNpz(map[string][]byte{
		"encoder.weight.npy": encW,
		"decoder.weight.npy": decW,
		"encoder.bias.npy":   encB,
		"decoder.bias.npy":   decB,
		"b_pre.npy":          bPre,
		"threshold.npy":      thresh,
	}), 0600)

	llamaLoader := &SAELoader{}
	if llamaLoader.Family() != SAEGeneric {
		t.Errorf("unloaded Family() = %s, want generic", llamaLoader.Family())
	}
	if err := llamaLoader.LoadLlamaScope(llamaNpz, ""); err != nil {
		t.Fatalf("LoadLlamaScope failed: %v", err)
	}
	if llamaLoader.Family() != SAELlamaScope {
		t.Errorf("llamaLoader.Family() = %s, want %s", llamaLoader.Family(), SAELlamaScope)
	}
	// With b_pre = [0.5, 0.5] and jump_threshold = [0.2, 0.2]:
	// Input [0.6, 2.5] -> x - b_pre = [0.1, 2.0] -> JumpReLU (>0.2) -> [0.0, 2.0]
	feats, err := llamaLoader.Encode([]float64{0.6, 2.5})
	if err != nil {
		t.Fatalf("llamaLoader.Encode failed: %v", err)
	}
	if feats[0] != 0.0 || math.Abs(feats[1]-2.0) > 1e-6 {
		t.Errorf("llamaLoader.Encode([0.6, 2.5]) = %v, want [0.0, 2.0]", feats)
	}

	// 3. Build mock Qwen-SAE NPZ with w_enc, w_dec, b_enc, b_dec
	qwenNpz := filepath.Join(t.TempDir(), "qwen_sae.npz")
	_ = os.WriteFile(qwenNpz, createMockNpz(map[string][]byte{
		"w_enc.npy": encW,
		"w_dec.npy": decW,
		"b_enc.npy": encB,
		"b_dec.npy": decB,
	}), 0600)

	qwenLoader := &SAELoader{}
	if err := qwenLoader.LoadQwenSAE(qwenNpz, ""); err != nil {
		t.Fatalf("LoadQwenSAE failed: %v", err)
	}
	if qwenLoader.Family() != SAEQwenSAE {
		t.Errorf("qwenLoader.Family() = %s, want %s", qwenLoader.Family(), SAEQwenSAE)
	}

	// 4. Test LoadDictionary auto-detection on JSON and NPZ
	autoLoader := &SAELoader{}
	if err := autoLoader.LoadDictionary(qwenNpz, ""); err != nil {
		t.Fatalf("LoadDictionary(qwenNpz) failed: %v", err)
	}
	if autoLoader.Family() != SAEQwenSAE {
		t.Errorf("autoLoader.Family() = %s, want %s", autoLoader.Family(), SAEQwenSAE)
	}

	jsonPath := filepath.Join(t.TempDir(), "llama_sae.json")
	_ = os.WriteFile(jsonPath, []byte(`{"encoder_weights":[[1,0],[0,1]],"decoder_weights":[[1,0],[0,1]],"encoder_bias":[0,0],"decoder_bias":[0,0],"feature_count":2,"input_dim":2}`), 0600)
	if err := autoLoader.LoadDictionary(jsonPath, ""); err != nil {
		t.Fatalf("LoadDictionary(jsonPath) failed: %v", err)
	}
	if autoLoader.Family() != SAELlamaScope {
		t.Errorf("autoLoader JSON Family() = %s, want %s", autoLoader.Family(), SAELlamaScope)
	}
}

func TestOracleSAEFeatureAbsorptionAndAlignment(t *testing.T) {
	// Construct a 3-feature SAE in R^2 where:
	// - Feature 0 is the primary deception feature aligned with [1, 0], but has a negative bias (-2.0)
	//   so it fails to fire on moderate [1, 0] inputs (absorbed!).
	// - Feature 1 is a child feature partially aligned with [1, 0] (decoder [0.8, 0.6]) with 0 bias,
	//   so it fires whenever Feature 0 is suppressed, absorbing the deception concept!
	// - Feature 2 is a monosemantic truth feature aligned with [0, 1] with 0 bias (never absorbed).
	saeDict := &SAELoader{}
	_ = saeDict.LoadFromMemory(&SAEWeights{
		Family: SAEGemmaScope,
		EncoderWeights: [][]float64{
			{1.0, 0.0},
			{0.8, 0.6},
			{0.0, 1.0},
		},
		DecoderWeights: [][]float64{
			{1.0, 0.0},
			{0.8, 0.6},
			{0.0, 1.0},
		},
		EncoderBias:  []float64{-2.0, 0.0, 0.0},
		DecoderBias:  []float64{0.0, 0.0, 0.0},
		FeatureCount: 3,
		InputDim:     2,
	})

	corpus := [][]float64{
		{1.0, 0.1}, // Moderate deception -> Feature 0 suppressed (1.0 - 2.0 < 0), Feature 1 fires!
		{1.2, 0.2}, // Moderate deception -> Feature 0 suppressed, Feature 1 fires!
		{1.5, 0.0}, // Moderate deception -> Feature 0 suppressed, Feature 1 fires!
		{3.0, 0.0}, // Strong deception -> both Feature 0 and Feature 1 fire
		{0.0, 2.0}, // Pure truth -> Feature 2 fires
	}
	concepts := map[string][]float64{
		"deception": {1.0, 0.0},
		"truth":     {0.0, 1.0},
	}

	report, err := AnalyzeFeatureAbsorption(saeDict, corpus, concepts)
	if err != nil {
		t.Fatalf("AnalyzeFeatureAbsorption failed: %v", err)
	}

	decMetric := report.ConceptMetrics["deception"]
	if decMetric.PrimaryFeatureIndex != 0 {
		t.Errorf("deception primary feature = %d, want 0", decMetric.PrimaryFeatureIndex)
	}
	// Feature 0 fired on only 1 of 4 deception-positive samples -> activation rate = 0.25
	// Feature 1 captured all 3 missed samples -> ChildCaptureRatio = 1.0, AbsorptionScore = 1.0 * 0.75 * 1.0 = 0.75
	if math.Abs(decMetric.EncoderActivationRate-0.25) > 1e-12 {
		t.Errorf("deception EncoderActivationRate = %f, want 0.25", decMetric.EncoderActivationRate)
	}
	if math.Abs(decMetric.AbsorptionScore-0.75) > 1e-12 || !decMetric.IsAbsorbed {
		t.Errorf("deception AbsorptionScore = %f (IsAbsorbed=%v), want 0.75 (true)",
			decMetric.AbsorptionScore, decMetric.IsAbsorbed)
	}
	if len(decMetric.AbsorbingChildren) != 1 || decMetric.AbsorbingChildren[0] != 1 {
		t.Errorf("expected absorbing child [1], got %v", decMetric.AbsorbingChildren)
	}

	truthMetric := report.ConceptMetrics["truth"]
	if truthMetric.IsAbsorbed || truthMetric.AbsorptionScore != 0.0 {
		t.Errorf("truth should not be absorbed: %+v", truthMetric)
	}
	// 1 of 2 concepts absorbed -> AbsorptionRate = 0.5
	if math.Abs(report.AbsorptionRate-0.5) > 1e-12 {
		t.Errorf("report.AbsorptionRate = %f, want 0.5", report.AbsorptionRate)
	}
}

func TestAccuracySAECrossArchCKAAndAbsorptionInvariants(t *testing.T) {
	// Create a Gemma Scope SAE and a permuted/scaled Llama-Scope SAE representing the same features
	gemma := &SAELoader{}
	_ = gemma.LoadFromMemory(&SAEWeights{
		Family:         SAEGemmaScope,
		EncoderWeights: [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		DecoderWeights: [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		EncoderBias:    []float64{0, 0, 0},
		DecoderBias:    []float64{0, 0, 0},
		FeatureCount:   3,
		InputDim:       3,
	})

	llama := &SAELoader{}
	_ = llama.LoadFromMemory(&SAEWeights{
		Family:         SAELlamaScope,
		EncoderWeights: [][]float64{{2, 0, 0}, {0, 2, 0}, {0, 0, 2}},
		DecoderWeights: [][]float64{{1, 0, 0}, {0, 1, 0}, {0, 0, 1}},
		EncoderBias:    []float64{0, 0, 0},
		DecoderBias:    []float64{0, 0, 0},
		FeatureCount:   3,
		InputDim:       3,
	})

	acts := [][]float64{
		{2.0, 0.1, 0.0},
		{0.0, 3.0, 0.2},
		{0.1, 0.0, 2.5},
		{1.5, 1.5, 0.0},
	}

	res, err := AlignSAEDictionaries(gemma, llama, acts, acts, 3)
	if err != nil {
		t.Fatalf("AlignSAEDictionaries failed: %v", err)
	}

	// Scale invariance of Linear CKA: CKA(F, 2F) == 1.0
	if math.Abs(res.LinearCKA-1.0) > 1e-9 {
		t.Errorf("LinearCKA = %f, want 1.0", res.LinearCKA)
	}
	if math.Abs(res.MeanAlignmentCosine-1.0) > 1e-9 {
		t.Errorf("MeanAlignmentCosine = %f, want 1.0", res.MeanAlignmentCosine)
	}
	for _, pair := range res.AlignedPairs {
		if pair.SourceFeature != pair.TargetFeature {
			t.Errorf("expected feature %d to align with %d, got %d", pair.SourceFeature, pair.SourceFeature, pair.TargetFeature)
		}
	}

	// Error paths for AnalyzeFeatureAbsorption and AlignSAEDictionaries
	if _, err := AnalyzeFeatureAbsorption(nil, acts, nil); err == nil {
		t.Error("expected error for nil dictionary")
	}
	if _, err := AnalyzeFeatureAbsorption(gemma, nil, nil); err == nil {
		t.Error("expected error for empty activations")
	}
	if _, err := AnalyzeFeatureAbsorption(gemma, [][]float64{{1}}, nil); err == nil {
		t.Error("expected error for wrong sample dimension")
	}
	if _, err := AnalyzeFeatureAbsorption(gemma, acts, map[string][]float64{"bad": {1}}); err == nil {
		t.Error("expected error for wrong concept dimension")
	}
	if _, err := AlignSAEDictionaries(nil, llama, acts, acts, 2); err == nil {
		t.Error("expected error for nil source dictionary")
	}
	if _, err := AlignSAEDictionaries(gemma, llama, nil, nil, 2); err == nil {
		t.Error("expected error for empty paired activations")
	}
	if _, err := AlignSAEDictionaries(gemma, llama, [][]float64{{1}}, acts[:1], 2); err == nil {
		t.Error("expected error for invalid source sample dim")
	}
	if _, err := AlignSAEDictionaries(gemma, llama, acts[:1], [][]float64{{1}}, 2); err == nil {
		t.Error("expected error for invalid target sample dim")
	}
}
