package profile

import (
	"strings"
	"testing"
)

func TestDetectModelFamily(t *testing.T) {
	// Llama detection via ID
	llamaSpec := DetectModelFamily("meta-llama/Llama-3-8B-Instruct", nil, nil)
	if llamaSpec.Family != FamilyLlama {
		t.Errorf("expected Llama family, got %s", llamaSpec.Family)
	}
	if llamaSpec.HasAttentionBias {
		t.Errorf("Llama should not have attention bias")
	}

	// Qwen detection via config.json
	qwenConfig := []byte(`{"architectures": ["Qwen2ForCausalLM"], "num_hidden_layers": 28}`)
	qwenSpec := DetectModelFamily("custom-model", qwenConfig, nil)
	if qwenSpec.Family != FamilyQwen {
		t.Errorf("expected Qwen family, got %s", qwenSpec.Family)
	}
	if !qwenSpec.HasAttentionBias {
		t.Errorf("Qwen must have HasAttentionBias=true for QKV projection bias")
	}

	// Gemma detection via tensor names
	gemmaTensors := []string{
		"model.layers.0.pre_feedforward_layernorm.weight",
		"model.layers.0.post_feedforward_layernorm.weight",
	}
	gemmaSpec := DetectModelFamily("unknown-model", nil, gemmaTensors)
	if gemmaSpec.Family != FamilyGemma {
		t.Errorf("expected Gemma family, got %s", gemmaSpec.Family)
	}
	if !gemmaSpec.RMSNormWithOffset {
		t.Errorf("Gemma must have RMSNormWithOffset=true")
	}
	if !gemmaSpec.EmbeddingScaling {
		t.Errorf("Gemma must have EmbeddingScaling=true")
	}
}

func TestFormatPromptForFamily(t *testing.T) {
	prompt := "What is 2+2?"
	sys := "You are a helpful assistant."

	// Llama formatting
	llamaPrompt := FormatPromptForFamily(FamilyLlama, sys, prompt)
	if !strings.Contains(llamaPrompt, "<|begin_of_text|>") || !strings.Contains(llamaPrompt, "<|eot_id|>") {
		t.Errorf("Llama prompt missing special tokens: %s", llamaPrompt)
	}

	// Qwen formatting
	qwenPrompt := FormatPromptForFamily(FamilyQwen, sys, prompt)
	if !strings.Contains(qwenPrompt, "<|im_start|>system") || !strings.Contains(qwenPrompt, "<|im_end|>") {
		t.Errorf("Qwen prompt missing im_start/im_end: %s", qwenPrompt)
	}

	// Gemma formatting
	gemmaPrompt := FormatPromptForFamily(FamilyGemma, "", prompt)
	if !strings.Contains(gemmaPrompt, "<start_of_turn>user") || !strings.Contains(gemmaPrompt, "<end_of_turn>") {
		t.Errorf("Gemma prompt missing start_of_turn: %s", gemmaPrompt)
	}

	// Additional formatting branches (empty/non-empty system prompt across all families)
	_ = FormatPromptForFamily(FamilyLlama, "", prompt)
	_ = FormatPromptForFamily(FamilyQwen, "", prompt)
	_ = FormatPromptForFamily(FamilyGemma, sys, prompt)
	_ = FormatPromptForFamily(FamilyMistral, "", prompt)
	_ = FormatPromptForFamily(FamilyMistral, sys, prompt)
	_ = FormatPromptForFamily(FamilyUnknown, "", prompt)
	_ = FormatPromptForFamily(FamilyUnknown, sys, prompt)
}

func TestDetectModelFamily_AllBranches(t *testing.T) {
	// Config architectures: Llama, Gemma, Mistral
	if s := DetectModelFamily("x", []byte(`{"architectures":["LlamaForCausalLM"],"num_hidden_layers":40}`), nil); s.Family != FamilyLlama {
		t.Errorf("expected Llama, got %s", s.Family)
	}
	if s := DetectModelFamily("x", []byte(`{"architectures":["Gemma2ForCausalLM"],"num_hidden_layers":26}`), nil); s.Family != FamilyGemma {
		t.Errorf("expected Gemma, got %s", s.Family)
	}
	if s := DetectModelFamily("x", []byte(`{"architectures":["MistralForCausalLM"],"num_hidden_layers":48}`), nil); s.Family != FamilyMistral {
		t.Errorf("expected Mistral, got %s", s.Family)
	}

	// Config model_type: llama, qwen, gemma, mistral
	if s := DetectModelFamily("x", []byte(`{"model_type":"llama","num_hidden_layers":16}`), nil); s.Family != FamilyLlama {
		t.Errorf("expected Llama from model_type, got %s", s.Family)
	}
	if s := DetectModelFamily("x", []byte(`{"model_type":"qwen2","num_hidden_layers":32}`), nil); s.Family != FamilyQwen {
		t.Errorf("expected Qwen from model_type, got %s", s.Family)
	}
	if s := DetectModelFamily("x", []byte(`{"model_type":"gemma","num_hidden_layers":18}`), nil); s.Family != FamilyGemma {
		t.Errorf("expected Gemma from model_type, got %s", s.Family)
	}
	if s := DetectModelFamily("x", []byte(`{"model_type":"mixtral","num_hidden_layers":32}`), nil); s.Family != FamilyMistral {
		t.Errorf("expected Mistral from model_type, got %s", s.Family)
	}

	// ModelID substrings: qwen, gemma, mistral
	if s := DetectModelFamily("Qwen/Qwen2.5-7B", nil, nil); s.Family != FamilyQwen {
		t.Errorf("expected Qwen from modelID, got %s", s.Family)
	}
	if s := DetectModelFamily("google/gemma-2-9b", nil, nil); s.Family != FamilyGemma {
		t.Errorf("expected Gemma from modelID, got %s", s.Family)
	}
	if s := DetectModelFamily("mistralai/Mistral-7B", nil, nil); s.Family != FamilyMistral {
		t.Errorf("expected Mistral from modelID, got %s", s.Family)
	}

	// Tensor names: Qwen bias, Unknown fallback
	if s := DetectModelFamily("custom", nil, []string{"model.layers.0.self_attn.q_proj.bias"}); s.Family != FamilyQwen {
		t.Errorf("expected Qwen from tensor bias, got %s", s.Family)
	}
	if s := DetectModelFamily("completely-unknown", []byte(`{"model_type":"gpt2","num_hidden_layers":12}`), []string{"wte.weight"}); s.Family != FamilyUnknown {
		t.Errorf("expected Unknown family, got %s", s.Family)
	}

	// GetArchitectureSpec with 0 layers and scaleLayerRange edge cases
	_ = GetArchitectureSpec(FamilyUnknown, 0)
	_ = scaleLayerRange(10, -0.5, 1.5)
	_ = scaleLayerRange(10, 0.8, 0.2)
}
