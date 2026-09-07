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
}
