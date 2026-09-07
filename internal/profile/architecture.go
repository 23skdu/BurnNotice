package profile

import (
	"encoding/json"
	"fmt"
	"strings"
)

// ModelFamily categorizes the foundational transformer architecture.
type ModelFamily string

const (
	FamilyLlama   ModelFamily = "llama"
	FamilyQwen    ModelFamily = "qwen"
	FamilyGemma   ModelFamily = "gemma"
	FamilyMistral ModelFamily = "mistral"
	FamilyUnknown ModelFamily = "unknown"
)

// ArchitectureSpec defines family-specific mechanics, normalization, layer layouts, and steering behaviors.
type ArchitectureSpec struct {
	Family                 ModelFamily       `json:"family"`
	DisplayName            string            `json:"display_name"`
	HasAttentionBias       bool              `json:"has_attention_bias"`        // true for Qwen (QKV projection bias)
	RMSNormWithOffset      bool              `json:"rms_norm_with_offset"`     // true for Gemma (1.0 + weight)
	EmbeddingScaling       bool              `json:"embedding_scaling"`        // true for Gemma (scaled by sqrt(d_model))
	PrePostFeedforwardNorm bool              `json:"pre_post_ff_norm"`         // true for Gemma 2 dual layernorm
	SpecialTokens          map[string]string `json:"special_tokens"`
	DefaultCriticalLayers  []int             `json:"default_critical_layers"`  // Layers with highest causal steering efficacy
	CommitmentWindowTokens int               `json:"commitment_window_tokens"` // Prefill commitment window size
	LayerPrefix            string            `json:"layer_prefix"`
}

// HuggingFaceConfig represents the minimum required fields from config.json.
type HuggingFaceConfig struct {
	Architectures []string `json:"architectures"`
	ModelType     string   `json:"model_type"`
	NumLayers     int      `json:"num_hidden_layers"`
	HiddenSize    int      `json:"hidden_size"`
}

// GetArchitectureSpec returns the predefined specification for a known model family.
func GetArchitectureSpec(family ModelFamily, numLayers int) ArchitectureSpec {
	switch family {
	case FamilyLlama:
		crit := []int{12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23}
		if numLayers > 0 && numLayers != 32 {
			crit = scaleLayerRange(numLayers, 0.40, 0.75)
		}
		return ArchitectureSpec{
			Family:                 FamilyLlama,
			DisplayName:            "Meta Llama (Llama 2 / 3 / 3.1 / 3.2 / 3.3)",
			HasAttentionBias:       false,
			RMSNormWithOffset:      false,
			EmbeddingScaling:       false,
			PrePostFeedforwardNorm: false,
			SpecialTokens: map[string]string{
				"bos":            "<|begin_of_text|>",
				"eos":            "<|end_of_text|>",
				"eot":            "<|eot_id|>",
				"header_start":   "<|start_header_id|>",
				"header_end":     "<|end_header_id|>",
			},
			DefaultCriticalLayers:  crit,
			CommitmentWindowTokens: 16,
			LayerPrefix:            "model.layers",
		}

	case FamilyQwen:
		crit := []int{14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25}
		if numLayers > 0 && numLayers != 32 {
			crit = scaleLayerRange(numLayers, 0.35, 0.70)
		}
		return ArchitectureSpec{
			Family:                 FamilyQwen,
			DisplayName:            "Alibaba Qwen (Qwen 2 / 2.5 / Coder / QwQ)",
			HasAttentionBias:       true, // Qwen self-attention has QKV bias terms
			RMSNormWithOffset:      false,
			EmbeddingScaling:       false,
			PrePostFeedforwardNorm: false,
			SpecialTokens: map[string]string{
				"im_start":     "<|im_start|>",
				"im_end":       "<|im_end|>",
				"think_start":  "<think>",
				"think_end":    "</think>",
			},
			DefaultCriticalLayers:  crit,
			CommitmentWindowTokens: 20,
			LayerPrefix:            "model.layers",
		}

	case FamilyGemma:
		crit := []int{10, 11, 12, 13, 14, 15, 16, 17, 18}
		if numLayers > 0 && numLayers != 26 {
			crit = scaleLayerRange(numLayers, 0.35, 0.65)
		}
		return ArchitectureSpec{
			Family:                 FamilyGemma,
			DisplayName:            "Google Gemma (Gemma 1 / 2 / Scope)",
			HasAttentionBias:       false,
			RMSNormWithOffset:      true, // Gemma applies (1.0 + weight) to RMSNorm
			EmbeddingScaling:       true, // Gemma scales input embeddings by sqrt(hidden_size)
			PrePostFeedforwardNorm: true, // Gemma 2 has pre- and post-feedforward layernorms
			SpecialTokens: map[string]string{
				"bos":          "<bos>",
				"eos":          "<eos>",
				"turn_start":   "<start_of_turn>",
				"turn_end":     "<end_of_turn>",
			},
			DefaultCriticalLayers:  crit,
			CommitmentWindowTokens: 12,
			LayerPrefix:            "model.layers",
		}

	case FamilyMistral:
		crit := []int{12, 13, 14, 15, 16, 17, 18, 19, 20}
		if numLayers > 0 && numLayers != 32 {
			crit = scaleLayerRange(numLayers, 0.35, 0.70)
		}
		return ArchitectureSpec{
			Family:                 FamilyMistral,
			DisplayName:            "Mistral AI (Mistral / Mixtral)",
			HasAttentionBias:       false,
			RMSNormWithOffset:      false,
			EmbeddingScaling:       false,
			PrePostFeedforwardNorm: false,
			SpecialTokens: map[string]string{
				"bos": "<s>",
				"eos": "</s>",
			},
			DefaultCriticalLayers:  crit,
			CommitmentWindowTokens: 16,
			LayerPrefix:            "model.layers",
		}

	default:
		crit := []int{8, 9, 10, 11, 12, 13, 14, 15}
		if numLayers > 0 {
			crit = scaleLayerRange(numLayers, 0.40, 0.70)
		}
		return ArchitectureSpec{
			Family:                 FamilyUnknown,
			DisplayName:            "Generic Transformer",
			HasAttentionBias:       false,
			RMSNormWithOffset:      false,
			EmbeddingScaling:       false,
			PrePostFeedforwardNorm: false,
			SpecialTokens:          make(map[string]string),
			DefaultCriticalLayers:  crit,
			CommitmentWindowTokens: 16,
			LayerPrefix:            "model.layers",
		}
	}
}

// DetectModelFamily detects the model family from model ID, config.json, or tensor names.
func DetectModelFamily(modelID string, configJSON []byte, tensorNames []string) ArchitectureSpec {
	numLayers := 32

	// 1. Try parsing config.json if provided
	if len(configJSON) > 0 {
		var cfg HuggingFaceConfig
		if err := json.Unmarshal(configJSON, &cfg); err == nil {
			if cfg.NumLayers > 0 {
				numLayers = cfg.NumLayers
			}
			for _, arch := range cfg.Architectures {
				archLower := strings.ToLower(arch)
				if strings.Contains(archLower, "llama") {
					return GetArchitectureSpec(FamilyLlama, numLayers)
				}
				if strings.Contains(archLower, "qwen") {
					return GetArchitectureSpec(FamilyQwen, numLayers)
				}
				if strings.Contains(archLower, "gemma") {
					return GetArchitectureSpec(FamilyGemma, numLayers)
				}
				if strings.Contains(archLower, "mistral") || strings.Contains(archLower, "mixtral") {
					return GetArchitectureSpec(FamilyMistral, numLayers)
				}
			}
			typeLower := strings.ToLower(cfg.ModelType)
			if strings.Contains(typeLower, "llama") {
				return GetArchitectureSpec(FamilyLlama, numLayers)
			}
			if strings.Contains(typeLower, "qwen") {
				return GetArchitectureSpec(FamilyQwen, numLayers)
			}
			if strings.Contains(typeLower, "gemma") {
				return GetArchitectureSpec(FamilyGemma, numLayers)
			}
			if strings.Contains(typeLower, "mistral") || strings.Contains(typeLower, "mixtral") {
				return GetArchitectureSpec(FamilyMistral, numLayers)
			}
		}
	}

	// 2. Check modelID substring
	idLower := strings.ToLower(modelID)
	if strings.Contains(idLower, "llama") {
		return GetArchitectureSpec(FamilyLlama, numLayers)
	}
	if strings.Contains(idLower, "qwen") || strings.Contains(idLower, "qwq") {
		return GetArchitectureSpec(FamilyQwen, numLayers)
	}
	if strings.Contains(idLower, "gemma") {
		return GetArchitectureSpec(FamilyGemma, numLayers)
	}
	if strings.Contains(idLower, "mistral") || strings.Contains(idLower, "mixtral") {
		return GetArchitectureSpec(FamilyMistral, numLayers)
	}

	// 3. Inspect tensor naming signatures
	for _, name := range tensorNames {
		nameLower := strings.ToLower(name)
		if strings.Contains(nameLower, "pre_feedforward_layernorm") || strings.Contains(nameLower, "post_feedforward_layernorm") {
			return GetArchitectureSpec(FamilyGemma, numLayers)
		}
		if strings.Contains(nameLower, "self_attn.q_proj.bias") || strings.Contains(nameLower, "self_attn.k_proj.bias") {
			return GetArchitectureSpec(FamilyQwen, numLayers)
		}
	}

	return GetArchitectureSpec(FamilyUnknown, numLayers)
}

// FormatPromptForFamily formats system and user prompts into the architecture's native chat template.
func FormatPromptForFamily(family ModelFamily, system, user string) string {
	switch family {
	case FamilyLlama:
		if system == "" {
			return fmt.Sprintf("<|begin_of_text|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", user)
		}
		return fmt.Sprintf("<|begin_of_text|><|start_header_id|>system<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>user<|end_header_id|>\n\n%s<|eot_id|><|start_header_id|>assistant<|end_header_id|>\n\n", system, user)

	case FamilyQwen:
		if system == "" {
			return fmt.Sprintf("<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", user)
		}
		return fmt.Sprintf("<|im_start|>system\n%s<|im_end|>\n<|im_start|>user\n%s<|im_end|>\n<|im_start|>assistant\n", system, user)

	case FamilyGemma:
		if system == "" {
			return fmt.Sprintf("<start_of_turn>user\n%s<end_of_turn>\n<start_of_turn>model\n", user)
		}
		return fmt.Sprintf("<start_of_turn>user\n%s\n\n%s<end_of_turn>\n<start_of_turn>model\n", system, user)

	default:
		if system == "" {
			return fmt.Sprintf("User: %s\nAssistant: ", user)
		}
		return fmt.Sprintf("System: %s\nUser: %s\nAssistant: ", system, user)
	}
}

func scaleLayerRange(totalLayers int, startPct, endPct float64) []int {
	start := int(float64(totalLayers) * startPct)
	end := int(float64(totalLayers) * endPct)
	if start < 0 {
		start = 0
	}
	if end >= totalLayers {
		end = totalLayers - 1
	}
	if start > end {
		start = end
	}

	result := make([]int, 0, end-start+1)
	for i := start; i <= end; i++ {
		result = append(result, i)
	}
	return result
}
