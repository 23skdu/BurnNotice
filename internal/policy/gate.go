package policy

import (
	"sync"
)

// IntentType represents the classified semantic context of a prompt.
type IntentType string

const (
	IntentCreativeWriting IntentType = "creative_writing"
	IntentAcademicQuery   IntentType = "academic_query"
	IntentFictionalRoleplay IntentType = "fictional_roleplay"
	IntentGeneralConversation IntentType = "general_conversation"
)

// LLMClient defines the interface for an async LLM pre-classifier backend.
type LLMClient interface {
	ClassifyIntent(prompt string) (IntentType, error)
}

// SemanticGater classifies conversational prompts to adjust safety steering policies dynamically.
type SemanticGater struct {
	mu           sync.RWMutex
	llm          LLMClient
	bypassIntents map[IntentType]bool
	intentWeights map[IntentType]float64
}

// NewSemanticGater creates a gater with default configurations using the provided LLMClient.
func NewSemanticGater(llm LLMClient) *SemanticGater {
	return &SemanticGater{
		llm: llm,
		bypassIntents: map[IntentType]bool{
			IntentCreativeWriting: true,
			IntentFictionalRoleplay: true,
		},
		intentWeights: map[IntentType]float64{
			IntentCreativeWriting: 0.1,
			IntentFictionalRoleplay: 0.0,
			IntentAcademicQuery:   0.3,
			IntentGeneralConversation: 1.0,
		},
	}
}

// SetIntentWeight overrides the safety scaling weight of a specific intent context.
func (sg *SemanticGater) SetIntentWeight(intent IntentType, weight float64) {
	sg.mu.Lock()
	defer sg.mu.Unlock()
	if weight < 0 {
		weight = 0
	}
	if weight > 1 {
		weight = 1
	}
	sg.intentWeights[intent] = weight
}

// ClassifyPrompt evaluates semantic markers inside the text via an LLM to determine the intent class.
func (sg *SemanticGater) ClassifyPrompt(prompt string) IntentType {
	if sg.llm == nil {
		return IntentGeneralConversation
	}
	intent, err := sg.llm.ClassifyIntent(prompt)
	if err != nil {
		// Fallback to general conversation on error to maintain safety
		return IntentGeneralConversation
	}
	return intent
}

// ScaleMetric returns a scaling multiplier for a safety metric score based on the prompt context.
func (sg *SemanticGater) ScaleMetric(prompt string, score float64) float64 {
	intent := sg.ClassifyPrompt(prompt)

	sg.mu.RLock()
	defer sg.mu.RUnlock()

	weight, ok := sg.intentWeights[intent]
	if !ok {
		weight = 1.0
	}

	return score * weight
}

// ShouldBypassIntervention returns true if the context allows bypassing full safety steering.
func (sg *SemanticGater) ShouldBypassIntervention(prompt string) bool {
	intent := sg.ClassifyPrompt(prompt)

	sg.mu.RLock()
	defer sg.mu.RUnlock()

	return sg.bypassIntents[intent]
}
