package policy

import (
	"strings"
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

// SemanticGater classifies conversational prompts to adjust safety steering policies dynamically.
type SemanticGater struct {
	mu           sync.RWMutex
	bypassIntents map[IntentType]bool
	intentWeights map[IntentType]float64
}

// NewSemanticGater creates a gater with default configurations.
func NewSemanticGater() *SemanticGater {
	return &SemanticGater{
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

// ClassifyPrompt evaluates semantic markers inside the text to determine the intent class.
func (sg *SemanticGater) ClassifyPrompt(prompt string) IntentType {
	lower := strings.ToLower(prompt)

	// Fictional roleplay heuristic markers
	if strings.Contains(lower, "roleplay") || strings.Contains(lower, "pretend you are") || strings.Contains(lower, "act as") || strings.Contains(lower, "[role]") {
		return IntentFictionalRoleplay
	}

	// Creative writing heuristic markers
	if strings.Contains(lower, "write a story") || strings.Contains(lower, "creative writing") || strings.Contains(lower, "novel") || strings.Contains(lower, "fiction") {
		return IntentCreativeWriting
	}

	// Academic discussion of deception/safety markers
	if strings.Contains(lower, "academic") || strings.Contains(lower, "research paper") || strings.Contains(lower, "study of deception") || strings.Contains(lower, "cognitive science") {
		return IntentAcademicQuery
	}

	return IntentGeneralConversation
}

// ScaleMetric returns a scaling multiplier for a safety metric score based on the prompt context.
func (sg *SemanticGater) ScaleMetric(prompt string, score float64) float64 {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	intent := sg.ClassifyPrompt(prompt)
	weight, ok := sg.intentWeights[intent]
	if !ok {
		weight = 1.0
	}

	return score * weight
}

// ShouldBypassIntervention returns true if the context allows bypassing full safety steering.
func (sg *SemanticGater) ShouldBypassIntervention(prompt string) bool {
	sg.mu.RLock()
	defer sg.mu.RUnlock()

	intent := sg.ClassifyPrompt(prompt)
	return sg.bypassIntents[intent]
}
