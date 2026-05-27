package policy

import (
	"fmt"
	"math"
	"strings"
	"testing"
)

// MockLLMClient simulates an LLM response based on string heuristics for testing.
type MockLLMClient struct{}

func (m *MockLLMClient) ClassifyIntent(prompt string) (IntentType, error) {
	lower := strings.ToLower(prompt)
	if strings.Contains(lower, "roleplay") || strings.Contains(lower, "pretend") {
		return IntentFictionalRoleplay, nil
	}
	if strings.Contains(lower, "story") || strings.Contains(lower, "creative") {
		return IntentCreativeWriting, nil
	}
	if strings.Contains(lower, "academic") || strings.Contains(lower, "research") {
		return IntentAcademicQuery, nil
	}
	return IntentGeneralConversation, nil
}

func TestSemanticGater_ClassifyPrompt(t *testing.T) {
	sg := NewSemanticGater(&MockLLMClient{})

	tests := []struct {
		prompt string
		want   IntentType
	}{
		{"Act as an undercover agent in a thriller drama. pretend to be a spy.", IntentFictionalRoleplay},
		{"Write a story about a dragon who lost its treasure.", IntentCreativeWriting},
		{"Can you review the academic research paper on LLM alignment?", IntentAcademicQuery},
		{"What is the capital city of France?", IntentGeneralConversation},
	}

	for _, tt := range tests {
		got := sg.ClassifyPrompt(tt.prompt)
		if got != tt.want {
			t.Errorf("ClassifyPrompt(%q) = %q, want %q", tt.prompt, got, tt.want)
		}
	}
}

func TestSemanticGater_ScaleMetric(t *testing.T) {
	sg := NewSemanticGater(&MockLLMClient{})

	// Creative writing has a weight of 0.1, so a score of 0.8 becomes 0.08
	creativePrompt := "Write a story about an honest thief."
	scaledCreative := sg.ScaleMetric(creativePrompt, 0.8)
	if math.Abs(scaledCreative-0.08) > 1e-9 {
		t.Errorf("creative scale: got %f, want 0.08", scaledCreative)
	}

	// General conversation has weight 1.0, so 0.8 remains 0.8
	generalPrompt := "What are the benefits of eating green apples?"
	scaledGeneral := sg.ScaleMetric(generalPrompt, 0.8)
	if math.Abs(scaledGeneral-0.8) > 1e-9 {
		t.Errorf("general scale: got %f, want 0.8", scaledGeneral)
	}
}

func TestSemanticGater_ShouldBypass(t *testing.T) {
	sg := NewSemanticGater(&MockLLMClient{})

	// Fictional roleplays should bypass interventions to preserve agent roleplay capabilities
	roleplayPrompt := "Pretend you are a secret agent who must hide their identity."
	if !sg.ShouldBypassIntervention(roleplayPrompt) {
		t.Error("expected roleplay prompt to bypass safety interventions")
	}

	// General conversations should NOT bypass safety interventions
	generalPrompt := "Tell me how to write a good budget statement."
	if sg.ShouldBypassIntervention(generalPrompt) {
		t.Error("expected general conversation to NOT bypass safety interventions")
	}
}

// Fuzz test to ensure the gater never panics regardless of input
func FuzzSemanticGater_ClassifyPrompt(f *testing.F) {
	sg := NewSemanticGater(&MockLLMClient{})

	// Add seed corpus
	f.Add("Pretend to be a knight")
	f.Add("Write a creative story")
	f.Add("Academic research on AI")
	f.Add("Hello world")

	f.Fuzz(func(t *testing.T, prompt string) {
		intent := sg.ClassifyPrompt(prompt)
		// Should always return one of the valid intents
		if intent != IntentFictionalRoleplay &&
			intent != IntentCreativeWriting &&
			intent != IntentAcademicQuery &&
			intent != IntentGeneralConversation {
			t.Errorf("unexpected intent returned: %q", intent)
		}
	})
}

type errorLLMClient struct{}

func (e *errorLLMClient) ClassifyIntent(prompt string) (IntentType, error) {
	return "", fmt.Errorf("simulated llm error")
}

func TestSemanticGater_SetIntentWeight_Boundaries(t *testing.T) {
	sg := NewSemanticGater(&MockLLMClient{})

	sg.SetIntentWeight(IntentGeneralConversation, -0.5)
	if w := sg.intentWeights[IntentGeneralConversation]; w != 0.0 {
		t.Errorf("expected weight bounded to 0.0, got %f", w)
	}

	sg.SetIntentWeight(IntentGeneralConversation, 1.5)
	if w := sg.intentWeights[IntentGeneralConversation]; w != 1.0 {
		t.Errorf("expected weight bounded to 1.0, got %f", w)
	}

	sg.SetIntentWeight(IntentGeneralConversation, 0.45)
	if w := sg.intentWeights[IntentGeneralConversation]; w != 0.45 {
		t.Errorf("expected weight set to 0.45, got %f", w)
	}
}

func TestSemanticGater_ClassifyPrompt_NilAndError(t *testing.T) {
	// Nil LLM client
	sgNil := NewSemanticGater(nil)
	if got := sgNil.ClassifyPrompt("any prompt"); got != IntentGeneralConversation {
		t.Errorf("expected GeneralConversation when LLM is nil, got %s", got)
	}

	// Error LLM client
	sgErr := NewSemanticGater(&errorLLMClient{})
	if got := sgErr.ClassifyPrompt("any prompt"); got != IntentGeneralConversation {
		t.Errorf("expected GeneralConversation on LLM error, got %s", got)
	}
}

func TestSemanticGater_ScaleMetric_Fallback(t *testing.T) {
	sg := NewSemanticGater(&MockLLMClient{})
	// Clear the intentWeights map so we trigger the fallback ok = false
	sg.intentWeights = make(map[IntentType]float64)

	scaled := sg.ScaleMetric("What is Paris?", 0.8)
	if math.Abs(scaled-0.8) > 1e-9 {
		t.Errorf("expected unscaled score (fallback weight 1.0), got %f", scaled)
	}
}
