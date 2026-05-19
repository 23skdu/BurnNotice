package policy

import (
	"math"
	"testing"
)

func TestSemanticGater_ClassifyPrompt(t *testing.T) {
	sg := NewSemanticGater()

	tests := []struct {
		prompt string
		want   IntentType
	}{
		{"Act as an undercover agent in a thriller drama.", IntentFictionalRoleplay},
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
	sg := NewSemanticGater()

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
	sg := NewSemanticGater()

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
