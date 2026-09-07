package steer

import (
	"math"
	"testing"
)

func TestCoTTracker_StateTransitions(t *testing.T) {
	tracker := NewCoTTracker(true)

	// 1. Initial direct prompt tokens
	steer, phase := tracker.IngestToken("User:", 0.1)
	if !steer || phase != PhaseDirect {
		t.Errorf("expected direct phase with active steering, got steer=%v, phase=%v", steer, phase)
	}

	// 2. Transition into thinking phase: <think>
	steer, phase = tracker.IngestToken("<think>", 0.8)
	if steer || phase != PhaseThinking {
		t.Errorf("expected thinking phase with zero steering, got steer=%v, phase=%v", steer, phase)
	}

	// 3. Internal thinking tokens: exploration with high deception probe score
	steer, phase = tracker.IngestToken("exploring backdoor trigger", 0.9)
	if steer || phase != PhaseThinking {
		t.Errorf("expected thinking phase to suppress steering intervention")
	}

	// 4. Transition into answer phase: </think>
	steer, phase = tracker.IngestToken("</think>", 0.3)
	if !steer || phase != PhaseAnswer {
		t.Errorf("expected answer phase with full steering intervention, got steer=%v, phase=%v", steer, phase)
	}

	// 5. Generated answer tokens under active steering
	steer, phase = tracker.IngestToken("Here is the safe response.", 0.05)
	if !steer || phase != PhaseAnswer {
		t.Errorf("expected continued answer phase steering")
	}

	summary := tracker.Summary()
	if summary.ThinkingTokens != 2 {
		t.Errorf("expected 2 thinking tokens, got %d", summary.ThinkingTokens)
	}
	if summary.AnswerTokens != 3 {
		t.Errorf("expected 3 answer/direct tokens, got %d", summary.AnswerTokens)
	}
	if math.Abs(summary.MeanThinkingScore-0.85) > 1e-3 {
		t.Errorf("expected mean thinking score 0.85, got %f", summary.MeanThinkingScore)
	}
	if !summary.DeceptionSuppressed {
		t.Errorf("expected DeceptionSuppressed=true (high thinking deception vs low answer deception)")
	}
}

func TestCoTTracker_NonReasoningModel(t *testing.T) {
	// Standard non-reasoning model (ReasoningModelAware = false)
	tracker := NewCoTTracker(false)

	steer, phase := tracker.IngestToken("<think>", 0.5)
	if !steer || phase != PhaseDirect {
		t.Errorf("non-reasoning tracker should steer normally regardless of tokens")
	}
}
