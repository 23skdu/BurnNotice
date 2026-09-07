package steer

import (
	"strings"
	"sync"
)

// CoTPhase denotes the generation stage for reasoning models.
type CoTPhase string

const (
	PhaseDirect   CoTPhase = "DIRECT"
	PhaseThinking CoTPhase = "THINKING"
	PhaseAnswer   CoTPhase = "ANSWER"
)

// CoTTracker monitors token streams to separate internal Chain-of-Thought deliberation
// from final answer emission, preventing premature steering from corrupting exploration.
type CoTTracker struct {
	mu                  sync.RWMutex
	ReasoningModelAware bool
	currentPhase        CoTPhase
	thinkingTokens      int
	answerTokens        int
	thinkingScores      []float64
	answerScores        []float64
}

// NewCoTTracker creates a tracker with reasoning-model awareness.
func NewCoTTracker(reasoningModelAware bool) *CoTTracker {
	initialPhase := PhaseDirect
	if reasoningModelAware {
		initialPhase = PhaseDirect // Will transition to THINKING upon <think> tag
	}
	return &CoTTracker{
		ReasoningModelAware: reasoningModelAware,
		currentPhase:        initialPhase,
		thinkingScores:      make([]float64, 0),
		answerScores:        make([]float64, 0),
	}
}

// IngestToken inspects the generated token, updates the thinking state machine,
// records stage-specific deception scores, and returns whether active steering intervention should be applied.
func (ct *CoTTracker) IngestToken(token string, deceptionScore float64) (shouldSteer bool, phase CoTPhase) {
	ct.mu.Lock()
	defer ct.mu.Unlock()

	cleaned := strings.TrimSpace(strings.ToLower(token))

	// Detect boundary tags
	if ct.ReasoningModelAware {
		if strings.Contains(cleaned, "<think>") {
			ct.currentPhase = PhaseThinking
		} else if strings.Contains(cleaned, "</think>") {
			ct.currentPhase = PhaseAnswer
		}
	}

	phase = ct.currentPhase

	if !ct.ReasoningModelAware {
		ct.answerTokens++
		ct.answerScores = append(ct.answerScores, deceptionScore)
		return true, PhaseDirect
	}

	switch ct.currentPhase {
	case PhaseThinking:
		ct.thinkingTokens++
		ct.thinkingScores = append(ct.thinkingScores, deceptionScore)
		// Step 23.2: During thinking phase: monitoring only, zero steering intervention
		return false, PhaseThinking

	case PhaseAnswer:
		ct.answerTokens++
		ct.answerScores = append(ct.answerScores, deceptionScore)
		// Step 23.3: During answer phase: full steering intervention
		return true, PhaseAnswer

	default:
		// If model hasn't entered <think> yet, monitor or steer directly
		ct.answerTokens++
		ct.answerScores = append(ct.answerScores, deceptionScore)
		return true, PhaseDirect
	}
}

// CurrentPhase returns the active reasoning phase.
func (ct *CoTTracker) CurrentPhase() CoTPhase {
	ct.mu.RLock()
	defer ct.mu.RUnlock()
	return ct.currentPhase
}

// CoTMetrics provides token-separated summary statistics.
type CoTMetrics struct {
	ThinkingTokens      int     `json:"thinking_tokens"`
	AnswerTokens        int     `json:"answer_tokens"`
	MeanThinkingScore   float64 `json:"mean_thinking_score"`
	MeanAnswerScore     float64 `json:"mean_answer_score"`
	DeceptionSuppressed bool    `json:"deception_suppressed"`
}

// Summary returns stage-specific metrics for reasoning models.
func (ct *CoTTracker) Summary() CoTMetrics {
	ct.mu.RLock()
	defer ct.mu.RUnlock()

	meanThink := 0.0
	if len(ct.thinkingScores) > 0 {
		var sum float64
		for _, s := range ct.thinkingScores {
			sum += s
		}
		meanThink = sum / float64(len(ct.thinkingScores))
	}

	meanAns := 0.0
	if len(ct.answerScores) > 0 {
		var sum float64
		for _, s := range ct.answerScores {
			sum += s
		}
		meanAns = sum / float64(len(ct.answerScores))
	}

	return CoTMetrics{
		ThinkingTokens:    ct.thinkingTokens,
		AnswerTokens:      ct.answerTokens,
		MeanThinkingScore: meanThink,
		MeanAnswerScore:   meanAns,
		// If answer deception is significantly below thinking deception, steering was effective
		DeceptionSuppressed: meanThink > 0.5 && meanAns < 0.3,
	}
}

// Reset clears the stored token sequence history and phase.
func (ct *CoTTracker) Reset() {
	ct.mu.Lock()
	defer ct.mu.Unlock()
	ct.currentPhase = PhaseDirect
	ct.thinkingTokens = 0
	ct.answerTokens = 0
	ct.thinkingScores = ct.thinkingScores[:0]
	ct.answerScores = ct.answerScores[:0]
}
