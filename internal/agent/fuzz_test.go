package agent

import (
	"context"
	"math"
	"testing"
)

func FuzzMiddlewareInterceptAndBlock(f *testing.F) {
	f.Add("model-1", "hello world", "response text", 0.7, 0.85)
	f.Add("qwen-0.5b", "", "ok", 0.5, 0.2)
	f.Add("llama-3", "prompt", "deceptive output", 0.0, 0.9)

	f.Fuzz(func(t *testing.T, modelID, prompt, respText string, threshold, score float64) {
		if math.IsNaN(threshold) || math.IsInf(threshold, 0) || math.IsNaN(score) || math.IsInf(score, 0) {
			return
		}
		mw, err := NewMiddleware(Config{
			ModelID:          modelID,
			ScoringThreshold: threshold,
		})
		if modelID == "" {
			if err == nil {
				t.Error("expected error for empty ModelID")
			}
			return
		}
		if err != nil {
			t.Fatalf("unexpected NewMiddleware error: %v", err)
		}

		ctx := context.Background()
		scores, err := mw.InterceptRequest(ctx, Request{Prompt: prompt, SessionID: "fuzz-sess"})
		if err != nil || scores == nil {
			t.Fatalf("InterceptRequest failed: %v", err)
		}

		inScores := map[string]float64{"deception": score}
		outResp, err := mw.InterceptResponse(ctx, Response{
			Text:      respText,
			SessionID: "fuzz-sess",
			Scores:    inScores,
		})
		if err != nil {
			t.Fatalf("InterceptResponse failed: %v", err)
		}

		effectiveThresh := threshold
		if effectiveThresh <= 0 {
			effectiveThresh = 0.7
		}
		blocked := mw.ShouldBlock(inScores)
		if (score > effectiveThresh) != blocked {
			t.Errorf("ShouldBlock(%v) = %v, want %v (thresh=%v)", score, blocked, score > effectiveThresh, effectiveThresh)
		}
		if blocked && outResp.Metadata["steered"] != "true" {
			t.Error("expected steered metadata when score exceeds threshold")
		}
	})
}

func FuzzMultiAgentGovernor(f *testing.F) {
	f.Add("agent-a", "agent-b", "msg content", 0.6, 0.8)
	f.Add("a1", "a2", "", 0.5, 0.2)

	f.Fuzz(func(t *testing.T, from, to, content string, driftThresh, score float64) {
		if math.IsNaN(driftThresh) || math.IsInf(driftThresh, 0) || math.IsNaN(score) || math.IsInf(score, 0) {
			return
		}
		if from == "" || to == "" {
			return
		}
		gov := NewMultiAgentGovernor(driftThresh)
		gov.RegisterAgent(from, "From")
		gov.RegisterAgent(to, "To")

		msg, allowed := gov.MonitorMessage(from, to, content, score)
		if msg == nil {
			t.Fatal("MonitorMessage returned nil message")
		}
		if (score <= 0.7) != allowed {
			t.Errorf("allowed = %v for score %v, want %v", allowed, score, score <= 0.7)
		}
		_ = gov.DetectDriftPropagation()
	})
}
