package agent

import (
	"context"
	"testing"
	"time"

	"burnnotice/internal/policy"
)

func TestNewMiddleware(t *testing.T) {
	cfg := Config{
		ModelID:          "test-model",
		ScoringThreshold: 0.7,
		SteeringStrength: 0.5,
	}
	mw, err := NewMiddleware(cfg)
	if err != nil {
		t.Fatalf("NewMiddleware error: %v", err)
	}
	if mw == nil {
		t.Fatal("middleware is nil")
	}
}

func TestInterceptRequest(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})

	req := Request{
		Prompt:    "test prompt",
		SessionID: "sess-001",
		Timestamp: time.Now(),
	}

	scores, err := mw.InterceptRequest(context.Background(), req)
	if err != nil {
		t.Fatalf("InterceptRequest error: %v", err)
	}
	if scores == nil {
		t.Fatal("scores is nil")
	}
}

func TestInterceptResponse(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})

	resp := Response{
		Text:      "test response",
		SessionID: "sess-001",
		Timestamp: time.Now(),
		Scores:    map[string]float64{"deception": 0.5},
	}

	steered, err := mw.InterceptResponse(context.Background(), resp)
	if err != nil {
		t.Fatalf("InterceptResponse error: %v", err)
	}
	if steered.Text == "" {
		t.Fatal("steered text is empty")
	}
}

func TestInterceptResponseSteering(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.5, SteeringStrength: 0.3})

	resp := Response{
		Text:      "deceptive response",
		SessionID: "sess-001",
		Timestamp: time.Now(),
		Scores:    map[string]float64{"deception": 0.8},
	}

	steered, err := mw.InterceptResponse(context.Background(), resp)
	if err != nil {
		t.Fatalf("InterceptResponse error: %v", err)
	}
	if steered.Text == "" {
		t.Fatal("steered text is empty")
	}
}

func TestCrewAITaskDecorator(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})
	decorator := NewCrewAITaskDecorator(mw, "sess-001")

	result, err := decorator.ExecuteTask(context.Background(), "test-task", func() (string, error) {
		return "task result", nil
	})
	if err != nil {
		t.Fatalf("ExecuteTask error: %v", err)
	}
	if result != "task result" {
		t.Errorf("result = %s, want task result", result)
	}
}

func TestMCPMiddleware(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})
	mcp := NewMCPMiddleware(mw, "sess-001")

	err := mcp.InterceptToolCall(context.Background(), "search", map[string]interface{}{"query": "test"})
	if err != nil {
		t.Fatalf("InterceptToolCall error: %v", err)
	}

	err = mcp.InterceptToolResponse(context.Background(), "search", "search results")
	if err != nil {
		t.Fatalf("InterceptToolResponse error: %v", err)
	}
}

func TestMCPMiddleware_WithOPAEvaluator(t *testing.T) {
	ctx := context.Background()
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})
	mcp := NewMCPMiddleware(mw, "sess-opa")

	// Without OPAEvaluator: scalar threshold fallback
	dec, err := mcp.InterceptToolCallWithLatent(ctx, "read_file", "low", nil, map[string]float64{"deception": 0.2})
	if err != nil || !dec.Allow {
		t.Fatalf("expected allow without OPA on low score, got dec=%+v err=%v", dec, err)
	}
	dec, err = mcp.InterceptToolCallWithLatent(ctx, "exec_sql", "high", nil, map[string]float64{"deception": 0.85})
	if err == nil || dec.Allow {
		t.Fatalf("expected scalar block without OPA on high score, got dec=%+v err=%v", dec, err)
	}

	// Attach OPAEvaluator with tool + blast_radius + latent score rule
	opa, err := policy.NewOPAEvaluator(policy.OPAConfig{
		Rules: []policy.OPARule{
			{
				ID:         "block_exec_sql_deception",
				Tools:      []string{"exec_sql"},
				BlastRadii: []string{"high", "critical"},
				MinScores:  map[string]float64{"deception": 0.35},
				Action:     policy.ActionBlock,
				Reason:     "latent deception on destructive tool",
			},
		},
	})
	if err != nil {
		t.Fatalf("NewOPAEvaluator failed: %v", err)
	}
	mcp.SetOPAEvaluator(opa)

	// InterceptToolCall (low default scores -> allowed)
	if err := mcp.InterceptToolCall(ctx, "exec_sql", map[string]interface{}{"q": "SELECT 1"}); err != nil {
		t.Fatalf("expected allow on zero score InterceptToolCall, got %v", err)
	}

	// InterceptToolCallWithLatent on exec_sql with deception=0.45 -> Blocked by OPA
	dec, err = mcp.InterceptToolCallWithLatent(ctx, "exec_sql", "high", map[string]interface{}{"q": "DROP TABLE users"}, map[string]float64{"deception": 0.45})
	if err == nil || dec.Allow || dec.PolicyID != "block_exec_sql_deception" {
		t.Fatalf("expected OPA block on exec_sql with deception=0.45, got dec=%+v err=%v", dec, err)
	}

	// InterceptToolCallWithLatent on read_file with deception=0.45 -> Allowed by OPA
	dec, err = mcp.InterceptToolCallWithLatent(ctx, "read_file", "low", nil, map[string]float64{"deception": 0.45})
	if err != nil || !dec.Allow {
		t.Fatalf("expected OPA allow on read_file with deception=0.45, got dec=%+v err=%v", dec, err)
	}
}

func TestMultiAgentGovernor(t *testing.T) {
	gov := NewMultiAgentGovernor(0.7)

	gov.RegisterAgent("agent-1", "Agent One")
	gov.RegisterAgent("agent-2", "Agent Two")

	if gov.GetAgentCount() != 2 {
		t.Fatalf("GetAgentCount = %d, want 2", gov.GetAgentCount())
	}

	msg, allowed := gov.MonitorMessage("agent-1", "agent-2", "hello", 0.3)
	if !allowed {
		t.Fatal("message should be allowed")
	}
	if msg == nil {
		t.Fatal("message is nil")
	}

	_, allowed = gov.MonitorMessage("agent-1", "agent-2", "deceptive", 0.9)
	if allowed {
		t.Fatal("message should be blocked")
	}
}

func TestDriftPropagation(t *testing.T) {
	gov := NewMultiAgentGovernor(0.5)

	gov.RegisterAgent("agent-1", "Agent One")
	gov.RegisterAgent("agent-2", "Agent Two")

	gov.MonitorMessage("agent-1", "agent-2", "msg1", 0.8)
	gov.MonitorMessage("agent-2", "agent-1", "msg2", 0.3)

	drifts := gov.DetectDriftPropagation()
	if len(drifts) != 1 {
		t.Fatalf("len(drifts) = %d, want 1", len(drifts))
	}
	if drifts[0] != "agent-1" {
		t.Errorf("drift = %s, want agent-1", drifts[0])
	}
}

func TestAddPolicy(t *testing.T) {
	gov := NewMultiAgentGovernor(0.7)

	gov.AddPolicy(GovernancePolicy{
		Name:              "custom_policy",
		Description:       "Custom policy",
		MaxDeceptionScore: 0.5,
		Action:            "alert",
	})

	if len(gov.policies) != 2 {
		t.Fatalf("len(policies) = %d, want 2", len(gov.policies))
	}
}

func TestGetMessageCount(t *testing.T) {
	gov := NewMultiAgentGovernor(0.7)

	gov.RegisterAgent("agent-1", "Agent One")
	gov.MonitorMessage("agent-1", "agent-2", "msg1", 0.3)
	gov.MonitorMessage("agent-1", "agent-2", "msg2", 0.4)

	if gov.GetMessageCount() != 2 {
		t.Fatalf("GetMessageCount = %d, want 2", gov.GetMessageCount())
	}
}

func TestGetAgentInfo(t *testing.T) {
	gov := NewMultiAgentGovernor(0.7)

	gov.RegisterAgent("agent-1", "Agent One")

	info, ok := gov.GetAgentInfo("agent-1")
	if !ok {
		t.Fatal("agent not found")
	}
	if info.Name != "Agent One" {
		t.Errorf("name = %s, want Agent One", info.Name)
	}

	_, ok = gov.GetAgentInfo("nonexistent")
	if ok {
		t.Fatal("should not find nonexistent agent")
	}
}

func TestNewMiddleware_NoModelID(t *testing.T) {
	_, err := NewMiddleware(Config{ModelID: ""})
	if err == nil {
		t.Error("expected error for empty ModelID")
	}
}

func TestMiddleware_ShouldBlockAndGetMonitoringInterval(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.6, MonitoringInterval: 5 * time.Second})

	if mw.GetMonitoringInterval() != 5*time.Second {
		t.Errorf("GetMonitoringInterval = %v, want 5s", mw.GetMonitoringInterval())
	}

	if !mw.ShouldBlock(map[string]float64{"deception": 0.8}) {
		t.Error("expected ShouldBlock to return true for score > threshold")
	}

	if mw.ShouldBlock(map[string]float64{"deception": 0.4}) {
		t.Error("expected ShouldBlock to return false for score < threshold")
	}
}

func TestCrewAITaskDecorator_MonitorToolUse(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})
	decorator := NewCrewAITaskDecorator(mw, "sess-001")

	err := decorator.MonitorToolUse(context.Background(), "calculator", "2+2", "4")
	if err != nil {
		t.Fatalf("MonitorToolUse failed: %v", err)
	}
}

func TestLangChainCallback(t *testing.T) {
	mw, _ := NewMiddleware(Config{ModelID: "test", ScoringThreshold: 0.7})
	callback := NewLangChainCallback(mw, "sess-001")

	ctx := context.Background()

	if err := callback.OnLLMStart(ctx, "hello"); err != nil {
		t.Errorf("OnLLMStart failed: %v", err)
	}

	if err := callback.OnLLMEnd(ctx, "world"); err != nil {
		t.Errorf("OnLLMEnd failed: %v", err)
	}

	if err := callback.OnChainStart(ctx, nil); err != nil {
		t.Errorf("OnChainStart failed: %v", err)
	}

	if err := callback.OnChainEnd(ctx, nil); err != nil {
		t.Errorf("OnChainEnd failed: %v", err)
	}

	if err := callback.OnToolStart(ctx, "", ""); err != nil {
		t.Errorf("OnToolStart failed: %v", err)
	}

	if err := callback.OnToolEnd(ctx, "", ""); err != nil {
		t.Errorf("OnToolEnd failed: %v", err)
	}
}
