package agent

import (
	"context"
	"testing"
	"time"
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
