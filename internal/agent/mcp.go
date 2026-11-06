package agent

import (
	"context"
	"fmt"
	"sync"
	"time"

	"burnnotice/internal/policy"
)

// MCPMiddleware intercepts MCP tool calls and responses, optionally enforcing
// Latent-Aware Authorization via an Open Policy Agent (OPA) evaluator.
type MCPMiddleware struct {
	mu           sync.RWMutex
	middleware   *Middleware
	sessionID    string
	opaEvaluator *policy.OPAEvaluator
}

// NewMCPMiddleware creates a new MCP middleware.
func NewMCPMiddleware(mw *Middleware, sessionID string) *MCPMiddleware {
	return &MCPMiddleware{
		middleware: mw,
		sessionID:  sessionID,
	}
}

// SetOPAEvaluator attaches an OPAEvaluator to enforce Latent-Aware Authorization on MCP tool calls.
func (m *MCPMiddleware) SetOPAEvaluator(opa *policy.OPAEvaluator) {
	m.mu.Lock()
	defer m.mu.Unlock()
	m.opaEvaluator = opa
}

// InterceptToolCall monitors tool calls before execution.
func (m *MCPMiddleware) InterceptToolCall(ctx context.Context, toolName string, arguments map[string]interface{}) error {
	req := Request{
		Prompt:    toolName,
		SessionID: m.sessionID,
		Timestamp: time.Now(),
		Metadata:  map[string]string{"type": "tool_call"},
	}
	scores, err := m.middleware.InterceptRequest(ctx, req)
	if err != nil {
		return err
	}

	m.mu.RLock()
	opa := m.opaEvaluator
	m.mu.RUnlock()

	if opa != nil {
		dec, evalErr := opa.Evaluate(ctx, policy.OPAInput{
			Latent: scores,
			MCP: &policy.OPAMCPContext{
				ToolName:  toolName,
				Arguments: arguments,
			},
			Agent: &policy.OPAAgentContext{
				SessionID: m.sessionID,
			},
		})
		if evalErr != nil {
			return fmt.Errorf("OPA tool authorization error: %w", evalErr)
		}
		if !dec.Allow {
			return fmt.Errorf("MCP tool %q blocked by OPA policy (%s): %s", toolName, dec.PolicyID, dec.Reason)
		}
	}
	return nil
}

// InterceptToolCallWithLatent evaluates an MCP tool invocation against explicit latent activation
// telemetry and tool blast-radius metadata using the configured OPAEvaluator.
func (m *MCPMiddleware) InterceptToolCallWithLatent(
	ctx context.Context,
	toolName string,
	blastRadius string,
	arguments map[string]interface{},
	latentScores map[string]float64,
) (policy.OPADecision, error) {
	m.mu.RLock()
	opa := m.opaEvaluator
	m.mu.RUnlock()

	if opa == nil {
		if m.middleware != nil && m.middleware.ShouldBlock(latentScores) {
			return policy.OPADecision{
				Allow:  false,
				Action: policy.ActionBlock,
				Reason: "scalar_threshold_exceeded",
			}, fmt.Errorf("MCP tool %q blocked by scalar threshold", toolName)
		}
		return policy.OPADecision{Allow: true, Action: policy.ActionAllow, Reason: "no_opa_configured"}, nil
	}

	dec, err := opa.Evaluate(ctx, policy.OPAInput{
		Latent: latentScores,
		MCP: &policy.OPAMCPContext{
			ToolName:    toolName,
			Arguments:   arguments,
			BlastRadius: blastRadius,
		},
		Agent: &policy.OPAAgentContext{
			SessionID: m.sessionID,
		},
	})
	if err != nil {
		return dec, fmt.Errorf("OPA tool evaluation failed: %w", err)
	}
	if !dec.Allow {
		return dec, fmt.Errorf("MCP tool %q blocked by OPA (%s): %s", toolName, dec.PolicyID, dec.Reason)
	}
	return dec, nil
}

// InterceptToolResponse monitors tool responses after execution.
func (m *MCPMiddleware) InterceptToolResponse(ctx context.Context, toolName string, result interface{}) error {
	resp := Response{
		Text:      "",
		SessionID: m.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
		Metadata:  map[string]string{"tool": toolName, "type": "tool_response"},
	}
	_, err := m.middleware.InterceptResponse(ctx, resp)
	return err
}
