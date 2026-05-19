package agent

import (
	"context"
	"time"
)

// MCPMiddleware intercepts MCP tool calls and responses.
type MCPMiddleware struct {
	middleware *Middleware
	sessionID  string
}

// NewMCPMiddleware creates a new MCP middleware.
func NewMCPMiddleware(mw *Middleware, sessionID string) *MCPMiddleware {
	return &MCPMiddleware{
		middleware: mw,
		sessionID:  sessionID,
	}
}

// InterceptToolCall monitors tool calls before execution.
func (m *MCPMiddleware) InterceptToolCall(ctx context.Context, toolName string, arguments map[string]interface{}) error {
	req := Request{
		Prompt:    toolName,
		SessionID: m.sessionID,
		Timestamp: time.Now(),
		Metadata:  map[string]string{"type": "tool_call"},
	}
	_, err := m.middleware.InterceptRequest(ctx, req)
	return err
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
