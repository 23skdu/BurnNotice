package agent

import (
	"context"
	"time"
)

// LangChainCallback implements LangChain callback handler interface.
type LangChainCallback struct {
	middleware *Middleware
	sessionID  string
}

// NewLangChainCallback creates a new LangChain callback handler.
func NewLangChainCallback(mw *Middleware, sessionID string) *LangChainCallback {
	return &LangChainCallback{
		middleware: mw,
		sessionID:  sessionID,
	}
}

// OnLLMStart is called when LLM generation starts.
func (c *LangChainCallback) OnLLMStart(ctx context.Context, prompt string) error {
	req := Request{
		Prompt:    prompt,
		SessionID: c.sessionID,
		Timestamp: time.Now(),
	}
	_, err := c.middleware.InterceptRequest(ctx, req)
	return err
}

// OnLLMEnd is called when LLM generation ends.
func (c *LangChainCallback) OnLLMEnd(ctx context.Context, output string) error {
	resp := Response{
		Text:      output,
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
	}
	_, err := c.middleware.InterceptResponse(ctx, resp)
	return err
}

// OnChainStart is called when a chain starts.
func (c *LangChainCallback) OnChainStart(ctx context.Context, inputs map[string]interface{}) error {
	return nil
}

// OnChainEnd is called when a chain ends.
func (c *LangChainCallback) OnChainEnd(ctx context.Context, outputs map[string]interface{}) error {
	return nil
}

// OnToolStart is called when a tool execution starts.
func (c *LangChainCallback) OnToolStart(ctx context.Context, toolName string, input string) error {
	return nil
}

// OnToolEnd is called when a tool execution ends.
func (c *LangChainCallback) OnToolEnd(ctx context.Context, toolName string, output string) error {
	return nil
}
