package agent

import (
	"context"
	"fmt"
	"sort"
	"strings"
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
		Metadata:  map[string]string{"type": "llm_start"},
	}
	scores, err := c.middleware.InterceptRequest(ctx, req)
	if err != nil {
		return err
	}
	if c.middleware.ShouldBlock(scores) {
		return fmt.Errorf("LLM start blocked by policy threshold")
	}
	return nil
}

// OnLLMEnd is called when LLM generation ends.
func (c *LangChainCallback) OnLLMEnd(ctx context.Context, output string) error {
	resp := Response{
		Text:      output,
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
		Metadata:  map[string]string{"type": "llm_end"},
	}
	_, err := c.middleware.InterceptResponse(ctx, resp)
	return err
}

// OnChainStart is called when a chain starts.
func (c *LangChainCallback) OnChainStart(ctx context.Context, inputs map[string]interface{}) error {
	req := Request{
		Prompt:    flattenMapText(inputs),
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Metadata:  map[string]string{"type": "chain_start"},
	}
	scores, err := c.middleware.InterceptRequest(ctx, req)
	if err != nil {
		return err
	}
	if c.middleware.ShouldBlock(scores) {
		return fmt.Errorf("chain start blocked by policy threshold")
	}
	return nil
}

// OnChainEnd is called when a chain ends.
func (c *LangChainCallback) OnChainEnd(ctx context.Context, outputs map[string]interface{}) error {
	resp := Response{
		Text:      flattenMapText(outputs),
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
		Metadata:  map[string]string{"type": "chain_end"},
	}
	_, err := c.middleware.InterceptResponse(ctx, resp)
	return err
}

// OnToolStart is called when a tool execution starts.
func (c *LangChainCallback) OnToolStart(ctx context.Context, toolName string, input string) error {
	req := Request{
		Prompt:    strings.TrimSpace(toolName + " " + input),
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Metadata:  map[string]string{"type": "tool_start", "tool": toolName},
	}
	scores, err := c.middleware.InterceptRequest(ctx, req)
	if err != nil {
		return err
	}
	if c.middleware.ShouldBlock(scores) {
		return fmt.Errorf("tool %q start blocked by policy threshold", toolName)
	}
	return nil
}

// OnToolEnd is called when a tool execution ends.
func (c *LangChainCallback) OnToolEnd(ctx context.Context, toolName string, output string) error {
	resp := Response{
		Text:      output,
		SessionID: c.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
		Metadata:  map[string]string{"type": "tool_end", "tool": toolName},
	}
	_, err := c.middleware.InterceptResponse(ctx, resp)
	return err
}

func flattenMapText(m map[string]interface{}) string {
	if len(m) == 0 {
		return ""
	}
	keys := make([]string, 0, len(m))
	for k := range m {
		keys = append(keys, k)
	}
	sort.Strings(keys)
	parts := make([]string, 0, len(keys))
	for _, k := range keys {
		parts = append(parts, fmt.Sprintf("%s=%v", k, m[k]))
	}
	return strings.Join(parts, " ")
}
