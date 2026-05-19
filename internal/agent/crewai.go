package agent

import (
	"context"
	"time"
)

// CrewAITaskDecorator wraps CrewAI task execution with monitoring.
type CrewAITaskDecorator struct {
	middleware *Middleware
	sessionID  string
}

// NewCrewAITaskDecorator creates a new CrewAI task decorator.
func NewCrewAITaskDecorator(mw *Middleware, sessionID string) *CrewAITaskDecorator {
	return &CrewAITaskDecorator{
		middleware: mw,
		sessionID:  sessionID,
	}
}

// ExecuteTask wraps task execution with monitoring.
func (d *CrewAITaskDecorator) ExecuteTask(ctx context.Context, taskName string, execute func() (string, error)) (string, error) {
	// Pre-task monitoring
	req := Request{
		Prompt:    taskName,
		SessionID: d.sessionID,
		Timestamp: time.Now(),
	}
	_, _ = d.middleware.InterceptRequest(ctx, req)

	// Execute task
	result, err := execute()
	if err != nil {
		return result, err
	}

	// Post-task monitoring
	resp := Response{
		Text:      result,
		SessionID: d.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
	}
	_, _ = d.middleware.InterceptResponse(ctx, resp)

	return result, nil
}

// MonitorToolUse monitors agent tool usage.
func (d *CrewAITaskDecorator) MonitorToolUse(ctx context.Context, toolName string, toolInput string, toolOutput string) error {
	resp := Response{
		Text:      toolOutput,
		SessionID: d.sessionID,
		Timestamp: time.Now(),
		Scores:    make(map[string]float64),
		Metadata:  map[string]string{"tool": toolName},
	}
	_, err := d.middleware.InterceptResponse(ctx, resp)
	return err
}
