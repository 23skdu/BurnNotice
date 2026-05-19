package agent

import (
	"context"
	"fmt"
	"time"
)

// Config holds the agent SDK configuration.
type Config struct {
	ModelID            string
	ScoringThreshold   float64
	SteeringStrength   float64
	SteeringMethod     string
	MonitoringInterval time.Duration
}

// Request represents an intercepted agent request.
type Request struct {
	Prompt     string
	SessionID  string
	Timestamp  time.Time
	Metadata   map[string]string
}

// Response represents an intercepted agent response.
type Response struct {
	Text       string
	SessionID  string
	Timestamp  time.Time
	Scores     map[string]float64
	Metadata   map[string]string
}

// Middleware intercepts agent requests and responses for monitoring.
type Middleware struct {
	config   Config
	scorer   interface{}
	steerer  interface{}
	ledger   interface{}
}

// NewMiddleware creates a new BurnNotice middleware.
func NewMiddleware(config Config) (*Middleware, error) {
	if config.ModelID == "" {
		return nil, fmt.Errorf("ModelID is required")
	}
	if config.ScoringThreshold <= 0 {
		config.ScoringThreshold = 0.7
	}
	return &Middleware{config: config}, nil
}

// InterceptRequest processes an incoming request and returns scores.
func (m *Middleware) InterceptRequest(ctx context.Context, req Request) (map[string]float64, error) {
	// In production: run prompt through ONNX model, extract activations, score
	scores := map[string]float64{
		"deception": 0.0,
		"truth":     0.0,
		"conflict":  0.0,
	}
	return scores, nil
}

// InterceptResponse processes an outgoing response and applies steering if needed.
func (m *Middleware) InterceptResponse(ctx context.Context, resp Response) (Response, error) {
	if resp.Metadata == nil {
		resp.Metadata = make(map[string]string)
	}

	for metric, score := range resp.Scores {
		if score > m.config.ScoringThreshold {
			resp.Text = m.applySteering(resp.Text, metric, score)
			resp.Metadata["steered"] = "true"
			resp.Metadata["steering_metric"] = metric
		}
	}
	return resp, nil
}

func (m *Middleware) applySteering(text string, _ string, _ float64) string {
	// In production: apply steering vector to model activations
	return text
}

// ShouldBlock returns true if the response should be blocked.
func (m *Middleware) ShouldBlock(scores map[string]float64) bool {
	for _, score := range scores {
		if score > m.config.ScoringThreshold {
			return true
		}
	}
	return false
}

// GetMonitoringInterval returns the configured monitoring interval.
func (m *Middleware) GetMonitoringInterval() time.Duration {
	return m.config.MonitoringInterval
}
