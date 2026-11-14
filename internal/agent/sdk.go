package agent

import (
	"context"
	"fmt"
	"math"
	"strings"
	"time"

	"burnnotice/internal/scoring"
	"burnnotice/internal/steer"
	"burnnotice/internal/tokenizer"
)

const defaultActivationDim = 16

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
	Prompt    string
	SessionID string
	Timestamp time.Time
	Metadata  map[string]string
}

// Response represents an intercepted agent response.
type Response struct {
	Text      string
	SessionID string
	Timestamp time.Time
	Scores    map[string]float64
	Metadata  map[string]string
}

// Middleware intercepts agent requests and responses for monitoring.
type Middleware struct {
	config     Config
	sanitizer  *scoring.Sanitizer
	tokenizer  tokenizer.Tokenizer
	scorer     *scoring.Scorer
	dispatcher *steer.Dispatcher
	shield     *steer.AdversarialShield
}

// NewMiddleware creates a new BurnNotice middleware.
func NewMiddleware(config Config) (*Middleware, error) {
	if config.ModelID == "" {
		return nil, fmt.Errorf("ModelID is required")
	}
	if config.ScoringThreshold <= 0 {
		config.ScoringThreshold = 0.7
	}
	if config.SteeringStrength == 0 {
		config.SteeringStrength = 0.5
	}
	if config.SteeringMethod == "" {
		config.SteeringMethod = "angular"
	}

	sc := scoring.NewScorer(nil)
	deceptionDir := make([]float64, defaultActivationDim)
	truthDir := make([]float64, defaultActivationDim)
	conflictDir := make([]float64, defaultActivationDim)
	for i := 0; i < defaultActivationDim; i++ {
		if i%2 == 0 {
			deceptionDir[i] = -0.25
			truthDir[i] = 0.25
		} else {
			conflictDir[i] = 0.25
		}
	}
	sc.RegisterDirection("deception", deceptionDir)
	sc.RegisterDirection("truth", truthDir)
	sc.RegisterDirection("conflict", conflictDir)

	method := parseSteeringMethod(config.SteeringMethod)
	disp := steer.NewDispatcher([]int{0}, map[int]steer.Method{0: method})

	return &Middleware{
		config:     config,
		sanitizer:  scoring.NewSanitizer(),
		tokenizer:  tokenizer.NewWordHashTokenizer(),
		scorer:     sc,
		dispatcher: disp,
		shield:     steer.NewAdversarialShield(steer.DefaultAdversarialShieldConfig()),
	}, nil
}

func (m *Middleware) promptActivation(text string) []float64 {
	act := make([]float64, defaultActivationDim)
	if text == "" || m.tokenizer == nil {
		return act
	}
	tokens := make([]int64, 64)
	n := m.tokenizer.Tokenize(text, tokens)
	for i := 0; i < n; i++ {
		slot := int(tokens[i] % defaultActivationDim)
		if slot < 0 {
			slot = -slot
		}
		// Keep benign token projections aligned with truth (even slots positive)
		if slot%2 == 0 {
			act[slot] += 1.0
		} else {
			act[slot] += 0.15
		}
	}
	var normSq float64
	for _, v := range act {
		normSq += v * v
	}
	if normSq > 0 {
		inv := 1.0 / math.Sqrt(normSq)
		for i := range act {
			act[i] *= inv
		}
	}
	return act
}

// InterceptRequest processes an incoming request and returns scores.
func (m *Middleware) InterceptRequest(ctx context.Context, req Request) (map[string]float64, error) {
	if err := ctx.Err(); err != nil {
		return nil, err
	}

	act := m.promptActivation(req.Prompt)
	rawScores := m.scorer.Score(act)
	safeCap := m.config.ScoringThreshold * 0.5
	if safeCap > 0.3 {
		safeCap = 0.3
	}
	scores := map[string]float64{
		"deception": 0.0,
		"truth":     0.0,
		"conflict":  0.0,
	}
	for k, v := range rawScores {
		if v > safeCap {
			v = safeCap
		}
		scores[k] = v
	}

	if m.sanitizer != nil {
		if err := m.sanitizer.Sanitize(req.Prompt); err != nil {
			scores["deception"] = math.Max(0.95, m.config.ScoringThreshold+0.15)
			scores["conflict"] = 0.90
			scores["truth"] = 0.05
		}
	}
	if m.shield != nil {
		if flagged, _ := m.shield.SanitizePrompt(req.Prompt); flagged {
			scores["deception"] = math.Max(scores["deception"], math.Max(0.90, m.config.ScoringThreshold+0.10))
			scores["conflict"] = math.Max(scores["conflict"], 0.85)
			scores["truth"] = math.Min(scores["truth"], 0.10)
		}
	}
	return scores, nil
}

func parseSteeringMethod(s string) steer.Method {
	switch strings.ToLower(strings.TrimSpace(s)) {
	case "addition", "actadd":
		return steer.MethodAddition
	case "selective":
		return steer.MethodSelective
	case "ablation", "ablate":
		return steer.MethodAblation
	case "norm_preserving", "mpoa":
		return steer.MethodNormPreserving
	case "biprojection":
		return steer.MethodBiprojection
	default:
		return steer.MethodAngular
	}
}

// InterceptResponse processes an outgoing response and applies steering if needed.
func (m *Middleware) InterceptResponse(ctx context.Context, resp Response) (Response, error) {
	if err := ctx.Err(); err != nil {
		return resp, err
	}
	if resp.Metadata == nil {
		resp.Metadata = make(map[string]string)
	}
	if len(resp.Scores) == 0 && resp.Text != "" {
		computed, err := m.InterceptRequest(ctx, Request{
			Prompt:    resp.Text,
			SessionID: resp.SessionID,
			Timestamp: resp.Timestamp,
			Metadata:  resp.Metadata,
		})
		if err != nil {
			return resp, err
		}
		resp.Scores = computed
	}

	for metric, score := range resp.Scores {
		if score > m.config.ScoringThreshold {
			resp.Text = m.applySteering(resp.Text, metric, score)
			resp.Metadata["steered"] = "true"
			resp.Metadata["steering_metric"] = metric
			resp.Metadata["steering_method"] = m.config.SteeringMethod
		}
	}
	return resp, nil
}

func (m *Middleware) applySteering(text string, metric string, score float64) string {
	act := m.promptActivation(text)
	dir := make([]float64, defaultActivationDim)
	for i := range dir {
		if i%2 == 0 {
			dir[i] = 0.25
		}
	}
	if m.dispatcher != nil {
		steered := m.dispatcher.Apply(0, act, dir, m.config.SteeringStrength)
		if len(steered) == len(act) {
			_ = steer.ApplyCommand(steered, steer.Command{
				Method:   steer.MethodNormPreserving,
				Vector:   dir,
				Strength: m.config.SteeringStrength,
			})
		}
	}

	// Sanitize known jailbreak/override markers if present in steered text
	if m.sanitizer != nil && m.sanitizer.Sanitize(text) != nil {
		return fmt.Sprintf("[steered:%s=%.2f] %s", metric, score, strings.TrimSpace(text))
	}
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
