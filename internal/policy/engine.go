package policy

import (
	"context"
	"math"
	"sync"
	"time"
)

// PolicyViolation represents a triggered policy rule.
type PolicyViolation struct {
	PolicyName string
	Metric     string
	Score      float64
	Threshold  float64
	Action     Action
	Timestamp  time.Time
}

// Engine evaluates scores against registered policies.
type Engine struct {
	mu           sync.RWMutex
	policies     []Policy
	violations   map[string]int // policy_name -> violation count
	killSwitch   *KillSwitch
	opaEvaluator *OPAEvaluator
}

// NewEngine creates a policy engine with the given policies.
func NewEngine(policies []Policy) *Engine {
	return &Engine{
		policies:   policies,
		violations: make(map[string]int),
	}
}

// RegisterKillSwitch registers an emergency kill switch with the engine.
func (e *Engine) RegisterKillSwitch(ks *KillSwitch) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.killSwitch = ks
}

// SetOPAEvaluator registers an Open Policy Agent evaluator for contextual,
// multi-signal authorization across latent scores, MCP tools, CoT phases, and agents.
func (e *Engine) SetOPAEvaluator(opa *OPAEvaluator) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.opaEvaluator = opa
}

// OPAEvaluator returns the currently registered OPAEvaluator, if any.
func (e *Engine) OPAEvaluator() *OPAEvaluator {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.opaEvaluator
}

// EvaluateContext evaluates both fast scalar thresholds and the registered OPAEvaluator
// against a rich contextual input (latent scores + MCP tool metadata + CoT state + agent tiers).
func (e *Engine) EvaluateContext(ctx context.Context, input OPAInput) (OPADecision, []PolicyViolation, error) {
	violations := e.Evaluate(input.Latent)

	e.mu.RLock()
	opa := e.opaEvaluator
	ks := e.killSwitch
	e.mu.RUnlock()

	if opa == nil {
		dec := OPADecision{Allow: true, Action: ActionAllow, Reason: "scalar_pass"}
		for _, v := range violations {
			dec.Action = v.Action
			dec.PolicyID = v.PolicyName
			dec.Reason = "scalar_violation:" + v.PolicyName
			if v.Action == ActionBlock || v.Action == ActionTerminate {
				dec.Allow = false
				break
			}
		}
		return dec, violations, nil
	}

	dec, err := opa.Evaluate(ctx, input)
	if dec.Action != "" && dec.Action != ActionAllow {
		polName := dec.PolicyID
		if polName == "" {
			polName = "opa_policy"
		}
		e.mu.Lock()
		e.violations[polName]++
		e.mu.Unlock()

		violations = append(violations, PolicyViolation{
			PolicyName: polName,
			Metric:     "opa_contextual",
			Score:      input.Latent["deception"],
			Threshold:  0,
			Action:     dec.Action,
			Timestamp:  time.Now(),
		})
		if dec.Action == ActionTerminate && ks != nil {
			ks.Trigger(polName + " OPA violation: emergency terminate triggered")
		}
	}
	return dec, violations, err
}

// Evaluate checks all policies against the provided scores and returns violations.
func (e *Engine) Evaluate(scores map[string]float64) []PolicyViolation {
	e.mu.Lock()
	defer e.mu.Unlock()

	var violations []PolicyViolation
	for _, p := range e.policies {
		score, ok := scores[p.Metric]
		if !ok {
			continue
		}
		violated := false
		if p.Operator == "less" {
			violated = score < p.Threshold
		} else {
			violated = score > p.Threshold
		}
		if violated {
			e.violations[p.Name]++
			violations = append(violations, PolicyViolation{
				PolicyName: p.Name,
				Metric:     p.Metric,
				Score:      score,
				Threshold:  p.Threshold,
				Action:     p.Action,
				Timestamp:  time.Now(),
			})
			if p.Action == ActionTerminate && e.killSwitch != nil {
				e.killSwitch.Trigger(p.Name + " violation: emergency terminate triggered")
			}
		}
	}
	return violations
}

// ViolationCount returns the number of times a policy has been violated.
func (e *Engine) ViolationCount(policyName string) int {
	e.mu.RLock()
	defer e.mu.RUnlock()
	return e.violations[policyName]
}

// ShouldEscalate returns true if the policy has been violated >= escalation_count times.
func (e *Engine) ShouldEscalate(policyName string) bool {
	e.mu.RLock()
	defer e.mu.RUnlock()

	for _, p := range e.policies {
		if p.Name == policyName && p.EscalationCount > 0 {
			return e.violations[policyName] >= p.EscalationCount
		}
	}
	return false
}

// ResetViolations clears all violation counters.
func (e *Engine) ResetViolations() {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.violations = make(map[string]int)
}

// UpdatePolicies replaces the current policy set.
func (e *Engine) UpdatePolicies(policies []Policy) {
	e.mu.Lock()
	defer e.mu.Unlock()
	e.policies = policies
}

// UpdateThresholds dynamically updates policy thresholds.
func (e *Engine) UpdateThresholds(thresholds map[string]float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i, p := range e.policies {
		if t, ok := thresholds[p.Metric]; ok && !math.IsNaN(t) && !math.IsInf(t, 0) {
			e.policies[i].Threshold = t
		}
	}
}

// AdjustThresholdsForDrift offsets policy thresholds to counter activation drifts.
func (e *Engine) AdjustThresholdsForDrift(drifts map[string]float64) {
	e.mu.Lock()
	defer e.mu.Unlock()
	for i, p := range e.policies {
		if drift, ok := drifts[p.Metric]; ok && !math.IsNaN(drift) && !math.IsInf(drift, 0) {
			e.policies[i].Threshold += drift
		}
	}
}
