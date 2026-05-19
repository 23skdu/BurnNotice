package policy

import (
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
	mu         sync.RWMutex
	policies   []Policy
	violations map[string]int // policy_name -> violation count
	killSwitch *KillSwitch
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
