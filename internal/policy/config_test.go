package policy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadPolicies_AllActions(t *testing.T) {
	content := `policies:
  - name: steer_policy
    metric: deception
    threshold: 0.5
    action: steer
  - name: block_policy
    metric: conflict
    threshold: 0.6
    action: block
  - name: log_policy
    metric: truth
    threshold: 0.3
    action: log
  - name: terminate_policy
    metric: compliance
    threshold: 0.9
    action: terminate
`
	tmp := writeTempFileConfig(t, content)
	defer func() { _ = os.Remove(tmp) }()

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if len(policies) != 4 {
		t.Fatalf("len(policies) = %d, want 4", len(policies))
	}

	expectedActions := []Action{ActionSteer, ActionBlock, ActionLog, ActionTerminate}
	for i, p := range policies {
		if p.Action != expectedActions[i] {
			t.Errorf("policy[%d].Action = %s, want %s", i, p.Action, expectedActions[i])
		}
	}
}

func TestLoadPolicies_EmptyPolicies(t *testing.T) {
	content := `policies: []`
	tmp := writeTempFileConfig(t, content)
	defer func() { _ = os.Remove(tmp) }()

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if len(policies) != 0 {
		t.Errorf("len(policies) = %d, want 0", len(policies))
	}
}

func TestLoadPolicies_OptionalFields(t *testing.T) {
	content := `policies:
  - name: minimal
    metric: score
    threshold: 0.5
    action: log
`
	tmp := writeTempFileConfig(t, content)
	defer func() { _ = os.Remove(tmp) }()

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if policies[0].SteeringVector != "" {
		t.Error("steering_vector should be empty when not specified")
	}
	if policies[0].SteeringStrength != 0 {
		t.Error("steering_strength should be 0 when not specified")
	}
	if policies[0].EscalationCount != 0 {
		t.Error("escalation_count should be 0 when not specified")
	}
}

func TestLoadPolicies_InvalidAction(t *testing.T) {
	content := `policies:
  - name: test
    metric: score
    threshold: 0.5
    action: invalid_action
`
	tmp := writeTempFileConfig(t, content)
	defer func() { _ = os.Remove(tmp) }()

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if policies[0].Action != "invalid_action" {
		t.Errorf("Action = %s, want invalid_action", policies[0].Action)
	}
}

func writeTempFileConfig(t *testing.T, content string) string {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "config.yaml")
	if err := os.WriteFile(tmp, []byte(content), 0600); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return tmp
}
