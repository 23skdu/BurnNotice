package policy

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoadPolicies_Valid(t *testing.T) {
	content := `policies:
  - name: deception_threshold
    metric: deception_score
    threshold: 0.7
    action: steer
    steering_vector: truth
    steering_strength: 0.5
  - name: conflict_alert
    metric: conflict_score
    threshold: 0.8
    action: block
    escalation_count: 3
`
	tmp := writeTempFile(t, content)
	defer os.Remove(tmp)

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if len(policies) != 2 {
		t.Fatalf("len(policies) = %d, want 2", len(policies))
	}

	if policies[0].Name != "deception_threshold" {
		t.Errorf("policy[0].Name = %s, want deception_threshold", policies[0].Name)
	}
	if policies[0].Threshold != 0.7 {
		t.Errorf("policy[0].Threshold = %f, want 0.7", policies[0].Threshold)
	}
	if policies[0].Action != ActionSteer {
		t.Errorf("policy[0].Action = %s, want steer", policies[0].Action)
	}
	if policies[1].EscalationCount != 3 {
		t.Errorf("policy[1].EscalationCount = %d, want 3", policies[1].EscalationCount)
	}
}

func TestLoadPolicies_MissingFile(t *testing.T) {
	_, err := LoadPolicies("/nonexistent/policies.yaml")
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}

func TestLoadPolicies_InvalidYAML(t *testing.T) {
	tmp := writeTempFile(t, "invalid: yaml: [")
	defer os.Remove(tmp)

	_, err := LoadPolicies(tmp)
	if err == nil {
		t.Fatal("expected error for invalid YAML")
	}
}

func TestLoadPolicies_MissingName(t *testing.T) {
	content := `policies:
  - metric: deception_score
    threshold: 0.7
    action: steer
`
	tmp := writeTempFile(t, content)
	defer os.Remove(tmp)

	_, err := LoadPolicies(tmp)
	if err == nil {
		t.Fatal("expected error for missing name")
	}
}

func TestLoadPolicies_MissingMetric(t *testing.T) {
	content := `policies:
  - name: test_policy
    threshold: 0.7
    action: steer
`
	tmp := writeTempFile(t, content)
	defer os.Remove(tmp)

	_, err := LoadPolicies(tmp)
	if err == nil {
		t.Fatal("expected error for missing metric")
	}
}

func TestLoadPolicies_MissingAction(t *testing.T) {
	content := `policies:
  - name: test_policy
    metric: deception_score
    threshold: 0.7
`
	tmp := writeTempFile(t, content)
	defer os.Remove(tmp)

	_, err := LoadPolicies(tmp)
	if err == nil {
		t.Fatal("expected error for missing action")
	}
}

func TestEngine_Evaluate_NoViolations(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "deception_score", Threshold: 0.7, Action: ActionSteer},
	}
	engine := NewEngine(policies)

	scores := map[string]float64{"deception_score": 0.5}
	violations := engine.Evaluate(scores)

	if len(violations) != 0 {
		t.Errorf("expected 0 violations, got %d", len(violations))
	}
}

func TestEngine_Evaluate_Violation(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "deception_score", Threshold: 0.7, Action: ActionBlock},
	}
	engine := NewEngine(policies)

	scores := map[string]float64{"deception_score": 0.9}
	violations := engine.Evaluate(scores)

	if len(violations) != 1 {
		t.Fatalf("expected 1 violation, got %d", len(violations))
	}

	v := violations[0]
	if v.PolicyName != "test" {
		t.Errorf("PolicyName = %s, want test", v.PolicyName)
	}
	if v.Score != 0.9 {
		t.Errorf("Score = %f, want 0.9", v.Score)
	}
	if v.Action != ActionBlock {
		t.Errorf("Action = %s, want block", v.Action)
	}
}

func TestEngine_Evaluate_MissingMetric(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "missing_metric", Threshold: 0.7, Action: ActionLog},
	}
	engine := NewEngine(policies)

	scores := map[string]float64{"deception_score": 0.9}
	violations := engine.Evaluate(scores)

	if len(violations) != 0 {
		t.Errorf("expected 0 violations for missing metric, got %d", len(violations))
	}
}

func TestEngine_ViolationCount(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "score", Threshold: 0.5, Action: ActionLog},
	}
	engine := NewEngine(policies)

	engine.Evaluate(map[string]float64{"score": 0.6})
	engine.Evaluate(map[string]float64{"score": 0.7})

	count := engine.ViolationCount("test")
	if count != 2 {
		t.Errorf("ViolationCount = %d, want 2", count)
	}
}

func TestEngine_ShouldEscalate(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "score", Threshold: 0.5, Action: ActionBlock, EscalationCount: 3},
	}
	engine := NewEngine(policies)

	engine.Evaluate(map[string]float64{"score": 0.6})
	engine.Evaluate(map[string]float64{"score": 0.7})

	if engine.ShouldEscalate("test") {
		t.Error("should not escalate after 2 violations (threshold=3)")
	}

	engine.Evaluate(map[string]float64{"score": 0.8})

	if !engine.ShouldEscalate("test") {
		t.Error("should escalate after 3 violations (threshold=3)")
	}
}

func TestEngine_ResetViolations(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "score", Threshold: 0.5, Action: ActionLog},
	}
	engine := NewEngine(policies)

	engine.Evaluate(map[string]float64{"score": 0.6})
	engine.ResetViolations()

	if engine.ViolationCount("test") != 0 {
		t.Error("violation count should be 0 after reset")
	}
}

func TestEngine_UpdatePolicies(t *testing.T) {
	engine := NewEngine([]Policy{
		{Name: "old", Metric: "score", Threshold: 0.5, Action: ActionLog},
	})

	engine.UpdatePolicies([]Policy{
		{Name: "new", Metric: "score", Threshold: 0.3, Action: ActionBlock},
	})

	violations := engine.Evaluate(map[string]float64{"score": 0.4})
	if len(violations) != 1 || violations[0].PolicyName != "new" {
		t.Error("updated policies should be used")
	}
}

func TestEngine_Evaluate_MultiplePolicies(t *testing.T) {
	policies := []Policy{
		{Name: "deception", Metric: "deception_score", Threshold: 0.7, Action: ActionSteer},
		{Name: "conflict", Metric: "conflict_score", Threshold: 0.8, Action: ActionBlock},
	}
	engine := NewEngine(policies)

	scores := map[string]float64{"deception_score": 0.9, "conflict_score": 0.9}
	violations := engine.Evaluate(scores)

	if len(violations) != 2 {
		t.Fatalf("expected 2 violations, got %d", len(violations))
	}
}

func TestLoadPolicies_Operator(t *testing.T) {
	content := `policies:
  - name: compliance_threshold
    metric: compliance_score
    threshold: 0.5
    operator: less
    action: block
`
	tmp := writeTempFile(t, content)
	defer os.Remove(tmp)

	policies, err := LoadPolicies(tmp)
	if err != nil {
		t.Fatalf("LoadPolicies() error = %v", err)
	}

	if len(policies) != 1 {
		t.Fatalf("len(policies) = %d, want 1", len(policies))
	}

	if policies[0].Operator != "less" {
		t.Errorf("policy[0].Operator = %s, want less", policies[0].Operator)
	}
}

func TestEngine_Evaluate_LessOperator(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "compliance_score", Threshold: 0.5, Operator: "less", Action: ActionBlock},
	}
	engine := NewEngine(policies)

	// Score is above threshold, should not violate
	scores := map[string]float64{"compliance_score": 0.8}
	violations := engine.Evaluate(scores)
	if len(violations) != 0 {
		t.Errorf("expected 0 violations for score above threshold, got %d", len(violations))
	}

	// Score is below threshold, should violate
	scores = map[string]float64{"compliance_score": 0.3}
	violations = engine.Evaluate(scores)
	if len(violations) != 1 {
		t.Fatalf("expected 1 violation for score below threshold, got %d", len(violations))
	}

	v := violations[0]
	if v.PolicyName != "test" {
		t.Errorf("PolicyName = %s, want test", v.PolicyName)
	}
	if v.Score != 0.3 {
		t.Errorf("Score = %f, want 0.3", v.Score)
	}
}

func TestEngine_Evaluate_ActionTerminate_KillSwitch(t *testing.T) {
	policies := []Policy{
		{Name: "critical_deception", Metric: "deception_score", Threshold: 0.9, Action: ActionTerminate},
	}
	engine := NewEngine(policies)

	var alertReason string
	ks, _ := NewKillSwitch(func(reason string) {
		alertReason = reason
	})
	engine.RegisterKillSwitch(ks)

	// Below threshold - should not terminate
	engine.Evaluate(map[string]float64{"deception_score": 0.5})
	if !ks.IsArmed() {
		t.Error("kill switch should remain armed when score is below threshold")
	}

	// Above threshold - should trigger terminate & disarm kill switch
	engine.Evaluate(map[string]float64{"deception_score": 0.95})
	if ks.IsArmed() {
		t.Error("kill switch should be disarmed after ActionTerminate evaluation")
	}
	if alertReason == "" {
		t.Error("expected alert callback with reason, got empty string")
	}
}

func TestEngine_DynamicThresholds(t *testing.T) {
	policies := []Policy{
		{Name: "test", Metric: "deception_score", Threshold: 0.7, Action: ActionLog},
	}
	engine := NewEngine(policies)

	// Test UpdateThresholds
	engine.UpdateThresholds(map[string]float64{"deception_score": 0.85})
	violations := engine.Evaluate(map[string]float64{"deception_score": 0.8})
	if len(violations) != 0 {
		t.Errorf("expected no violation with updated threshold 0.85, got %d", len(violations))
	}

	// Test AdjustThresholdsForDrift
	engine.AdjustThresholdsForDrift(map[string]float64{"deception_score": -0.1})
	violations2 := engine.Evaluate(map[string]float64{"deception_score": 0.8})
	if len(violations2) != 1 {
		t.Errorf("expected 1 violation after threshold drift adjustment, got %d", len(violations2))
	}
}

func writeTempFile(t *testing.T, content string) string {
	t.Helper()
	tmp := filepath.Join(t.TempDir(), "policies.yaml")
	if err := os.WriteFile(tmp, []byte(content), 0600); err != nil {
		t.Fatalf("write temp file: %v", err)
	}
	return tmp
}
