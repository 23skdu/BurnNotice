package policy

import (
	"fmt"
	"os"

	"gopkg.in/yaml.v3"
)

// Action defines what to do when a policy is violated.
type Action string

const (
	ActionSteer     Action = "steer"
	ActionBlock     Action = "block"
	ActionLog       Action = "log"
	ActionTerminate Action = "terminate"
)

// Policy defines a single monitoring rule.
type Policy struct {
	Name             string  `yaml:"name"`
	Metric           string  `yaml:"metric"`
	Threshold        float64 `yaml:"threshold"`
	Action           Action  `yaml:"action"`
	SteeringVector   string  `yaml:"steering_vector,omitempty"`
	SteeringStrength float64 `yaml:"steering_strength,omitempty"`
	EscalationCount  int     `yaml:"escalation_count,omitempty"`
}

// LoadPolicies reads policies from a YAML config file.
func LoadPolicies(path string) ([]Policy, error) {
	// #nosec G304 -- CLI tool accepts user-provided config paths
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read policy file: %w", err)
	}

	var config struct {
		Policies []Policy `yaml:"policies"`
	}
	if err := yaml.Unmarshal(data, &config); err != nil {
		return nil, fmt.Errorf("parse policy file: %w", err)
	}

	for i, p := range config.Policies {
		if p.Name == "" {
			return nil, fmt.Errorf("policy[%d]: name is required", i)
		}
		if p.Metric == "" {
			return nil, fmt.Errorf("policy[%d]: metric is required", i)
		}
		if p.Action == "" {
			return nil, fmt.Errorf("policy[%d]: action is required", i)
		}
	}

	return config.Policies, nil
}
