package steer

import (
	"fmt"
	"math"
)

// ConditionOperator defines logical operators combining multiple category conditions.
type ConditionOperator string

const (
	OpSingle ConditionOperator = "SINGLE"
	OpAND    ConditionOperator = "AND"
	OpOR     ConditionOperator = "OR"
	OpNOT    ConditionOperator = "NOT"
)

// ConditionNode represents an evaluation node in a conditional steering rule graph.
type ConditionNode struct {
	Category       string            `json:"category"`
	Vector         []float64         `json:"vector,omitempty"`
	Threshold      float64           `json:"threshold"`
	Operator       ConditionOperator `json:"operator"`
	Children       []ConditionNode   `json:"children,omitempty"`
	SmoothScaling  bool              `json:"smooth_scaling"` // If true, scale linearly in [threshold, 1.0]; if false, binary step
}

// ConditionalSteerer evaluates prompt and activation context against condition vectors
// to scale or gate intervention without requiring slow external LLM classification passes.
type ConditionalSteerer struct{}

// NewConditionalSteerer constructs a new CAST engine.
func NewConditionalSteerer() *ConditionalSteerer {
	return &ConditionalSteerer{}
}

// Evaluate computes the gating multiplier in [0.0, 1.0] for the activation against the condition tree.
func (cs *ConditionalSteerer) Evaluate(activation []float64, node ConditionNode) (float64, bool, error) {
	if len(activation) == 0 {
		return 0.0, false, fmt.Errorf("empty activation vector")
	}

	switch node.Operator {
	case OpSingle, "":
		if len(node.Vector) == 0 {
			return 1.0, true, nil // Trivial pass
		}
		if len(node.Vector) != len(activation) {
			return 0.0, false, fmt.Errorf("condition vector dim mismatch: %d != %d", len(node.Vector), len(activation))
		}

		dot := vecDot(activation, node.Vector)
		aNorm := vecNorm(activation)
		cNorm := vecNorm(node.Vector)
		if aNorm < 1e-10 || cNorm < 1e-10 {
			return 0.0, false, nil
		}

		sim := dot / (aNorm * cNorm)
		if sim < node.Threshold {
			return 0.0, false, nil
		}

		if node.SmoothScaling {
			span := 1.0 - node.Threshold
			if span < 1e-6 {
				return 1.0, true, nil
			}
			scale := (sim - node.Threshold) / span
			return math.Min(1.0, math.Max(0.0, scale)), true, nil
		}
		return 1.0, true, nil

	case OpAND:
		minScale := 1.0
		for _, child := range node.Children {
			scale, match, err := cs.Evaluate(activation, child)
			if err != nil {
				return 0.0, false, err
			}
			if !match || scale <= 0 {
				return 0.0, false, nil
			}
			if scale < minScale {
				minScale = scale
			}
		}
		return minScale, true, nil

	case OpOR:
		maxScale := 0.0
		matched := false
		for _, child := range node.Children {
			scale, match, err := cs.Evaluate(activation, child)
			if err != nil {
				return 0.0, false, err
			}
			if match && scale > 0 {
				matched = true
				if scale > maxScale {
					maxScale = scale
				}
			}
		}
		if !matched {
			return 0.0, false, nil
		}
		return maxScale, true, nil

	case OpNOT:
		if len(node.Children) == 0 {
			return 1.0, true, nil
		}
		scale, match, err := cs.Evaluate(activation, node.Children[0])
		if err != nil {
			return 0.0, false, err
		}
		if match && scale > 0 {
			return 0.0, false, nil
		}
		return 1.0, true, nil

	default:
		return 0.0, false, fmt.Errorf("unknown condition operator: %s", node.Operator)
	}
}

// ApplyConditional evaluates CAST gating and applies the scaled intervention to the residual stream.
func (cs *ConditionalSteerer) ApplyConditional(activation []float64, cmd Command, rule ConditionNode) ([]float64, float64, error) {
	scale, matched, err := cs.Evaluate(activation, rule)
	if err != nil {
		return activation, 0.0, err
	}
	if !matched || scale <= 0 {
		return activation, 0.0, nil
	}

	scaledCmd := cmd
	scaledCmd.Strength = cmd.Strength * scale
	steered := ApplyCommand(activation, scaledCmd)
	return steered, scale, nil
}
