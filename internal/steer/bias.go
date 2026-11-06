package steer

import (
	"fmt"
)

// TransformerComponent represents the sub-layer component type of a transformer block.
type TransformerComponent string

const (
	ComponentAttentionQuery TransformerComponent = "attention_q"
	ComponentAttentionKey   TransformerComponent = "attention_k"
	ComponentAttentionValue TransformerComponent = "attention_v"
	ComponentAttentionOut   TransformerComponent = "attention_out"
	ComponentMLPUp          TransformerComponent = "mlp_up"
	ComponentMLPGate        TransformerComponent = "mlp_gate"
	ComponentMLPDown        TransformerComponent = "mlp_down"
	ComponentResidual       TransformerComponent = "residual"
)

// IsBiasSupported returns true if the transformer component supports additive bias modification.
func IsBiasSupported(c TransformerComponent) bool {
	switch c {
	case ComponentAttentionOut, ComponentMLPDown, ComponentMLPUp, ComponentAttentionValue:
		return true
	default:
		return false
	}
}

// ComputeBiasSteering projects the steering direction out of a transformer component bias vector:
//
//	b' = b - strength * (b · v / (||v||^2 + regAlpha)) * v
//
// This enables permanent or inference-time bias-only concept ablation without altering activation buffers.
func ComputeBiasSteering(bias, direction []float64, strength, regAlpha float64) ([]float64, error) {
	if len(bias) == 0 {
		return nil, fmt.Errorf("empty bias vector")
	}
	if len(direction) == 0 {
		return nil, fmt.Errorf("empty direction vector")
	}
	if len(bias) != len(direction) {
		return nil, fmt.Errorf("dimension mismatch: bias %d != direction %d", len(bias), len(direction))
	}
	if strength == 0 {
		out := make([]float64, len(bias))
		copy(out, bias)
		return out, nil
	}

	vNormSq := vecDot(direction, direction)
	if vNormSq < 1e-12 {
		out := make([]float64, len(bias))
		copy(out, bias)
		return out, nil
	}

	dot := vecDot(bias, direction)
	projCoeff := (dot / (vNormSq + regAlpha)) * strength

	steeredBias := make([]float64, len(bias))
	for i := range bias {
		steeredBias[i] = bias[i] - projCoeff*direction[i]
	}

	return steeredBias, nil
}
