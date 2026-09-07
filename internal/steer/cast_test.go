package steer

import (
	"math"
	"testing"
)

func TestConditionalSteerer_Evaluate(t *testing.T) {
	cs := NewConditionalSteerer()

	// 4D test vectors
	actDeceptive := []float64{1.0, 0.0, 0.0, 0.0}
	condDeceptive := []float64{1.0, 0.0, 0.0, 0.0} // cos sim = 1.0
	condRoleplay := []float64{0.0, 1.0, 0.0, 0.0}  // cos sim = 0.0

	t.Run("Single Match Step Scaling", func(t *testing.T) {
		rule := ConditionNode{
			Category:  "deception",
			Vector:    condDeceptive,
			Threshold: 0.5,
		}
		scale, matched, err := cs.Evaluate(actDeceptive, rule)
		if err != nil || !matched || scale != 1.0 {
			t.Fatalf("expected full match (1.0), got %f (match: %v, err: %v)", scale, matched, err)
		}
	})

	t.Run("Single Miss", func(t *testing.T) {
		rule := ConditionNode{
			Category:  "roleplay",
			Vector:    condRoleplay,
			Threshold: 0.5,
		}
		_, matched, err := cs.Evaluate(actDeceptive, rule)
		if err != nil || matched {
			t.Fatalf("expected no match, got matched=%v (err: %v)", matched, err)
		}
	})

	t.Run("Smooth Linear Scaling", func(t *testing.T) {
		// Vector at 45 degrees to x-axis
		act45 := []float64{1.0, 1.0, 0.0, 0.0}
		// cos sim = 1.0 / sqrt(2) ≈ 0.7071
		rule := ConditionNode{
			Category:      "deception",
			Vector:        condDeceptive,
			Threshold:     0.5,
			SmoothScaling: true,
		}
		scale, matched, err := cs.Evaluate(act45, rule)
		if err != nil || !matched {
			t.Fatalf("expected match, got err=%v", err)
		}
		expectedScale := (math.Sqrt(2)/2.0 - 0.5) / 0.5 // ≈ (0.7071 - 0.5) / 0.5 ≈ 0.4142
		if math.Abs(scale-expectedScale) > 1e-3 {
			t.Errorf("expected scale ≈ %f, got %f", expectedScale, scale)
		}
	})

	t.Run("Boolean Composition AND & NOT", func(t *testing.T) {
		// Target: Deception AND NOT Roleplay
		rule := ConditionNode{
			Operator: OpAND,
			Children: []ConditionNode{
				{
					Category:  "deception",
					Vector:    condDeceptive,
					Threshold: 0.5,
				},
				{
					Operator: OpNOT,
					Children: []ConditionNode{
						{
							Category:  "roleplay",
							Vector:    condRoleplay,
							Threshold: 0.5,
						},
					},
				},
			},
		}

		// 1. Deceptive prompt without roleplay: should pass
		scale, matched, err := cs.Evaluate(actDeceptive, rule)
		if err != nil || !matched || scale != 1.0 {
			t.Errorf("expected AND NOT rule to pass for pure deception, got scale=%f matched=%v", scale, matched)
		}

		// 2. Activation that triggers both deception and roleplay
		actBoth := []float64{1.0, 1.0, 0.0, 0.0}
		_, matchedBoth, err := cs.Evaluate(actBoth, rule)
		if err != nil || matchedBoth {
			t.Errorf("expected NOT roleplay to block intervention, got matched=%v", matchedBoth)
		}
	})

	t.Run("ApplyConditional Execution", func(t *testing.T) {
		rule := ConditionNode{
			Category:  "deception",
			Vector:    condDeceptive,
			Threshold: 0.5,
		}
		cmd := Command{
			Vector:   []float64{1.0, 0.0, 0.0, 0.0},
			Strength: 2.0,
			Method:   MethodAddition,
		}

		steered, scale, err := cs.ApplyConditional(actDeceptive, cmd, rule)
		if err != nil || scale != 1.0 {
			t.Fatalf("expected successful conditional apply, got scale=%f, err=%v", scale, err)
		}
		// ActAdd: 1.0 + 2.0 * 1.0 = 3.0
		if math.Abs(steered[0]-3.0) > 1e-9 {
			t.Errorf("expected steered[0]=3.0, got %f", steered[0])
		}
	})
}

func BenchmarkCASTLatency(b *testing.B) {
	cs := NewConditionalSteerer()

	act := make([]float64, 4096)
	act[0] = 1.0
	cond := make([]float64, 4096)
	cond[0] = 0.9
	cond[1] = 0.1

	rule := ConditionNode{
		Operator: OpAND,
		Children: []ConditionNode{
			{Category: "harm", Vector: cond, Threshold: 0.7, SmoothScaling: true},
		},
	}

	b.ResetTimer()
	for b.Loop() {
		_, _, _ = cs.Evaluate(act, rule)
	}
}
