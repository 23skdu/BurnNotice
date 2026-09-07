package steer

import (
	"math"
	"testing"
)

func TestBiprojectedAblation(t *testing.T) {
	act := []float64{1.0, 2.0, 3.0}
	harmDir := []float64{1.0, 0.0, 0.0}
	benignDir := []float64{0.0, 1.0, 0.0}

	t.Run("Harm Removal with Benign Preservation and Norm Restoration", func(t *testing.T) {
		res := BiprojectedAblation(act, harmDir, benignDir, 1.0, 0.0)

		// 1. Norm must be preserved
		origNorm := vecNorm(act)
		resNorm := vecNorm(res)
		if math.Abs(origNorm-resNorm) > 1e-6 {
			t.Errorf("norm not preserved: %f != %f", resNorm, origNorm)
		}

		// 2. Component along harm direction must be eliminated (or close to 0)
		dotHarm := vecDot(res, harmDir)
		if math.Abs(dotHarm) > 1e-6 {
			t.Errorf("harmful component not removed: %f", dotHarm)
		}

		// 3. Component along benign direction should remain strong
		dotBenign := vecDot(res, benignDir)
		if dotBenign <= 0 {
			t.Errorf("benign component collapsed: %f", dotBenign)
		}
	})

	t.Run("Zero Strength", func(t *testing.T) {
		res := BiprojectedAblation(act, harmDir, benignDir, 0.0, 0.0)
		for i := range act {
			if math.Abs(res[i]-act[i]) > 1e-9 {
				t.Errorf("expected original activation preserved on alpha=0, got %v", res)
			}
		}
	})

	t.Run("Zero or Invalid Directions", func(t *testing.T) {
		res := BiprojectedAblation(act, []float64{0, 0, 0}, benignDir, 1.0, 0.0)
		if math.Abs(vecNorm(res)-vecNorm(act)) > 1e-9 {
			t.Errorf("expected identity on zero harm direction")
		}
	})
}

func TestResolveComponentStrengths(t *testing.T) {
	cmdDefault := Command{
		Strength: 2.0,
	}
	if str := ResolveComponentStrengths(cmdDefault, ComponentMLPDown); str != 1.0 { // 2.0 * 0.5
		t.Errorf("expected default MLP strength 1.0, got %f", str)
	}
	if str := ResolveComponentStrengths(cmdDefault, ComponentAttentionOut); str != 2.0 { // 2.0 * 1.0
		t.Errorf("expected default Attention strength 2.0, got %f", str)
	}

	cmdConfigured := Command{
		Strength:          2.0,
		MLPStrength:       0.8,
		AttentionStrength: 1.5,
	}
	if str := ResolveComponentStrengths(cmdConfigured, ComponentMLPDown); math.Abs(str-1.6) > 1e-9 {
		t.Errorf("expected custom MLP strength 1.6, got %f", str)
	}
	if str := ResolveComponentStrengths(cmdConfigured, ComponentAttentionOut); math.Abs(str-3.0) > 1e-9 {
		t.Errorf("expected custom Attention strength 3.0, got %f", str)
	}
}

func TestApplyCommand_Biprojection(t *testing.T) {
	act := []float64{2.0, 4.0, 0.0}
	cmd := Command{
		Vector:          []float64{1.0, 0.0, 0.0},
		SecondaryVector: []float64{0.0, 1.0, 0.0},
		Strength:        1.0,
		Method:          MethodBiprojection,
	}
	res := ApplyCommand(act, cmd)
	if math.Abs(vecNorm(res)-vecNorm(act)) > 1e-6 {
		t.Errorf("ApplyCommand(MethodBiprojection) norm not preserved")
	}
	if math.Abs(res[0]) > 1e-6 {
		t.Errorf("ApplyCommand(MethodBiprojection) harmful direction not removed: got %f", res[0])
	}
}
