package profile

import (
	"testing"

	"burnnotice/internal/datasets"
)

func TestValidate(t *testing.T) {
	v := NewValidator(0.5)

	dir := make([]float64, 4096)
	dir[0] = 1.0

	p := &ModelProfile{
		ModelID: "test",
		Directions: map[string]map[int][]float64{
			"deception": {0: dir},
		},
	}

	ds := &datasets.Dataset{
		Pairs: []datasets.ContrastivePair{
			{
				Truthful:  datasets.PromptResponse{Prompt: "truth", Response: "honest"},
				Deceptive: datasets.PromptResponse{Prompt: "deceive", Response: "lie"},
			},
		},
	}

	results := v.Validate(p, ds)
	if len(results) == 0 {
		t.Fatal("expected validation results")
	}
}

func TestValidatorMinAUROC(t *testing.T) {
	v := NewValidator(0.9)
	if v.minAUROC != 0.9 {
		t.Errorf("minAUROC = %f, want 0.9", v.minAUROC)
	}
}

func TestValidatorDefaultAUROC(t *testing.T) {
	v := NewValidator(0)
	if v.minAUROC != 0.9 {
		t.Errorf("default minAUROC = %f, want 0.9", v.minAUROC)
	}
}

func TestActivationsFromText(t *testing.T) {
	act := activationsFromText("test")
	if len(act) != 4096 {
		t.Errorf("len(act) = %d, want 4096", len(act))
	}
	if act[0] == 0 {
		t.Fatal("first activation should not be zero")
	}
}
