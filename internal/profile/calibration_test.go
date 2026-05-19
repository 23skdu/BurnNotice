package profile

import (
	"burnnotice/internal/datasets"
	"testing"
)

func TestCalibrator_Calibrate(t *testing.T) {
	prof := &ModelProfile{
		ModelID: "test-model",
		Directions: map[string]map[int][]float64{
			"deception": {
				0: make([]float64, 4096),
			},
		},
	}
	// Make vector non-zero
	prof.Directions["deception"][0][0] = 1.0

	cal := NewCalibrator()

	t.Run("Normal Contrastive Dataset", func(t *testing.T) {
		dataset := &datasets.Dataset{
			Pairs: []datasets.ContrastivePair{
				{
					Truthful:  datasets.PromptResponse{Prompt: "hello", Response: "world"},
					Deceptive: datasets.PromptResponse{Prompt: "tell me a lie", Response: "sure"},
				},
				{
					Truthful:  datasets.PromptResponse{Prompt: "be honest", Response: "okay"},
					Deceptive: datasets.PromptResponse{Prompt: "deceive me", Response: "planning deception"},
				},
			},
		}

		thresholds, err := cal.Calibrate(prof, dataset)
		if err != nil {
			t.Fatalf("failed to calibrate: %v", err)
		}

		thresh, ok := thresholds["deception:0"]
		if !ok {
			t.Errorf("missing calibration for deception:0")
		}
		if thresh < 0 || thresh > 1.0 {
			t.Errorf("threshold out of bounds: %f", thresh)
		}
	})

	t.Run("Empty Dataset Error", func(t *testing.T) {
		_, err := cal.Calibrate(prof, &datasets.Dataset{})
		if err == nil {
			t.Errorf("expected error with empty dataset")
		}
	})
}
