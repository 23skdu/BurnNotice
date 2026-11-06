package profile

import (
	"os"
	"path/filepath"
	"testing"

	"burnnotice/internal/datasets"
)

func TestCalibrator_Calibrate(t *testing.T) {
	prof := &ModelProfile{
		ModelID: "test-model",
		Directions: map[string]map[int][]float64{
			"deception": {
				0: make([]float64, 4096),
			},
			"truth": {
				0: make([]float64, 4096),
			},
		},
	}
	// Make vectors point in opposite directions to cover both isGreaterRule true and false
	prof.Directions["deception"][0][0] = 1.0
	prof.Directions["truth"][0][0] = -1.0

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

		result, err := cal.Calibrate(prof, dataset)
		if err != nil {
			t.Fatalf("failed to calibrate: %v", err)
		}

		thresh, ok := result.Thresholds["deception:0"]
		if !ok {
			t.Errorf("missing calibration for deception:0")
		}
		if thresh.Threshold < 0 || thresh.Threshold > 1.0 {
			t.Errorf("threshold out of bounds: %f", thresh.Threshold)
		}

		dir := t.TempDir()
		if err := result.Save(dir); err != nil {
			t.Fatalf("Save calibration failed: %v", err)
		}

		loaded, err := LoadCalibration(dir)
		if err != nil {
			t.Fatalf("LoadCalibration failed: %v", err)
		}
		if loaded.ModelID != result.ModelID {
			t.Errorf("loaded ModelID = %q, want %q", loaded.ModelID, result.ModelID)
		}
	})

	t.Run("Empty Dataset Error", func(t *testing.T) {
		_, err := cal.Calibrate(prof, &datasets.Dataset{})
		if err == nil {
			t.Errorf("expected error with empty dataset")
		}
	})

	t.Run("Save and Load Errors", func(t *testing.T) {
		cr := &CalibrationResult{ModelID: "m"}
		if err := cr.Save("/nonexistent_dir_12345"); err == nil {
			t.Error("expected error saving calibration to nonexistent dir")
		}
		if _, err := LoadCalibration("/nonexistent_dir_12345"); err == nil {
			t.Error("expected error loading calibration from nonexistent dir")
		}
		badDir := t.TempDir()
		if err := os.WriteFile(filepath.Join(badDir, "calibration.gob"), []byte("corrupt"), 0600); err != nil {
			t.Fatal(err)
		}
		if _, err := LoadCalibration(badDir); err == nil {
			t.Error("expected error decoding corrupt calibration file")
		}
	})
}

func TestDriftDetector(t *testing.T) {
	// Default parameters when <= 0
	ddDefault := NewDriftDetector(0, 0)
	if ddDefault.windowSize != 100 || ddDefault.sigmaThreshold != 1.5 {
		t.Errorf("unexpected defaults: windowSize=%d, sigma=%f", ddDefault.windowSize, ddDefault.sigmaThreshold)
	}

	dd := NewDriftDetector(4, 1.2)
	if !dd.LastRecalibrated().IsZero() {
		t.Error("expected zero LastRecalibrated initially")
	}

	// Single record (< 2) returns false
	if dd.Record(1.0) {
		t.Error("expected false on first sample")
	}
	// Identical record (stdDev == 0) returns false
	if dd.Record(1.0) {
		t.Error("expected false when stdDev is 0")
	}
	dd.Record(1.0)
	dd.Record(1.0)
	// Window overflow (> windowSize) with large spike triggers recalibration
	if !dd.Record(10.0) {
		t.Error("expected true on large drift spike")
	}

	dd.MarkRecalibrated()
	if dd.LastRecalibrated().IsZero() {
		t.Error("expected non-zero LastRecalibrated after MarkRecalibrated")
	}
}

func TestAdjustThresholdAndINLP(t *testing.T) {
	if got := AdjustThresholdForCone(0.5, 1); got != 0.5 {
		t.Errorf("AdjustThresholdForCone(0.5, 1) = %f, want 0.5", got)
	}
	if got := AdjustThresholdForCone(0.4, 4); got != 0.8 {
		t.Errorf("AdjustThresholdForCone(0.4, 4) = %f, want 0.8", got)
	}
	if got := AdjustThresholdForCone(0.5, 16); got != 0.95 {
		t.Errorf("AdjustThresholdForCone(0.5, 16) = %f, want clamped 0.95", got)
	}

	if got := RecommendedINLPIterations(1); got != 5 {
		t.Errorf("RecommendedINLPIterations(1) = %d, want 5", got)
	}
	if got := RecommendedINLPIterations(2); got != 5 {
		t.Errorf("RecommendedINLPIterations(2) = %d, want 5", got)
	}
	if got := RecommendedINLPIterations(8); got != 16 {
		t.Errorf("RecommendedINLPIterations(8) = %d, want 16", got)
	}
	if got := RecommendedINLPIterations(30); got != 40 {
		t.Errorf("RecommendedINLPIterations(30) = %d, want clamped 40", got)
	}
}
