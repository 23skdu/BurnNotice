package profile

import (
	"os"
	"path/filepath"
	"testing"
)

func TestModelProfile_SaveAndLoad(t *testing.T) {
	original := &ModelProfile{
		ModelID: "test-model",
		Directions: map[string]map[int][]float64{
			"deception": {
				0: {0.1, 0.2, 0.3},
				1: {0.4, 0.5, 0.6},
			},
			"truth": {
				0: {-0.1, -0.2, -0.3},
			},
		},
		LayerScores: map[string][]float64{
			"deception": {0.8, 0.6, 0.9},
		},
		TruthNeurons:    []int{5, 12, 42},
		MonitoredLayers: []int{0, 1, 2},
	}

	dir := t.TempDir()
	if err := original.Save(dir); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := LoadProfile(dir)
	if err != nil {
		t.Fatalf("LoadProfile error: %v", err)
	}

	if loaded.ModelID != original.ModelID {
		t.Errorf("ModelID = %q, want %q", loaded.ModelID, original.ModelID)
	}

	if len(loaded.Directions["deception"]) != len(original.Directions["deception"]) {
		t.Errorf("directions count mismatch")
	}

	if len(loaded.TruthNeurons) != len(original.TruthNeurons) {
		t.Errorf("TruthNeurons len = %d, want %d", len(loaded.TruthNeurons), len(original.TruthNeurons))
	}
	for i, n := range original.TruthNeurons {
		if loaded.TruthNeurons[i] != n {
			t.Errorf("TruthNeurons[%d] = %d, want %d", i, loaded.TruthNeurons[i], n)
		}
	}
}

func TestLoadProfile_NotFound(t *testing.T) {
	_, err := LoadProfile("/nonexistent/path")
	if err == nil {
		t.Error("expected error for nonexistent profile")
	}
}

func TestModelProfile_Save_CreatesDirectory(t *testing.T) {
	dir := t.TempDir()
	nested := filepath.Join(dir, "a", "b", "c")

	p := &ModelProfile{ModelID: "test"}
	if err := p.Save(nested); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	if _, err := os.Stat(filepath.Join(nested, "profile.gob")); err != nil {
		t.Errorf("profile.gob not created: %v", err)
	}
}

func TestModelProfile_Save_WriteError(t *testing.T) {
	p := &ModelProfile{ModelID: "test"}
	err := p.Save("/proc/nonexistent/profile.gob")
	if err == nil {
		t.Error("expected error for unwritable path")
	}
}
