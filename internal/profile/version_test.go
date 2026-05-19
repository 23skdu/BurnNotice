package profile

import (
	"testing"
	"time"
)

func TestNextVersion(t *testing.T) {
	tmp := t.TempDir()
	vm := NewVersionManager(tmp)

	v, err := vm.NextVersion()
	if err != nil {
		t.Fatalf("NextVersion error: %v", err)
	}
	if v != 1 {
		t.Errorf("version = %d, want 1", v)
	}
}

func TestSaveVersion(t *testing.T) {
	tmp := t.TempDir()
	vm := NewVersionManager(tmp)

	p := &ModelProfile{
		ModelID:    "test-model",
		Directions: map[string]map[int][]float64{"deception": {0: {0.1, 0.2}}},
	}

	v, err := vm.SaveVersion(p, "")
	if err != nil {
		t.Fatalf("SaveVersion error: %v", err)
	}
	if v.Version != 1 {
		t.Errorf("version = %d, want 1", v.Version)
	}
	if v.ModelID != "test-model" {
		t.Errorf("modelID = %s, want test-model", v.ModelID)
	}
}

func TestLoadVersion(t *testing.T) {
	tmp := t.TempDir()
	vm := NewVersionManager(tmp)

	p := &ModelProfile{
		ModelID:    "test-model",
		Directions: map[string]map[int][]float64{"deception": {0: {0.1, 0.2}}},
	}

	_, err := vm.SaveVersion(p, "")
	if err != nil {
		t.Fatalf("SaveVersion error: %v", err)
	}

	loaded, err := vm.LoadVersion(1)
	if err != nil {
		t.Fatalf("LoadVersion error: %v", err)
	}
	if loaded.ModelID != "test-model" {
		t.Errorf("modelID = %s, want test-model", loaded.ModelID)
	}
}

func TestListVersions(t *testing.T) {
	tmp := t.TempDir()
	vm := NewVersionManager(tmp)

	p := &ModelProfile{ModelID: "test-model", Directions: map[string]map[int][]float64{}}

	_, _ = vm.SaveVersion(p, "")
	time.Sleep(10 * time.Millisecond)
	_, _ = vm.SaveVersion(p, "v1")

	versions, err := vm.ListVersions()
	if err != nil {
		t.Fatalf("ListVersions error: %v", err)
	}
	if len(versions) != 2 {
		t.Fatalf("len(versions) = %d, want 2", len(versions))
	}
}

func TestDiff(t *testing.T) {
	old := &ModelProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1.0, 0.0, 0.0}},
		},
	}
	new := &ModelProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1.0, 0.0, 0.0}},
		},
	}

	distances := Diff(old, new)
	if len(distances) != 1 {
		t.Fatalf("len(distances) = %d, want 1", len(distances))
	}
	if distances["deception:0"] != 0 {
		t.Errorf("distance = %f, want 0", distances["deception:0"])
	}
}

func TestDiffDifferent(t *testing.T) {
	old := &ModelProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {1.0, 0.0, 0.0}},
		},
	}
	new := &ModelProfile{
		Directions: map[string]map[int][]float64{
			"deception": {0: {0.0, 1.0, 0.0}},
		},
	}

	distances := Diff(old, new)
	if distances["deception:0"] == 0 {
		t.Fatal("distance should not be 0 for different vectors")
	}
}

func TestDetectVersionDrift(t *testing.T) {
	distances := map[string]float64{
		"deception:0": 0.5,
		"truth:0":     0.1,
	}

	drifts := DetectVersionDrift(distances, 0.3)
	if len(drifts) != 1 {
		t.Fatalf("len(drifts) = %d, want 1", len(drifts))
	}
	if drifts[0] != "deception:0" {
		t.Errorf("drift = %s, want deception:0", drifts[0])
	}
}

func TestVersionCosineDistanceEdgeCases(t *testing.T) {
	if d := versionCosineDistance(nil, nil); d != 0 {
		t.Errorf("nil vectors distance = %f, want 0", d)
	}

	if d := versionCosineDistance([]float64{}, []float64{}); d != 0 {
		t.Errorf("empty vectors distance = %f, want 0", d)
	}

	if d := versionCosineDistance([]float64{0, 0}, []float64{0, 0}); d != 0 {
		t.Errorf("zero vectors distance = %f, want 0", d)
	}
}
