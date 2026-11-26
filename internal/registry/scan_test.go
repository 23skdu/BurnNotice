package registry

import (
	"os"
	"path/filepath"
	"testing"

	"burnnotice/internal/profile"
)

func TestScan_MissingBaseDir(t *testing.T) {
	r := NewRegistry(filepath.Join(t.TempDir(), "does-not-exist"))
	if err := r.Scan(); err != nil {
		t.Fatalf("Scan on missing dir should be a no-op, got: %v", err)
	}
	if r.Count() != 0 {
		t.Errorf("Count = %d, want 0", r.Count())
	}
}

func TestScan_UnreadableBaseDir(t *testing.T) {
	if os.Geteuid() == 0 {
		t.Skip("permissions enforced differently for root")
	}
	dir := t.TempDir()
	blocked := filepath.Join(dir, "blocked")
	if err := os.MkdirAll(blocked, 0750); err != nil {
		t.Fatal(err)
	}
	if err := os.Chmod(blocked, 0000); err != nil {
		t.Fatal(err)
	}
	t.Cleanup(func() { _ = os.Chmod(blocked, 0750) })

	r := NewRegistry(blocked)
	if err := r.Scan(); err == nil {
		t.Fatal("expected error for unreadable base dir")
	}
}

func TestScan_SkipsInvalidEntries(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	// Plain file: not a directory, skipped.
	if err := os.WriteFile(filepath.Join(tmp, "stray.txt"), []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	// Directory without profile.gob: skipped.
	if err := os.MkdirAll(filepath.Join(tmp, "no-profile"), 0750); err != nil {
		t.Fatal(err)
	}
	// Directory with corrupt profile.gob: LoadProfile fails, skipped.
	corruptDir := filepath.Join(tmp, "corrupt")
	if err := os.MkdirAll(corruptDir, 0750); err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(filepath.Join(corruptDir, "profile.gob"), []byte{0xff, 0xfe}, 0600); err != nil {
		t.Fatal(err)
	}
	// Valid profile with directions.
	validDir := filepath.Join(tmp, "valid-model")
	p := &profile.ModelProfile{
		ModelID: "valid-model",
		Directions: map[string]map[int][]float64{
			"refusal":   {0: {1, 0}, 1: {0, 1}},
			"deception": {2: {1, 1}},
		},
	}
	if err := p.Save(validDir); err != nil {
		t.Fatal(err)
	}

	if err := r.Scan(); err != nil {
		t.Fatalf("Scan error: %v", err)
	}
	if r.Count() != 1 {
		t.Fatalf("Count = %d, want 1 (invalid entries skipped)", r.Count())
	}
	list := r.List("valid")
	if len(list) != 1 || list[0].DirectionCount != 3 {
		t.Fatalf("List = %+v, want one entry with DirectionCount 3", list)
	}
}

func TestGet_LoadProfileError(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	modelDir := filepath.Join(tmp, "gone")
	p := &profile.ModelProfile{ModelID: "gone", Directions: map[string]map[int][]float64{}}
	if err := p.Save(modelDir); err != nil {
		t.Fatal(err)
	}
	if err := r.Scan(); err != nil {
		t.Fatal(err)
	}
	if err := os.RemoveAll(modelDir); err != nil {
		t.Fatal(err)
	}

	if _, err := r.Get("gone"); err == nil {
		t.Fatal("expected error when profile dir was removed after scan")
	}
}

func TestGet_Success(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	modelDir := filepath.Join(tmp, "m1")
	p := &profile.ModelProfile{ModelID: "m1", Architecture: "qwen", Directions: map[string]map[int][]float64{}}
	if err := p.Save(modelDir); err != nil {
		t.Fatal(err)
	}
	if err := r.Scan(); err != nil {
		t.Fatal(err)
	}

	got, err := r.Get("m1")
	if err != nil {
		t.Fatalf("Get error: %v", err)
	}
	if got.ModelID != "m1" {
		t.Errorf("ModelID = %q, want m1", got.ModelID)
	}
}

func TestDelete_RemovesFiles(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	modelDir := filepath.Join(tmp, "del-me")
	p := &profile.ModelProfile{ModelID: "del-me", Directions: map[string]map[int][]float64{}}
	if err := p.Save(modelDir); err != nil {
		t.Fatal(err)
	}
	if err := r.Scan(); err != nil {
		t.Fatal(err)
	}
	if err := r.Delete("del-me"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}
	if _, err := os.Stat(modelDir); !os.IsNotExist(err) {
		t.Errorf("profile dir still present after Delete")
	}
	if r.Count() != 0 {
		t.Errorf("Count = %d, want 0", r.Count())
	}
}
