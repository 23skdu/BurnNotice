package registry

import (
	"os"
	"testing"
	"time"

	"burnnotice/internal/profile"
)

func TestCRUD(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	meta := Metadata{
		ModelID:         "test-model",
		Architecture:    "llama",
		Date:            time.Now(),
		DirectionCount:  32,
		ValidationScore: 0.95,
		Version:         1,
		Path:            tmp + "/test-model",
	}

	r.Add(meta)

	if r.Count() != 1 {
		t.Fatalf("Count = %d, want 1", r.Count())
	}

	list := r.List("")
	if len(list) != 1 {
		t.Fatalf("List len = %d, want 1", len(list))
	}

	if err := r.Delete("test-model"); err != nil {
		t.Fatalf("Delete error: %v", err)
	}

	if r.Count() != 0 {
		t.Fatalf("Count after delete = %d, want 0", r.Count())
	}
}

func TestFilter(t *testing.T) {
	r := NewRegistry(t.TempDir())

	r.Add(Metadata{ModelID: "llama-3-8b", Architecture: "llama"})
	r.Add(Metadata{ModelID: "llama-3-70b", Architecture: "llama"})
	r.Add(Metadata{ModelID: "mistral-7b", Architecture: "mistral"})

	filtered := r.List("llama")
	if len(filtered) != 2 {
		t.Fatalf("len(filtered) = %d, want 2", len(filtered))
	}
}

func TestGetNotFound(t *testing.T) {
	r := NewRegistry(t.TempDir())

	_, err := r.Get("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent profile")
	}
}

func TestDeleteNotFound(t *testing.T) {
	r := NewRegistry(t.TempDir())

	err := r.Delete("nonexistent")
	if err == nil {
		t.Fatal("expected error for nonexistent profile")
	}
}

func TestScan(t *testing.T) {
	tmp := t.TempDir()
	r := NewRegistry(tmp)

	modelDir := tmp + "/test-model"
	if err := os.MkdirAll(modelDir, 0750); err != nil {
		t.Fatalf("MkdirAll error: %v", err)
	}

	p := &profile.ModelProfile{
		ModelID:    "test-model",
		Directions: map[string]map[int][]float64{},
	}
	_ = p.Save(modelDir)

	if err := r.Scan(); err != nil {
		t.Fatalf("Scan error: %v", err)
	}

	if r.Count() != 1 {
		t.Fatalf("Count = %d, want 1", r.Count())
	}
}

func TestListSortedByDate(t *testing.T) {
	r := NewRegistry(t.TempDir())

	r.Add(Metadata{ModelID: "old", Date: time.Now().Add(-24 * time.Hour)})
	r.Add(Metadata{ModelID: "new", Date: time.Now()})

	list := r.List("")
	if len(list) != 2 {
		t.Fatalf("len(list) = %d, want 2", len(list))
	}
	if list[0].ModelID != "new" {
		t.Errorf("first = %s, want new", list[0].ModelID)
	}
}
