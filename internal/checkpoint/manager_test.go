package checkpoint

import (
	"os"
	"testing"
	"time"
)

func TestSaveLoadRoundTrip(t *testing.T) {
	tmp := t.TempDir() + "/checkpoint.gob"
	defer os.Remove(tmp)

	original := map[string]interface{}{"counter": float64(42), "name": "test"}

	m := NewManager(1*time.Hour, tmp)
	if err := m.Save(original); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	loaded, err := Load(tmp)
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}

	state, ok := loaded.(map[string]interface{})
	if !ok {
		t.Fatalf("loaded type = %T, want map[string]interface{}", loaded)
	}
	if state["counter"] != float64(42) {
		t.Errorf("counter = %v, want 42", state["counter"])
	}
	if state["name"] != "test" {
		t.Errorf("name = %v, want test", state["name"])
	}
}

func TestAtomicRename(t *testing.T) {
	tmp := t.TempDir() + "/checkpoint.gob"
	defer os.Remove(tmp)
	defer os.Remove(tmp + ".tmp")

	original := map[string]interface{}{"counter": 1}

	m := NewManager(1*time.Hour, tmp)
	if err := m.Save(original); err != nil {
		t.Fatalf("Save error: %v", err)
	}

	if _, err := os.Stat(tmp + ".tmp"); err == nil {
		t.Fatal("temp file should not exist after save")
	}

	if _, err := os.Stat(tmp); err != nil {
		t.Fatal("checkpoint file should exist after save")
	}
}

func TestExists(t *testing.T) {
	tmp := t.TempDir() + "/checkpoint.gob"
	defer os.Remove(tmp)

	if Exists(tmp) {
		t.Fatal("Exists should return false for non-existent file")
	}

	original := map[string]interface{}{"counter": 1}
	m := NewManager(1*time.Hour, tmp)
	_ = m.Save(original)

	if !Exists(tmp) {
		t.Fatal("Exists should return true for existing file")
	}
}

func TestStartStop(t *testing.T) {
	tmp := t.TempDir() + "/checkpoint.gob"
	defer os.Remove(tmp)

	m := NewManager(100*time.Millisecond, tmp)
	state := map[string]interface{}{"counter": 0}

	m.Start(&state)

	time.Sleep(250 * time.Millisecond)

	m.Stop()

	if !Exists(tmp) {
		t.Fatal("checkpoint should exist after Start")
	}
}

func TestLoadNotFound(t *testing.T) {
	_, err := Load("/nonexistent/path/checkpoint.gob")
	if err == nil {
		t.Fatal("expected error for non-existent file")
	}
}
