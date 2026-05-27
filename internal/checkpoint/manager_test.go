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
		t.Fatal("expected error for nonexistent file")
	}
}

func TestLoad_InvalidJSON(t *testing.T) {
	tmp := t.TempDir() + "/invalid.json"
	if err := os.WriteFile(tmp, []byte("invalid-json"), 0644); err != nil {
		t.Fatal(err)
	}

	_, err := Load(tmp)
	if err == nil {
		t.Error("expected error loading invalid JSON, got nil")
	}
}

func TestSave_MarshalError(t *testing.T) {
	tmp := t.TempDir() + "/marshal_error.json"
	m := NewManager(1*time.Hour, tmp)

	// Functions cannot be marshalled to JSON
	unmarshallable := map[string]interface{}{
		"fn": func() {},
	}

	err := m.Save(unmarshallable)
	if err == nil {
		t.Error("expected error saving unmarshallable state, got nil")
	}
}

func TestSave_DirCreationError(t *testing.T) {
	// A file already exists at the parent directory path, so MkdirAll should fail
	tmpDir := t.TempDir()
	filePath := tmpDir + "/parent"
	if err := os.WriteFile(filePath, []byte("some-data"), 0644); err != nil {
		t.Fatal(err)
	}

	// This path attempts to create a subdirectory under /parent, which is a file
	m := NewManager(1*time.Hour, filePath+"/checkpoint.json")
	err := m.Save(map[string]interface{}{"val": 1})
	if err == nil {
		t.Error("expected directory creation error, got nil")
	}
}

func TestSave_WriteFileError(t *testing.T) {
	tmpDir := t.TempDir()
	m := NewManager(1*time.Hour, tmpDir+"/checkpoint.json")

	// Pre-create the temp path as a directory so os.WriteFile fails
	if err := os.MkdirAll(tmpDir+"/checkpoint.json.tmp", 0750); err != nil {
		t.Fatal(err)
	}

	err := m.Save(map[string]interface{}{"val": 1})
	if err == nil {
		t.Error("expected write file error, got nil")
	}
}

func TestStart_SaveErrorPath(t *testing.T) {
	// If the initial save inside Start fails, it returns early without ticker
	tmpDir := t.TempDir()
	filePath := tmpDir + "/parent"
	if err := os.WriteFile(filePath, []byte("some-data"), 0644); err != nil {
		t.Fatal(err)
	}

	m := NewManager(1*time.Hour, filePath+"/checkpoint.json")
	m.Start(map[string]interface{}{"val": 1})
	if m.ticker != nil {
		t.Error("expected ticker to not be initialized on save error")
	}
}

func TestSave_RenameError(t *testing.T) {
	tmpDir := t.TempDir()
	filePath := tmpDir + "/checkpoint.json"

	// Pre-create filePath as a directory, so that os.Rename(tmp, filePath) will fail
	// because you cannot rename a file to replace a directory.
	if err := os.MkdirAll(filePath, 0750); err != nil {
		t.Fatal(err)
	}

	m := NewManager(1*time.Hour, filePath)
	err := m.Save(map[string]interface{}{"val": 1})
	if err == nil {
		t.Error("expected rename error, got nil")
	}
}

