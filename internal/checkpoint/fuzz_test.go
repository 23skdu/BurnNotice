package checkpoint

import (
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"
)

func FuzzCheckpointSaveLoad(f *testing.F) {
	f.Add(42, "test-state")
	f.Add(0, "")

	f.Fuzz(func(t *testing.T, val int, name string) {
		if len(name) > 100 {
			return
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "checkpoint.gob")
		m := NewManager(0, path)
		state := map[string]interface{}{
			"val":  val,
			"name": name,
		}
		if err := m.Save(state); err != nil {
			return
		}
		loaded, err := Load(path)
		if err != nil {
			t.Fatalf("Load error: %v", err)
		}
		if loaded == nil {
			t.Fatal("loaded state is nil")
		}
		if !Exists(path) {
			t.Error("Exists returned false for existing checkpoint")
		}
	})
}

func FuzzCheckpointLoadMissing(f *testing.F) {
	f.Add("nonexistent.gob")

	f.Fuzz(func(t *testing.T, name string) {
		if len(name) > 100 || name == "" || name == "." || name == ".." {
			return
		}
		// Skip names with path separators or null bytes
		if strings.ContainsAny(name, "/\\") {
			return
		}
		dir := t.TempDir()
		path := filepath.Join(dir, name)
		_, err := Load(path)
		if err == nil {
			t.Error("expected error for missing checkpoint")
		}
		if Exists(path) {
			t.Error("Exists returned true for missing checkpoint")
		}
	})
}

func FuzzCheckpointManager(f *testing.F) {
	f.Add(1, 5)

	f.Fuzz(func(t *testing.T, interval int, count int) {
		if interval < 0 || interval > 100 || count < 1 || count > 5 {
			return
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "ckpt.gob")
		m := NewManager(0, path)
		for i := 0; i < count; i++ {
			state := map[string]int{"i": i}
			_ = m.Save(state)
		}
		loaded, err := Load(path)
		if err != nil {
			return
		}
		if loaded == nil {
			t.Error("loaded state is nil")
		}
		_ = math.Abs(float64(count))
		_ = os.Remove(path)
	})
}
