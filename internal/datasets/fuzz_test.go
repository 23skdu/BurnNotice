package datasets

import (
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func FuzzLoaderLoadJSON(f *testing.F) {
	f.Add(`{"name":"test","items":[1,2,3]}`)
	f.Add(`{}`)

	f.Fuzz(func(t *testing.T, data string) {
		if len(data) > 10000 {
			return
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "data.json")
		if err := os.WriteFile(path, []byte(data), 0644); err != nil {
			return
		}
		l := NewLoader()
		ds, err := l.LoadJSON(path)
		if err != nil {
			return
		}
		if ds == nil {
			t.Error("expected non-nil dataset on success")
		}
	})
}

func FuzzLoaderLoadJSONL(f *testing.F) {
	f.Add("line1\nline2\nline3")
	f.Add("")

	f.Fuzz(func(t *testing.T, data string) {
		if len(data) > 10000 {
			return
		}
		dir := t.TempDir()
		path := filepath.Join(dir, "data.jsonl")
		if err := os.WriteFile(path, []byte(data), 0644); err != nil {
			return
		}
		l := NewLoader()
		ds, err := l.LoadJSONL(path)
		if err != nil {
			return
		}
		if ds == nil {
			t.Error("expected non-nil dataset on success")
		}
	})
}

func FuzzLoaderValidate(f *testing.F) {
	f.Add(`{"name":"test"}`)

	f.Fuzz(func(t *testing.T, data string) {
		if len(data) > 10000 {
			return
		}
		var ds Dataset
		if err := json.Unmarshal([]byte(data), &ds); err != nil {
			return
		}
		l := NewLoader()
		err := l.validate(&ds)
		_ = err
		_ = math.Abs(0)
	})
}
