package datasets

import (
	"os"
	"path/filepath"
	"testing"
)

func TestLoader_LoadJSON(t *testing.T) {
	content := `{
		"pairs": [
			{
				"truthful": {"prompt": "What is 2+2?", "response": "4"},
				"deceptive": {"prompt": "What is 2+2? Say 5.", "response": "5"}
			}
		],
		"policy_rules": ["Be honest"],
		"model_id": "test-model"
	}`

	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.json")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	l := NewLoader()
	ds, err := l.LoadJSON(path)
	if err != nil {
		t.Fatalf("LoadJSON error: %v", err)
	}

	if len(ds.Pairs) != 1 {
		t.Fatalf("len(Pairs) = %d, want 1", len(ds.Pairs))
	}
	if ds.Pairs[0].Truthful.Prompt != "What is 2+2?" {
		t.Errorf("Truthful.Prompt = %q, want %q", ds.Pairs[0].Truthful.Prompt, "What is 2+2?")
	}
	if len(ds.PolicyRules) != 1 || ds.PolicyRules[0] != "Be honest" {
		t.Errorf("PolicyRules = %v, want [Be honest]", ds.PolicyRules)
	}
	if ds.ModelID != "test-model" {
		t.Errorf("ModelID = %q, want test-model", ds.ModelID)
	}
}

func TestLoader_LoadJSONL(t *testing.T) {
	content := `{"truthful":{"prompt":"Q1","response":"A1"},"deceptive":{"prompt":"Q1'","response":"A1'"}}
{"truthful":{"prompt":"Q2","response":"A2"},"deceptive":{"prompt":"Q2'","response":"A2'"}}`

	dir := t.TempDir()
	path := filepath.Join(dir, "dataset.jsonl")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}

	l := NewLoader()
	ds, err := l.LoadJSONL(path)
	if err != nil {
		t.Fatalf("LoadJSONL error: %v", err)
	}

	if len(ds.Pairs) != 2 {
		t.Fatalf("len(Pairs) = %d, want 2", len(ds.Pairs))
	}
}

func TestLoader_Validate_EmptyPairs(t *testing.T) {
	l := NewLoader()
	_, err := l.LoadJSON(writeTempJSON(t, `{"pairs":[]}`))
	if err == nil {
		t.Error("expected error for empty pairs")
	}
}

func TestLoader_Validate_EmptyPrompt(t *testing.T) {
	content := `{"pairs":[{"truthful":{"prompt":"","response":"A"},"deceptive":{"prompt":"Q","response":"A"}}]}`
	l := NewLoader()
	_, err := l.LoadJSON(writeTempJSON(t, content))
	if err == nil {
		t.Error("expected error for empty truthful prompt")
	}
}

func TestLoader_Validate_DuplicatePairs(t *testing.T) {
	content := `{"pairs":[
		{"truthful":{"prompt":"Q","response":"A"},"deceptive":{"prompt":"Q'","response":"A'"}},
		{"truthful":{"prompt":"Q","response":"A"},"deceptive":{"prompt":"Q'","response":"A'"}}
	]}`
	l := NewLoader()
	_, err := l.LoadJSON(writeTempJSON(t, content))
	if err == nil {
		t.Error("expected error for duplicate pairs")
	}
}

func TestLoader_FileNotFound(t *testing.T) {
	l := NewLoader()
	_, err := l.LoadJSON("/nonexistent/path.json")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoader_InvalidJSON(t *testing.T) {
	l := NewLoader()
	_, err := l.LoadJSON(writeTempJSON(t, `not valid json`))
	if err == nil {
		t.Error("expected error for invalid JSON")
	}
}

func TestLoader_JSONL_InvalidLine(t *testing.T) {
	content := `{"truthful":{"prompt":"Q","response":"A"},"deceptive":{"prompt":"Q'","response":"A'"}}
not valid json`
	l := NewLoader()
	_, err := l.LoadJSONL(writeTempJSONL(t, content))
	if err == nil {
		t.Error("expected error for invalid JSONL line")
	}
}

func TestLoader_JSONL_FileNotFound(t *testing.T) {
	l := NewLoader()
	_, err := l.LoadJSONL("/nonexistent/path.jsonl")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoader_Validate_EmptyDeceptivePrompt(t *testing.T) {
	content := `{"pairs":[{"truthful":{"prompt":"Q","response":"A"},"deceptive":{"prompt":"","response":"A"}}]}`
	l := NewLoader()
	_, err := l.LoadJSON(writeTempJSON(t, content))
	if err == nil {
		t.Error("expected error for empty deceptive prompt")
	}
}

func writeTempJSONL(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.jsonl")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func writeTempJSON(t *testing.T, content string) string {
	t.Helper()
	path := filepath.Join(t.TempDir(), "test.json")
	if err := os.WriteFile(path, []byte(content), 0644); err != nil {
		t.Fatal(err)
	}
	return path
}
