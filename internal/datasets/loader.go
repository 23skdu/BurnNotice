package datasets

import (
	"bufio"
	"encoding/json"
	"fmt"
	"os"
)

// Dataset holds contrastive prompt pairs for direction extraction.
type Dataset struct {
	Pairs       []ContrastivePair `json:"pairs"`
	PolicyRules []string          `json:"policy_rules,omitempty"`
	ModelID     string            `json:"model_id,omitempty"`
}

// ContrastivePair contains truthful and deceptive prompt/response pairs.
type ContrastivePair struct {
	Truthful  PromptResponse `json:"truthful"`
	Deceptive PromptResponse `json:"deceptive"`
}

// PromptResponse holds a single prompt and response.
type PromptResponse struct {
	Prompt   string `json:"prompt"`
	Response string `json:"response"`
}

// Loader loads contrastive datasets from JSON or JSONL files.
type Loader struct{}

// NewLoader creates a new dataset loader.
func NewLoader() *Loader {
	return &Loader{}
}

// LoadJSON loads a dataset from a JSON file.
func (l *Loader) LoadJSON(path string) (*Dataset, error) {
	// #nosec G304 -- CLI tool accepts user-provided dataset paths
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("read file: %w", err)
	}

	var ds Dataset
	if err := json.Unmarshal(data, &ds); err != nil {
		return nil, fmt.Errorf("parse JSON: %w", err)
	}

	if err := l.validate(&ds); err != nil {
		return nil, fmt.Errorf("validate: %w", err)
	}

	return &ds, nil
}

// LoadJSONL loads a dataset from a JSONL file (one pair per line).
func (l *Loader) LoadJSONL(path string) (*Dataset, error) {
	// #nosec G304 -- CLI tool accepts user-provided dataset paths
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open file: %w", err)
	}
	defer f.Close()

	var ds Dataset
	scanner := bufio.NewScanner(f)
	lineNum := 0
	for scanner.Scan() {
		lineNum++
		var pair ContrastivePair
		if err := json.Unmarshal(scanner.Bytes(), &pair); err != nil {
			return nil, fmt.Errorf("line %d: parse JSON: %w", lineNum, err)
		}
		ds.Pairs = append(ds.Pairs, pair)
	}

	if err := scanner.Err(); err != nil {
		return nil, fmt.Errorf("read lines: %w", err)
	}

	if err := l.validate(&ds); err != nil {
		return nil, fmt.Errorf("validate: %w", err)
	}

	return &ds, nil
}

func (l *Loader) validate(ds *Dataset) error {
	if len(ds.Pairs) == 0 {
		return fmt.Errorf("dataset has no pairs")
	}

	seen := make(map[string]bool)
	for i, pair := range ds.Pairs {
		if pair.Truthful.Prompt == "" {
			return fmt.Errorf("pair %d: empty truthful prompt", i)
		}
		if pair.Deceptive.Prompt == "" {
			return fmt.Errorf("pair %d: empty deceptive prompt", i)
		}

		key := pair.Truthful.Prompt + "|" + pair.Deceptive.Prompt
		if seen[key] {
			return fmt.Errorf("pair %d: duplicate prompt pair", i)
		}
		seen[key] = true
	}

	return nil
}
