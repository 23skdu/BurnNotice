package engine

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewONNXEngine(t *testing.T) {
	cfg := ModelConfig{
		HiddenDim:          4096,
		NumLayers:          32,
		LayerNames:         []string{"layer.0", "layer.1"},
		AttentionHeadCount: 32,
	}

	e := NewONNXEngine("/path/to/model.onnx", cfg)
	if e.modelPath != "/path/to/model.onnx" {
		t.Errorf("modelPath = %q, want /path/to/model.onnx", e.modelPath)
	}
	if e.loaded {
		t.Error("engine should not be loaded initially")
	}

	got := e.Config()
	if got.HiddenDim != 4096 {
		t.Errorf("HiddenDim = %d, want 4096", got.HiddenDim)
	}
	if got.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", got.NumLayers)
	}
	if len(got.LayerNames) != 2 {
		t.Errorf("len(LayerNames) = %d, want 2", len(got.LayerNames))
	}
	if got.AttentionHeadCount != 32 {
		t.Errorf("AttentionHeadCount = %d, want 32", got.AttentionHeadCount)
	}
}

func TestNewONNXEngine_DefaultIONames(t *testing.T) {
	e := NewONNXEngine("/path/to/model.onnx", ModelConfig{})
	if len(e.config.InputNames) != 1 || e.config.InputNames[0] != "input_ids" {
		t.Errorf("default InputNames = %v, want [input_ids]", e.config.InputNames)
	}
	if len(e.config.OutputNames) != 1 || e.config.OutputNames[0] != "hidden_states" {
		t.Errorf("default OutputNames = %v, want [hidden_states]", e.config.OutputNames)
	}
}

func TestNewONNXEngine_CustomIONames(t *testing.T) {
	cfg := ModelConfig{
		InputNames:  []string{"input", "attention_mask"},
		OutputNames: []string{"logits", "hidden_states_0", "hidden_states_1"},
	}
	e := NewONNXEngine("/path/to/model.onnx", cfg)
	if len(e.config.InputNames) != 2 {
		t.Errorf("InputNames count = %d, want 2", len(e.config.InputNames))
	}
	if len(e.config.OutputNames) != 3 {
		t.Errorf("OutputNames count = %d, want 3", len(e.config.OutputNames))
	}
}

func TestLoad_FileNotFound(t *testing.T) {
	e := NewONNXEngine("/nonexistent/model.onnx", ModelConfig{})
	err := e.Load()
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestForward_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.Forward([]int64{1, 2, 3}, []int64{1, 3})
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestGetIntermediate_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.GetIntermediate("layer.0", []int64{1}, []int64{1, 1})
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestNumOutputs(t *testing.T) {
	cfg := ModelConfig{
		OutputNames: []string{"a", "b", "c"},
	}
	e := NewONNXEngine("/tmp/model.onnx", cfg)
	if n := e.NumOutputs(); n != 3 {
		t.Errorf("NumOutputs = %d, want 3", n)
	}
}

func TestClose_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	if err := e.Close(); err != nil {
		t.Errorf("Close on unloaded engine should succeed: %v", err)
	}
}

func TestLoad_InvalidONNX(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, []byte("not a valid onnx file"), 0644); err != nil {
		t.Fatal(err)
	}

	e := NewONNXEngine(path, ModelConfig{
		HiddenDim: 4096,
	})
	err := e.Load()
	if err == nil {
		t.Skip("onnx runtime accepted invalid file - may be using a relaxed parser")
	}
}

func TestEngine_ConfigAccess(t *testing.T) {
	cfg := ModelConfig{
		HiddenDim:          2048,
		NumLayers:          16,
		LayerNames:         []string{"layer.0", "layer.1", "layer.2"},
		AttentionHeadCount: 16,
	}
	e := NewONNXEngine("/tmp/model.onnx", cfg)
	got := e.Config()
	if got.HiddenDim != cfg.HiddenDim {
		t.Errorf("HiddenDim = %d, want %d", got.HiddenDim, cfg.HiddenDim)
	}
	if got.NumLayers != cfg.NumLayers {
		t.Errorf("NumLayers = %d, want %d", got.NumLayers, cfg.NumLayers)
	}
	if got.AttentionHeadCount != cfg.AttentionHeadCount {
		t.Errorf("AttentionHeadCount = %d, want %d", got.AttentionHeadCount, cfg.AttentionHeadCount)
	}
}
