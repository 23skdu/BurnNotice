package engine

import (
	"os"
	"path/filepath"
	"testing"
)

func TestNewONNXEngine(t *testing.T) {
	cfg := ModelConfig{
		HiddenDim:         4096,
		NumLayers:         32,
		LayerNames:        []string{"layer.0", "layer.1"},
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

func TestLoad_FileNotFound(t *testing.T) {
	e := NewONNXEngine("/nonexistent/model.onnx", ModelConfig{})
	err := e.Load()
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestLoad_Success(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, []byte("fake onnx"), 0644); err != nil {
		t.Fatal(err)
	}

	e := NewONNXEngine(path, ModelConfig{})
	err := e.Load()
	if err != nil {
		t.Fatalf("Load error: %v", err)
	}
	if !e.loaded {
		t.Error("engine should be loaded after Load()")
	}
}

func TestForward_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.Forward([]float32{1, 2, 3})
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestGetIntermediate_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.GetIntermediate("layer.0")
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestForward_Success(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, []byte("fake onnx"), 0644); err != nil {
		t.Fatal(err)
	}

	e := NewONNXEngine(path, ModelConfig{})
	if err := e.Load(); err != nil {
		t.Fatal(err)
	}

	res, err := e.Forward([]float32{1.0, 2.0})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != nil {
		t.Errorf("expected nil result, got %v", res)
	}
}

func TestGetIntermediate_Success(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, []byte("fake onnx"), 0644); err != nil {
		t.Fatal(err)
	}

	e := NewONNXEngine(path, ModelConfig{})
	if err := e.Load(); err != nil {
		t.Fatal(err)
	}

	res, err := e.GetIntermediate("layer.0")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if res != nil {
		t.Errorf("expected nil result, got %v", res)
	}
}

