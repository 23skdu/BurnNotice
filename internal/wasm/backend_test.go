package wasm

import (
	"testing"
)

func TestNewWASMBackend_FileNotFound(t *testing.T) {
	_, err := NewWASMBackend("/nonexistent/model.wasm")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestForward_NotLoaded(t *testing.T) {
	b := &WASMBackend{loaded: false}
	_, err := b.Forward([]float32{1, 2, 3})
	if err == nil {
		t.Error("expected error when WASM not loaded")
	}
}

func TestGetIntermediate_NotLoaded(t *testing.T) {
	b := &WASMBackend{loaded: false}
	_, err := b.GetIntermediate(0)
	if err == nil {
		t.Error("expected error when WASM not loaded")
	}
}

func TestClose_NilRuntime(t *testing.T) {
	b := &WASMBackend{runtime: nil}
	err := b.Close()
	if err != nil {
		t.Errorf("Close(nil runtime) error: %v", err)
	}
}
