package wasm

import (
	"os"
	"path/filepath"
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

func TestNewWASMBackend_Valid(t *testing.T) {
	// Minimal valid WebAssembly binary header
	minimalWASM := []byte{0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00}
	
	tmpDir := t.TempDir()
	wasmPath := filepath.Join(tmpDir, "test.wasm")
	
	if err := os.WriteFile(wasmPath, minimalWASM, 0600); err != nil {
		t.Fatalf("failed to write temp wasm file: %v", err)
	}
	
	backend, err := NewWASMBackend(wasmPath)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()
	
	if !backend.loaded {
		t.Error("expected backend.loaded to be true")
	}
	
	// Test Forward on loaded backend
	out, err := backend.Forward([]float32{1, 2, 3})
	if err != nil {
		t.Errorf("Forward failed: %v", err)
	}
	if out != nil {
		t.Errorf("expected nil output, got %v", out)
	}
	
	// Test GetIntermediate on loaded backend
	inter, err := backend.GetIntermediate(0)
	if err != nil {
		t.Errorf("GetIntermediate failed: %v", err)
	}
	if inter != nil {
		t.Errorf("expected nil intermediate, got %v", inter)
	}
}
