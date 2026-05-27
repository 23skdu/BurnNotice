package wasm

import (
	"bytes"
	"fmt"
	"os"
	"path/filepath"
	"testing"
)

// leb128 encodes v as unsigned LEB128.
func leb128(v uint32) []byte {
	var b []byte
	for {
		byte_ := byte(v & 0x7f)
		v >>= 7
		if v != 0 {
			byte_ |= 0x80
		}
		b = append(b, byte_)
		if v == 0 {
			break
		}
	}
	return b
}

// buildTestModule returns a minimal WASM binary that exports:
//   - "memory" (linear memory, 1 page = 64KB)
//   - "forward" (i32) -> (i32): returns inputLen unchanged; data in/out at mem offset 0
func buildTestModule() []byte {
	var buf bytes.Buffer

	// Preamble
	buf.Write([]byte{0x00, 0x61, 0x73, 0x6d, 0x01, 0x00, 0x00, 0x00})

	// Type section (id=1): one type (i32)->(i32)
	typeContent := []byte{0x01, 0x60, 0x01, 0x7f, 0x01, 0x7f}
	buf.WriteByte(1)              // section id
	buf.Write(leb128(uint32(len(typeContent)))) // section size
	buf.Write(typeContent)

	// Function section (id=3): one function, type index 0
	funcContent := []byte{0x01, 0x00}
	buf.WriteByte(3)
	buf.Write(leb128(uint32(len(funcContent))))
	buf.Write(funcContent)

	// Memory section (id=5): 1 memory, min 1 page (64KB)
	memContent := []byte{0x01, 0x00, 0x01}
	buf.WriteByte(5)
	buf.Write(leb128(uint32(len(memContent))))
	buf.Write(memContent)

	// Export section (id=7): "memory" (kind=2 idx=0), "forward" (kind=0 idx=0)
	exportContent := []byte{
		0x02,             // 2 exports
		0x06, 'm', 'e', 'm', 'o', 'r', 'y', 0x02, 0x00, // "memory" mem=0
		0x07, 'f', 'o', 'r', 'w', 'a', 'r', 'd', 0x00, 0x00, // "forward" func=0
	}
	buf.WriteByte(7)
	buf.Write(leb128(uint32(len(exportContent))))
	buf.Write(exportContent)

	// Code section (id=10): 1 body
	// body: 0 locals, local.get 0, end
	body0 := []byte{0x00, 0x20, 0x00, 0x0b}
	codeContent := append([]byte{0x01}, byte(len(body0)))
	codeContent = append(codeContent, body0...)
	buf.WriteByte(10)
	buf.Write(leb128(uint32(len(codeContent))))
	buf.Write(codeContent)

	return buf.Bytes()
}

func TestTestModuleIsValid(t *testing.T) {
	mod := buildTestModule()
	// Check it starts with valid WASM magic
	if len(mod) < 8 || string(mod[:4]) != "\x00asm" {
		t.Fatal("invalid WASM preamble")
	}
	// Try compiling with wazero
	path := writeTestModule(t, mod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	backend.Close()
}

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

func writeTestModule(t *testing.T, data []byte) string {
	t.Helper()
	tmpDir := t.TempDir()
	path := filepath.Join(tmpDir, "test.wasm")
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatalf("failed to write temp wasm file: %v", err)
	}
	return path
}

func TestForward_Identity(t *testing.T) {
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()

	input := []float32{1.5, -2.0, 3.0, 0.0, -0.5}
	out, err := backend.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(out) != len(input) {
		t.Fatalf("got %d outputs, want %d", len(out), len(input))
	}
	for i := range input {
		if out[i] != input[i] {
			t.Errorf("out[%d] = %f, want %f", i, out[i], input[i])
		}
	}
}

func TestGetIntermediate_NoExport(t *testing.T) {
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()

	_, err = backend.GetIntermediate(0)
	if err == nil {
		t.Error("expected error when module has no get_intermediate export")
	}
}

func TestMultipleForwardCalls(t *testing.T) {
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()

	for i := 0; i < 5; i++ {
		input := []float32{float32(i), float32(i) * 2}
		out, err := backend.Forward(input)
		if err != nil {
			t.Fatalf("Forward call %d failed: %v", i, err)
		}
		if len(out) != 2 {
			t.Fatalf("call %d: got %d outputs, want 2", i, len(out))
		}
		if out[0] != input[0] || out[1] != input[1] {
			t.Errorf("call %d: got %v, want %v", i, out, input)
		}
	}
}

func TestLargeInput(t *testing.T) {
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()

	input := make([]float32, 10000)
	for i := range input {
		input[i] = float32(i)
	}
	out, err := backend.Forward(input)
	if err != nil {
		t.Fatalf("Forward failed: %v", err)
	}
	if len(out) != len(input) {
		t.Fatalf("got %d outputs, want %d", len(out), len(input))
	}
	for i := range input {
		if out[i] != input[i] {
			t.Fatalf("out[%d] mismatch", i)
		}
	}
}

func TestForward_ValidModule(t *testing.T) {
	// Verify the hand-rolled module can be loaded and compiled
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v\nModule bytes:\n%x", err, testMod)
	}
	defer backend.Close()

	if !backend.loaded {
		t.Error("expected backend.loaded to be true")
	}
}

func TestForward_EmptyInput(t *testing.T) {
	testMod := buildTestModule()
	path := writeTestModule(t, testMod)
	backend, err := NewWASMBackend(path)
	if err != nil {
		t.Fatalf("NewWASMBackend failed: %v", err)
	}
	defer backend.Close()

	out, err := backend.Forward(nil)
	if err != nil {
		t.Fatalf("Forward(nil) failed: %v", err)
	}
	if len(out) != 0 {
		t.Errorf("got %d outputs, want 0", len(out))
	}
}

// dumpModuleHex prints the full WASM binary for debugging.
func dumpModuleHex(t *testing.T, data []byte) {
	t.Helper()
	for i, b := range data {
		if i > 0 && i%16 == 0 {
			fmt.Fprint(os.Stderr, "\n")
		}
		fmt.Fprintf(os.Stderr, "0x%02x, ", b)
	}
	fmt.Fprintln(os.Stderr)
}
