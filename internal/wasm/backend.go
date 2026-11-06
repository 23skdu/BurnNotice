package wasm

import (
	"context"
	"encoding/binary"
	"fmt"
	"math"
	"os"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// WASMBackend executes ONNX models compiled to WebAssembly for pure-Go inference.
type WASMBackend struct {
	runtime wazero.Runtime
	module  wazero.CompiledModule
	config  wazero.ModuleConfig
	loaded  bool
}

// NewWASMBackend creates a new WASM inference backend.
func NewWASMBackend(wasmPath string) (*WASMBackend, error) {
	if _, err := os.Stat(wasmPath); os.IsNotExist(err) {
		return nil, fmt.Errorf("WASM file not found: %s", wasmPath)
	}

	ctx := context.Background()
	r := wazero.NewRuntime(ctx)

	_, err := wasi_snapshot_preview1.Instantiate(ctx, r)
	if err != nil {
		return nil, fmt.Errorf("instantiate WASI: %w", err)
	}

	// #nosec G304 -- CLI tool accepts user-provided WASM paths
	code, err := os.ReadFile(wasmPath)
	if err != nil {
		return nil, fmt.Errorf("read WASM file: %w", err)
	}

	mod, err := r.CompileModule(ctx, code)
	if err != nil {
		return nil, fmt.Errorf("compile WASM module: %w", err)
	}

	return &WASMBackend{
		runtime: r,
		module:  mod,
		config:  wazero.NewModuleConfig(),
		loaded:  true,
	}, nil
}

const wasmInputOffset = 0

// Forward runs inference on the input tensor via WASM execution.
// The WASM module must export:
//   - "memory" — linear memory for data exchange
//   - "forward" — function(inputLen: i32) -> (outputLen: i32)
//
// Input is written to WASM linear memory at offset 0.
// Output is read from WASM linear memory starting at offset 0,
// overwriting the input buffer.
func (b *WASMBackend) Forward(input []float32) ([]float32, error) {
	if !b.loaded {
		return nil, fmt.Errorf("WASM module not loaded")
	}

	if len(input) > 1<<24 {
		return nil, fmt.Errorf("input too large: %d floats", len(input))
	}

	ctx := context.Background()

	mod, err := b.runtime.InstantiateModule(ctx, b.module, b.config)
	if err != nil {
		return nil, fmt.Errorf("instantiate module: %w", err)
	}
	defer mod.Close(ctx)

	mem := mod.ExportedMemory("memory")
	if mem == nil {
		return nil, fmt.Errorf("module has no exported memory")
	}

	inputBuf := encodeF32(input)
	if !mem.Write(wasmInputOffset, inputBuf) {
		return nil, fmt.Errorf("failed to write input to WASM memory")
	}

	forward := mod.ExportedFunction("forward")
	if forward == nil {
		return nil, fmt.Errorf("module does not export 'forward'")
	}

	results, err := forward.Call(ctx, uint64(len(input))) // #nosec G115 — bounded by check above
	if err != nil {
		return nil, fmt.Errorf("forward execution failed: %w", err)
	}

	if len(results) == 0 {
		return nil, fmt.Errorf("forward returned no results")
	}
	// WASM i32 returns are stored in uint64, always < 2^32
	if results[0] > 1<<24 {
		return nil, fmt.Errorf("output length out of range: %d", results[0])
	}
	outputLen := int(results[0])                                    // #nosec G115 — bounded by check above
	outputBuf, ok := mem.Read(wasmInputOffset, uint32(outputLen*4)) // #nosec G115 — outputLen bounded to 1<<24 above
	if !ok {
		return nil, fmt.Errorf("failed to read output from WASM memory")
	}

	return decodeF32(outputBuf), nil
}

// GetIntermediate extracts activation values at a specified layer index from WASM memory.
// The WASM module must export a "get_intermediate" function with signature
// (layerIdx: i32) -> (offset: i32, length: i32).
func (b *WASMBackend) GetIntermediate(layerIdx int) ([]float32, error) {
	if !b.loaded {
		return nil, fmt.Errorf("WASM module not loaded")
	}

	ctx := context.Background()

	mod, err := b.runtime.InstantiateModule(ctx, b.module, b.config)
	if err != nil {
		return nil, fmt.Errorf("instantiate module: %w", err)
	}
	defer mod.Close(ctx)

	getIntermediate := mod.ExportedFunction("get_intermediate")
	if getIntermediate == nil {
		return nil, fmt.Errorf("module does not export 'get_intermediate'")
	}

	if layerIdx < 0 {
		return nil, fmt.Errorf("negative layer index: %d", layerIdx)
	}
	results, err := getIntermediate.Call(ctx, uint64(layerIdx))
	if err != nil {
		return nil, fmt.Errorf("get_intermediate failed: %w", err)
	}

	if len(results) < 2 {
		return nil, fmt.Errorf("get_intermediate returned %d values, want 2", len(results))
	}
	if results[0] > 1<<24 || results[1] > 1<<24 {
		return nil, fmt.Errorf("intermediate offset/length out of range: %d, %d", results[0], results[1])
	}
	offset := uint32(results[0]) // #nosec G115 — bounded by check above
	length := int(results[1])    // #nosec G115 — bounded by check above
	if length <= 0 {
		return nil, nil
	}

	mem := mod.ExportedMemory("memory")
	if mem == nil {
		return nil, fmt.Errorf("module has no exported memory")
	}

	if length > 1<<24 {
		return nil, fmt.Errorf("intermediate length out of range: %d", length)
	}
	buf, ok := mem.Read(offset, uint32(length*4))
	if !ok {
		return nil, fmt.Errorf("failed to read intermediate from WASM memory")
	}

	return decodeF32(buf), nil
}

// Close releases the WASM runtime resources.
func (b *WASMBackend) Close() error {
	if b.runtime != nil {
		return b.runtime.Close(context.Background())
	}
	return nil
}

func encodeF32(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func decodeF32(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

// Compile-time interface checks
var _ interface{ Close() error } = (*WASMBackend)(nil)
