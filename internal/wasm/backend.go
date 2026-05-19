package wasm

import (
	"context"
	"fmt"
	"os"

	"github.com/tetratelabs/wazero"
	"github.com/tetratelabs/wazero/imports/wasi_snapshot_preview1"
)

// WASMBackend executes ONNX models compiled to WebAssembly for pure-Go inference.
type WASMBackend struct {
	runtime  wazero.Runtime
	module   wazero.CompiledModule
	config   wazero.ModuleConfig
	loaded   bool
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

// Forward runs inference on the input tensor via WASM execution.
func (b *WASMBackend) Forward(input []float32) ([]float32, error) {
	if !b.loaded {
		return nil, fmt.Errorf("WASM module not loaded")
	}

	ctx := context.Background()

	// TODO: Execute WASM forward pass:
	// 1. Instantiate WASM module with input memory
	// 2. Write input []float32 to WASM linear memory
	// 3. Call exported "forward" function
	// 4. Read output from WASM memory
	// 5. Return output []float32

	_ = ctx
	return nil, nil
}

// GetIntermediate extracts activation values at a specified layer index from WASM memory.
func (b *WASMBackend) GetIntermediate(layerIdx int) ([]float32, error) {
	if !b.loaded {
		return nil, fmt.Errorf("WASM module not loaded")
	}

	// TODO: Read intermediate activation buffer from WASM memory at known offset for layerIdx
	return nil, nil
}

// Close releases the WASM runtime resources.
func (b *WASMBackend) Close() error {
	if b.runtime != nil {
		return b.runtime.Close(context.Background())
	}
	return nil
}
