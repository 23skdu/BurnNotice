package engine

import (
	"fmt"
	"os"
)

// ONNXEngine loads and executes models in ONNX format.
type ONNXEngine struct {
	modelPath string
	config    ModelConfig
	loaded    bool
}

// ModelConfig holds model architecture parameters.
type ModelConfig struct {
	HiddenDim         int
	NumLayers         int
	LayerNames        []string
	AttentionHeadCount int
}

// NewONNXEngine creates a new ONNX engine instance.
func NewONNXEngine(modelPath string, config ModelConfig) *ONNXEngine {
	return &ONNXEngine{
		modelPath: modelPath,
		config:    config,
	}
}

// Load loads the ONNX model from disk.
func (e *ONNXEngine) Load() error {
	if _, err := os.Stat(e.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("ONNX model file not found: %s", e.modelPath)
	}

	// TODO: Use owulveryck/onnx-go to:
	// 1. Read ONNX file bytes
	// 2. Parse model protobuf
	// 3. Initialize ONNX Runtime backend
	// 4. Validate input/output tensor shapes match config

	e.loaded = true
	return nil
}

// Forward runs inference on the input tensor and returns the output.
func (e *ONNXEngine) Forward(input []float32) ([]float32, error) {
	if !e.loaded {
		return nil, fmt.Errorf("model not loaded")
	}

	// TODO: Execute ONNX forward pass:
	// 1. Create input tensor from []float32
	// 2. Run inference via ONNX Runtime
	// 3. Extract output tensor as []float32

	return nil, nil
}

// GetIntermediate extracts intermediate activation values at a specified layer.
// This simulates forward hook behavior by reading intermediate tensor values.
func (e *ONNXEngine) GetIntermediate(layerName string) ([]float32, error) {
	if !e.loaded {
		return nil, fmt.Errorf("model not loaded")
	}

	// TODO: Extract intermediate tensor:
	// 1. Find tensor by layerName in ONNX graph
	// 2. Run partial forward to that layer
	// 3. Return tensor data as []float32

	return nil, nil
}

// Config returns the model configuration.
func (e *ONNXEngine) Config() ModelConfig {
	return e.config
}
