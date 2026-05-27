package engine

import (
	"fmt"
	"os"
	"sync"

	"github.com/yalue/onnxruntime_go"
)

var (
	onnxEnvInit sync.Once
	onnxEnvErr  error
)

// ModelConfig holds model architecture parameters and I/O tensor names.
type ModelConfig struct {
	HiddenDim          int
	NumLayers          int
	LayerNames         []string
	AttentionHeadCount int
	InputNames         []string
	OutputNames        []string
}

// ONNXEngine loads and executes models via ONNX Runtime, capturing per-layer activations.
type ONNXEngine struct {
	modelPath string
	config    ModelConfig
	session   *onnxruntime_go.DynamicAdvancedSession
	loaded    bool
	mu        sync.Mutex
}

// NewONNXEngine creates a new ONNX engine instance.
func NewONNXEngine(modelPath string, config ModelConfig) *ONNXEngine {
	if len(config.InputNames) == 0 {
		config.InputNames = []string{"input_ids"}
	}
	if len(config.OutputNames) == 0 {
		config.OutputNames = []string{"hidden_states"}
	}
	return &ONNXEngine{
		modelPath: modelPath,
		config:    config,
	}
}

// Load initializes the ONNX Runtime environment and creates an inference session.
func (e *ONNXEngine) Load() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, err := os.Stat(e.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("ONNX model file not found: %s", e.modelPath)
	}

	onnxEnvInit.Do(func() {
		if !onnxruntime_go.IsInitialized() {
			libPath := os.Getenv("ONNX_RUNTIME_LIB")
			if libPath != "" {
				onnxruntime_go.SetSharedLibraryPath(libPath)
			}
			onnxEnvErr = onnxruntime_go.InitializeEnvironment()
		}
	})
	if onnxEnvErr != nil {
		return fmt.Errorf("init onnx runtime: %w", onnxEnvErr)
	}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		e.modelPath,
		e.config.InputNames,
		e.config.OutputNames,
		nil,
	)
	if err != nil {
		return fmt.Errorf("create session: %w", err)
	}
	e.session = session
	e.loaded = true
	return nil
}

// Forward runs inference on int64 token IDs and returns the first output tensor as float32.
// input is expected as token IDs (int64) with shape [batch, seq_len].
func (e *ONNXEngine) Forward(input []int64, shape []int64) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return nil, fmt.Errorf("model not loaded")
	}

	inputTensor, err := onnxruntime_go.NewTensor(shape, input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	outShape := make([]int64, len(shape)+1)
	copy(outShape, shape)
	outShape[len(shape)] = int64(e.config.HiddenDim)

	outSize := int64(1)
	for _, d := range outShape {
		outSize *= d
	}

	outputData := make([]float32, outSize)
	outputTensor, err := onnxruntime_go.NewTensor(outShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = e.session.Run(
		[]onnxruntime_go.ArbitraryTensor{inputTensor},
		[]onnxruntime_go.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	return outputData, nil
}

// GetIntermediate extracts intermediate activations by name from the ONNX model.
// The model must export intermediate tensors as named outputs.
func (e *ONNXEngine) GetIntermediate(layerName string, input []int64, shape []int64) ([]float32, error) {
	e.mu.Lock()
	defer e.mu.Unlock()

	if !e.loaded {
		return nil, fmt.Errorf("model not loaded")
	}

	inputTensor, err := onnxruntime_go.NewTensor(shape, input)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	seqLen := int64(1)
	if len(shape) > 1 {
		seqLen = shape[len(shape)-1]
	}
	outShape := []int64{shape[0], seqLen, int64(e.config.HiddenDim)}

	outputData := make([]float32, shape[0]*seqLen*int64(e.config.HiddenDim))
	outputTensor, err := onnxruntime_go.NewTensor(outShape, outputData)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	intermediateSession, err := onnxruntime_go.NewDynamicAdvancedSession(
		e.modelPath,
		e.config.InputNames,
		[]string{layerName},
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create intermediate session for %s: %w", layerName, err)
	}
	defer intermediateSession.Destroy()

	err = intermediateSession.Run(
		[]onnxruntime_go.ArbitraryTensor{inputTensor},
		[]onnxruntime_go.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("run intermediate session: %w", err)
	}

	return outputData, nil
}

// NumOutputs returns the number of named output tensors.
func (e *ONNXEngine) NumOutputs() int {
	return len(e.config.OutputNames)
}

// Config returns the model configuration.
func (e *ONNXEngine) Config() ModelConfig {
	return e.config
}

// Close releases the ONNX Runtime session.
func (e *ONNXEngine) Close() error {
	e.mu.Lock()
	defer e.mu.Unlock()
	if e.session != nil {
		return e.session.Destroy()
	}
	return nil
}
