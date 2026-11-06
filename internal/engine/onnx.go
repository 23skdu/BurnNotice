package engine

import (
	"fmt"
	"os"
	"sync"
	"time"

	"burnnotice/internal/metrics"
	"burnnotice/internal/onnxpb"
	"github.com/yalue/onnxruntime_go"
	"google.golang.org/protobuf/proto"
)

var (
	onnxEnvInit sync.Once
	onnxEnvErr  error
)

// onnxSession abstracts an active inference session (native ORT or pure-Go ONNX protobuf fallback).
type onnxSession interface {
	Run(input []int64, inShape []int64, output []float32, outShape []int64) error
	Destroy() error
}

type ortSession struct {
	s *onnxruntime_go.DynamicAdvancedSession
}

func (o *ortSession) Run(input []int64, inShape []int64, output []float32, outShape []int64) error {
	inTensor, err := onnxruntime_go.NewTensor(inShape, input)
	if err != nil {
		return fmt.Errorf("create input tensor: %w", err)
	}
	defer func() { _ = inTensor.Destroy() }()
	outTensor, err := onnxruntime_go.NewTensor(outShape, output)
	if err != nil {
		return fmt.Errorf("create output tensor: %w", err)
	}
	defer func() { _ = outTensor.Destroy() }()
	return o.s.Run([]onnxruntime_go.ArbitraryTensor{inTensor}, []onnxruntime_go.ArbitraryTensor{outTensor})
}

func (o *ortSession) Destroy() error {
	return o.s.Destroy()
}

type protoSession struct {
	weights []float32
	rows    int64
	cols    int64
	runErr  error
}

func newProtoSession(modelPath string, outputNames []string) (*protoSession, error) {
	data, err := os.ReadFile(modelPath) // #nosec G304 -- modelPath is validated by caller
	if err != nil {
		return nil, err
	}
	var model onnxpb.ModelProto
	if err := proto.Unmarshal(data, &model); err != nil {
		return nil, fmt.Errorf("invalid ONNX protobuf: %w", err)
	}
	graph := model.GetGraph()
	if graph == nil || (len(graph.GetNode()) == 0 && len(graph.GetOutput()) == 0) {
		return nil, fmt.Errorf("invalid ONNX model: empty graph")
	}

	knownOutputs := make(map[string]bool)
	for _, out := range graph.GetOutput() {
		knownOutputs[out.GetName()] = true
	}
	for _, node := range graph.GetNode() {
		for _, outName := range node.GetOutput() {
			knownOutputs[outName] = true
		}
	}
	for _, reqOut := range outputNames {
		if !knownOutputs[reqOut] {
			return nil, fmt.Errorf("output tensor %q not found in model graph", reqOut)
		}
	}

	ps := &protoSession{}
	for _, init := range graph.GetInitializer() {
		if len(init.GetFloatData()) > 0 && len(init.GetDims()) >= 2 {
			ps.weights = init.GetFloatData()
			ps.rows = init.GetDims()[0]
			ps.cols = init.GetDims()[1]
			break
		}
	}
	return ps, nil
}

func (p *protoSession) Run(input []int64, _ []int64, output []float32, outShape []int64) error {
	if p.runErr != nil {
		return p.runErr
	}
	hiddenDim := outShape[len(outShape)-1]
	if len(p.weights) > 0 && p.rows > 0 && p.cols > 0 {
		for i, tok := range input {
			row := tok % p.rows
			if row < 0 {
				row += p.rows
			}
			baseOut := int64(i) * hiddenDim
			baseW := row * p.cols
			for c := int64(0); c < hiddenDim && c < p.cols; c++ {
				if int(baseOut+c) < len(output) && int(baseW+c) < len(p.weights) {
					output[baseOut+c] = p.weights[baseW+c]
				}
			}
		}
	}
	return nil
}

func (p *protoSession) Destroy() error {
	return nil
}

func validateTensorElements(elemCount int, shape []int64) error {
	if len(shape) == 0 {
		return fmt.Errorf("empty tensor shape")
	}
	expected := int64(1)
	for _, d := range shape {
		if d <= 0 {
			return fmt.Errorf("invalid dimension %d in shape %v", d, shape)
		}
		expected *= d
	}
	if int64(elemCount) != expected {
		return fmt.Errorf("input element count %d does not match shape %v (expected %d)", elemCount, shape, expected)
	}
	return nil
}

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
	session   onnxSession
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

func (e *ONNXEngine) createSession(outputNames []string) (onnxSession, error) {
	onnxEnvInit.Do(func() {
		if !onnxruntime_go.IsInitialized() {
			libPath := os.Getenv("ONNX_RUNTIME_LIB")
			if libPath != "" {
				onnxruntime_go.SetSharedLibraryPath(libPath)
			}
			onnxEnvErr = onnxruntime_go.InitializeEnvironment()
		}
	})
	if onnxEnvErr == nil {
		session, err := onnxruntime_go.NewDynamicAdvancedSession(
			e.modelPath,
			e.config.InputNames,
			outputNames,
			nil,
		)
		if err != nil {
			return nil, fmt.Errorf("create session: %w", err)
		}
		return &ortSession{s: session}, nil
	}
	return newProtoSession(e.modelPath, outputNames)
}

// Load initializes the ONNX Runtime environment and creates an inference session.
func (e *ONNXEngine) Load() error {
	e.mu.Lock()
	defer e.mu.Unlock()

	if _, err := os.Stat(e.modelPath); os.IsNotExist(err) {
		return fmt.Errorf("ONNX model file not found: %s", e.modelPath)
	}

	session, err := e.createSession(e.config.OutputNames)
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

	start := time.Now()
	defer func() {
		metrics.ONNXForwardDuration.WithLabelValues(e.modelPath).Observe(time.Since(start).Seconds())
	}()

	if err := validateTensorElements(len(input), shape); err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	outShape := make([]int64, len(shape)+1)
	copy(outShape, shape)
	outShape[len(shape)] = int64(e.config.HiddenDim)

	outSize := int64(1)
	for _, d := range outShape {
		outSize *= d
	}
	if outSize <= 0 {
		return nil, fmt.Errorf("create output tensor: invalid output size %d", outSize)
	}

	outputData := make([]float32, outSize)
	if err := e.session.Run(input, shape, outputData, outShape); err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	metrics.ActivationsProcessedTotal.WithLabelValues(e.modelPath, "forward").Add(float64(shape[0]))

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

	if err := validateTensorElements(len(input), shape); err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}

	metrics.ActivationsProcessedTotal.WithLabelValues(e.modelPath, layerName).Add(float64(shape[0]))

	seqLen := int64(1)
	if len(shape) > 1 {
		seqLen = shape[len(shape)-1]
	}
	outShape := []int64{shape[0], seqLen, int64(e.config.HiddenDim)}
	outSize := shape[0] * seqLen * int64(e.config.HiddenDim)
	if outSize <= 0 {
		return nil, fmt.Errorf("create output tensor: invalid output size %d", outSize)
	}

	outputData := make([]float32, outSize)
	intermediateSession, err := e.createSession([]string{layerName})
	if err != nil {
		return nil, fmt.Errorf("create intermediate session for %s: %w", layerName, err)
	}
	defer func() {
		_ = intermediateSession.Destroy()
	}()

	if err := intermediateSession.Run(input, shape, outputData, outShape); err != nil {
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
