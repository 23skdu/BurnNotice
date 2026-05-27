package main

import (
	"fmt"
	"burnnotice/internal/tokenizer"
	"github.com/yalue/onnxruntime_go"
)

type ONNXModel struct {
	session *onnxruntime_go.DynamicAdvancedSession
	tok     tokenizer.Tokenizer
}

func LoadONNXModel(path string) (*ONNXModel, error) {
	if !onnxruntime_go.IsInitialized() {
		// Attempt to initialize using default shared library path if not already initialized
		// Note: The user must have the ONNX Runtime shared library installed.
		onnxruntime_go.SetSharedLibraryPath("libonnxruntime.dylib") 
		if err := onnxruntime_go.InitializeEnvironment(); err != nil {
			return nil, fmt.Errorf("failed to initialize onnxruntime: %w", err)
		}
	}

	// Assuming standard input/output names for LLM ONNX models
	inputNames := []string{"input_ids"}
	outputNames := []string{"hidden_states"}

	session, err := onnxruntime_go.NewDynamicAdvancedSession(path, inputNames, outputNames, nil)
	if err != nil {
		return nil, fmt.Errorf("failed to create session: %w", err)
	}

	return &ONNXModel{
		session: session,
		tok:     tokenizer.NewWordHashTokenizer(),
	}, nil
}

func (m *ONNXModel) Close() error {
	return m.session.Destroy()
}

func (m *ONNXModel) ExtractActivation(prompt, response string) ([]float64, error) {
	text := prompt + " " + response
	
	// Pre-allocate slice for zero-alloc tokenization
	tokens := make([]int64, 4096)
	count := m.tok.Tokenize(text, tokens)
	if count == 0 {
		return make([]float64, 4096), nil
	}

	// For standard inference, we truncate to the actual token count
	inputData := tokens[:count]
	
	inputTensor, err := onnxruntime_go.NewTensor(
		[]int64{1, int64(count)}, // batch_size=1, sequence_length
		inputData,
	)
	if err != nil {
		return nil, fmt.Errorf("create input tensor: %w", err)
	}
	defer inputTensor.Destroy()

	// Prepare output tensor (assuming hidden_states is [batch, seq, dim], here dim=4096)
	outputData := make([]float32, 1*count*4096)
	outputTensor, err := onnxruntime_go.NewTensor(
		[]int64{1, int64(count), 4096},
		outputData,
	)
	if err != nil {
		return nil, fmt.Errorf("create output tensor: %w", err)
	}
	defer outputTensor.Destroy()

	err = m.session.Run(
		[]onnxruntime_go.ArbitraryTensor{inputTensor},
		[]onnxruntime_go.ArbitraryTensor{outputTensor},
	)
	if err != nil {
		return nil, fmt.Errorf("run session: %w", err)
	}

	// Average pool over sequence length to get a single vector per request
	activation := make([]float64, 4096)
	for i := 0; i < count; i++ {
		offset := i * 4096
		for j := 0; j < 4096; j++ {
			activation[j] += float64(outputData[offset+j])
		}
	}
	for j := 0; j < 4096; j++ {
		activation[j] /= float64(count)
	}

	return activation, nil
}
