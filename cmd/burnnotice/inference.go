package main

import (
	"fmt"
	"burnnotice/internal/engine"
	"burnnotice/internal/tokenizer"
)

type ONNXModel struct {
	eng  *engine.ONNXEngine
	tok  tokenizer.Tokenizer
	dim  int
}

func LoadONNXModel(path string) (*ONNXModel, error) {
	cfg := engine.ModelConfig{
		HiddenDim:  4096,
		InputNames: []string{"input_ids"},
		OutputNames: []string{"hidden_states"},
	}
	eng := engine.NewONNXEngine(path, cfg)
	if err := eng.Load(); err != nil {
		return nil, fmt.Errorf("load ONNX model: %w", err)
	}

	return &ONNXModel{
		eng: eng,
		tok: tokenizer.NewWordHashTokenizer(),
		dim: cfg.HiddenDim,
	}, nil
}

func (m *ONNXModel) Close() error {
	return m.eng.Close()
}

func (m *ONNXModel) ExtractActivation(prompt, response string) ([]float64, error) {
	text := prompt + " " + response

	tokens := make([]int64, m.dim)
	count := m.tok.Tokenize(text, tokens)
	if count == 0 {
		return make([]float64, m.dim), nil
	}

	inputData := tokens[:count]

	outputData, err := m.eng.Forward(inputData, []int64{1, int64(count)})
	if err != nil {
		return nil, fmt.Errorf("forward: %w", err)
	}

	// Average pool over sequence length to get a single vector per request
	activation := make([]float64, m.dim)
	for i := 0; i < count; i++ {
		offset := i * m.dim
		for j := 0; j < m.dim; j++ {
			activation[j] += float64(outputData[offset+j])
		}
	}
	for j := 0; j < m.dim; j++ {
		activation[j] /= float64(count)
	}

	return activation, nil
}
