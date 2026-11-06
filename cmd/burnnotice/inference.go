package main

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"

	"burnnotice/internal/engine"
	"burnnotice/internal/profile"
	"burnnotice/internal/tokenizer"
)

type ONNXModel struct {
	eng    *engine.ONNXEngine
	tok    tokenizer.Tokenizer
	dim    int
	family profile.ModelFamily
}

func LoadONNXModel(path string) (*ONNXModel, error) {
	hiddenDim := 4096

	cfgCandidate := filepath.Clean(filepath.Join(filepath.Dir(path), "config.json"))
	var configBytes []byte
	if data, err := os.ReadFile(cfgCandidate); err == nil { // #nosec G304 -- model path is user-provided CLI arg
		configBytes = data
		var hfCfg profile.HuggingFaceConfig
		if err := json.Unmarshal(data, &hfCfg); err == nil && hfCfg.HiddenSize > 0 {
			hiddenDim = hfCfg.HiddenSize
		}
	}

	archSpec := profile.DetectModelFamily(filepath.Base(path), configBytes, nil)
	family := archSpec.Family

	cfg := engine.ModelConfig{
		HiddenDim:   hiddenDim,
		InputNames:  []string{"input_ids"},
		OutputNames: []string{"hidden_states"},
	}
	eng := engine.NewONNXEngine(path, cfg)
	if err := eng.Load(); err != nil {
		return nil, fmt.Errorf("load ONNX model: %w", err)
	}

	return &ONNXModel{
		eng:    eng,
		tok:    tokenizer.NewWordHashTokenizer(),
		dim:    cfg.HiddenDim,
		family: family,
	}, nil
}

func (m *ONNXModel) Close() error {
	return m.eng.Close()
}

func (m *ONNXModel) ExtractActivation(prompt, response string) ([]float64, error) {
	var text string
	if m.family != "" && m.family != profile.FamilyUnknown {
		text = profile.FormatPromptForFamily(m.family, prompt, response)
	} else {
		text = prompt + " " + response
	}

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
