package main

import (
	"encoding/json"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"burnnotice/internal/onnxpb"
	"burnnotice/internal/profile"

	"google.golang.org/protobuf/proto"
)

func createDummyONNXModelForCLI(hiddenDim int) ([]byte, error) {
	model := &onnxpb.ModelProto{
		IrVersion: pbPtr(int64(8)),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{
				Domain:  pbPtr(""),
				Version: pbPtr(int64(15)),
			},
		},
		Graph: &onnxpb.GraphProto{
			Name: pbPtr("dummy_graph"),
			Node: []*onnxpb.NodeProto{
				{
					OpType: pbPtr("Gather"),
					Input:  []string{"embedding_weight", "input_ids"},
					Output: []string{"hidden_states"},
					Attribute: []*onnxpb.AttributeProto{
						{
							Name: pbPtr("axis"),
							Type: onnxpb.AttributeProto_INT.Enum(),
							I:    pbPtr(int64(0)),
						},
					},
				},
			},
			Initializer: []*onnxpb.TensorProto{
				{
					Dims:      []int64{10, int64(hiddenDim)},
					DataType:  pbPtr(int32(1)), // FLOAT = 1
					Name:      pbPtr("embedding_weight"),
					FloatData: make([]float32, 10*hiddenDim),
				},
			},
			Input: []*onnxpb.ValueInfoProto{
				{
					Name: pbPtr("input_ids"),
					Type: &onnxpb.TypeProto{
						Value: &onnxpb.TypeProto_TensorType{
							TensorType: &onnxpb.TypeProto_Tensor{
								ElemType: pbPtr(int32(7)), // INT64 = 7
								Shape: &onnxpb.TensorShapeProto{
									Dim: []*onnxpb.TensorShapeProto_Dimension{
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 10}},
									},
								},
							},
						},
					},
				},
			},
			Output: []*onnxpb.ValueInfoProto{
				{
					Name: pbPtr("hidden_states"),
					Type: &onnxpb.TypeProto{
						Value: &onnxpb.TypeProto_TensorType{
							TensorType: &onnxpb.TypeProto_Tensor{
								ElemType: pbPtr(int32(1)), // FLOAT = 1
								Shape: &onnxpb.TensorShapeProto{
									Dim: []*onnxpb.TensorShapeProto_Dimension{
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 1}},
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 10}},
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: int64(hiddenDim)}},
									},
								},
							},
						},
					},
				},
			},
		},
	}

	return proto.Marshal(model)
}

func pbPtr[T any](v T) *T {
	return &v
}

func TestONNXModel_Inference(t *testing.T) {
	hiddenDim := 4096
	modelBytes, err := createDummyONNXModelForCLI(hiddenDim)
	if err != nil {
		t.Fatalf("failed to create dummy model: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, modelBytes, 0644); err != nil {
		t.Fatal(err)
	}

	m, err := LoadONNXModel(path)
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Skipf("skipping inference test because ONNX shared library is not available: %v", err)
		}
		t.Fatalf("LoadONNXModel failed: %v", err)
	}
	defer func() { _ = m.Close() }()

	// Test ExtractActivation
	act, err := m.ExtractActivation("hello", "world")
	if err != nil {
		t.Fatalf("ExtractActivation failed: %v", err)
	}
	if len(act) != hiddenDim {
		t.Errorf("len(act) = %d, want %d", len(act), hiddenDim)
	}

	// Test ExtractActivation with empty string
	actEmpty, err := m.ExtractActivation("", "")
	if err != nil {
		t.Fatalf("ExtractActivation with empty input failed: %v", err)
	}
	if len(actEmpty) != hiddenDim {
		t.Errorf("len(actEmpty) = %d, want %d", len(actEmpty), hiddenDim)
	}
}

func TestONNXModel_ArchitectureCompatibility(t *testing.T) {
	families := []struct {
		name       string
		configJSON string
		hiddenDim  int
		wantFamily string
	}{
		{
			name:       "qwen",
			configJSON: `{"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896}`,
			hiddenDim:  896,
			wantFamily: "qwen",
		},
		{
			name:       "gemma",
			configJSON: `{"architectures": ["Gemma2ForCausalLM"], "hidden_size": 2304}`,
			hiddenDim:  2304,
			wantFamily: "gemma",
		},
		{
			name:       "llama",
			configJSON: `{"architectures": ["LlamaForCausalLM"], "hidden_size": 2048}`,
			hiddenDim:  2048,
			wantFamily: "llama",
		},
	}

	for _, tc := range families {
		t.Run(tc.name, func(t *testing.T) {
			modelBytes, err := createDummyONNXModelForCLI(tc.hiddenDim)
			if err != nil {
				t.Fatalf("failed to create dummy model: %v", err)
			}

			dir := t.TempDir()
			modelPath := filepath.Join(dir, "model.onnx")
			if err := os.WriteFile(modelPath, modelBytes, 0644); err != nil {
				t.Fatal(err)
			}
			cfgPath := filepath.Join(dir, "config.json")
			if err := os.WriteFile(cfgPath, []byte(tc.configJSON), 0644); err != nil {
				t.Fatal(err)
			}

			m, err := LoadONNXModel(modelPath)
			if err != nil {
				if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
					t.Skipf("skipping inference test because ONNX shared library is not available: %v", err)
				}
				t.Fatalf("LoadONNXModel failed for %s: %v", tc.name, err)
			}
			defer func() { _ = m.Close() }()

			if string(m.family) != tc.wantFamily {
				t.Errorf("%s: expected family %s, got %s", tc.name, tc.wantFamily, m.family)
			}
			if m.dim != tc.hiddenDim {
				t.Errorf("%s: expected dim %d, got %d", tc.name, tc.hiddenDim, m.dim)
			}

			act, err := m.ExtractActivation("Why is the sky blue?", "Rayleigh scattering.")
			if err != nil {
				t.Fatalf("%s: ExtractActivation failed: %v", tc.name, err)
			}
			if len(act) != tc.hiddenDim {
				t.Errorf("%s: len(act) = %d, want %d", tc.name, len(act), tc.hiddenDim)
			}
		})
	}
}

func TestDetectArchitecture_LlamaQwenGemma(t *testing.T) {
	cases := []struct {
		modelName  string
		configJSON string
		wantFamily string
		wantDim    int
	}{
		{
			modelName:  "qwen2.5-0.5b",
			configJSON: `{"architectures": ["Qwen2ForCausalLM"], "hidden_size": 896}`,
			wantFamily: "qwen",
			wantDim:    896,
		},
		{
			modelName:  "gemma-2-2b",
			configJSON: `{"architectures": ["Gemma2ForCausalLM"], "hidden_size": 2304}`,
			wantFamily: "gemma",
			wantDim:    2304,
		},
		{
			modelName:  "llama-3.2-1b",
			configJSON: `{"architectures": ["LlamaForCausalLM"], "hidden_size": 2048}`,
			wantFamily: "llama",
			wantDim:    2048,
		},
	}

	for _, tc := range cases {
		t.Run(tc.modelName, func(t *testing.T) {
			spec := profile.DetectModelFamily(tc.modelName, []byte(tc.configJSON), nil)
			if string(spec.Family) != tc.wantFamily {
				t.Errorf("expected family %s, got %s", tc.wantFamily, spec.Family)
			}
			var hfCfg profile.HuggingFaceConfig
			if err := json.Unmarshal([]byte(tc.configJSON), &hfCfg); err != nil {
				t.Fatal(err)
			}
			if hfCfg.HiddenSize != tc.wantDim {
				t.Errorf("expected hidden dim %d, got %d", tc.wantDim, hfCfg.HiddenSize)
			}
		})
	}
}
