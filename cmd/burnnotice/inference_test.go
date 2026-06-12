package main

import (
	"burnnotice/internal/onnxpb"
	"os"
	"path/filepath"
	"strings"
	"testing"

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
	defer m.Close()

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
