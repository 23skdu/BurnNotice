package engine

import (
	"burnnotice/internal/onnxpb"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"google.golang.org/protobuf/proto"
)

func TestNewONNXEngine(t *testing.T) {
	cfg := ModelConfig{
		HiddenDim:          4096,
		NumLayers:          32,
		LayerNames:         []string{"layer.0", "layer.1"},
		AttentionHeadCount: 32,
	}

	e := NewONNXEngine("/path/to/model.onnx", cfg)
	if e.modelPath != "/path/to/model.onnx" {
		t.Errorf("modelPath = %q, want /path/to/model.onnx", e.modelPath)
	}
	if e.loaded {
		t.Error("engine should not be loaded initially")
	}

	got := e.Config()
	if got.HiddenDim != 4096 {
		t.Errorf("HiddenDim = %d, want 4096", got.HiddenDim)
	}
	if got.NumLayers != 32 {
		t.Errorf("NumLayers = %d, want 32", got.NumLayers)
	}
	if len(got.LayerNames) != 2 {
		t.Errorf("len(LayerNames) = %d, want 2", len(got.LayerNames))
	}
	if got.AttentionHeadCount != 32 {
		t.Errorf("AttentionHeadCount = %d, want 32", got.AttentionHeadCount)
	}
}

func TestNewONNXEngine_DefaultIONames(t *testing.T) {
	e := NewONNXEngine("/path/to/model.onnx", ModelConfig{})
	if len(e.config.InputNames) != 1 || e.config.InputNames[0] != "input_ids" {
		t.Errorf("default InputNames = %v, want [input_ids]", e.config.InputNames)
	}
	if len(e.config.OutputNames) != 1 || e.config.OutputNames[0] != "hidden_states" {
		t.Errorf("default OutputNames = %v, want [hidden_states]", e.config.OutputNames)
	}
}

func TestNewONNXEngine_CustomIONames(t *testing.T) {
	cfg := ModelConfig{
		InputNames:  []string{"input", "attention_mask"},
		OutputNames: []string{"logits", "hidden_states_0", "hidden_states_1"},
	}
	e := NewONNXEngine("/path/to/model.onnx", cfg)
	if len(e.config.InputNames) != 2 {
		t.Errorf("InputNames count = %d, want 2", len(e.config.InputNames))
	}
	if len(e.config.OutputNames) != 3 {
		t.Errorf("OutputNames count = %d, want 3", len(e.config.OutputNames))
	}
}

func TestLoad_FileNotFound(t *testing.T) {
	e := NewONNXEngine("/nonexistent/model.onnx", ModelConfig{})
	err := e.Load()
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestForward_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.Forward([]int64{1, 2, 3}, []int64{1, 3})
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestGetIntermediate_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	_, err := e.GetIntermediate("layer.0", []int64{1}, []int64{1, 1})
	if err == nil {
		t.Error("expected error when model not loaded")
	}
}

func TestNumOutputs(t *testing.T) {
	cfg := ModelConfig{
		OutputNames: []string{"a", "b", "c"},
	}
	e := NewONNXEngine("/tmp/model.onnx", cfg)
	if n := e.NumOutputs(); n != 3 {
		t.Errorf("NumOutputs = %d, want 3", n)
	}
}

func TestClose_NotLoaded(t *testing.T) {
	e := NewONNXEngine("/tmp/model.onnx", ModelConfig{})
	if err := e.Close(); err != nil {
		t.Errorf("Close on unloaded engine should succeed: %v", err)
	}
}

func TestLoad_InvalidONNX(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, []byte("not a valid onnx file"), 0644); err != nil {
		t.Fatal(err)
	}

	e := NewONNXEngine(path, ModelConfig{
		HiddenDim: 4096,
	})
	err := e.Load()
	if err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Skip("skipping test because ONNX shared library is not available")
		}
	} else {
		t.Error("expected error for invalid ONNX file")
	}
}

func TestEngine_ConfigAccess(t *testing.T) {
	cfg := ModelConfig{
		HiddenDim:          2048,
		NumLayers:          16,
		LayerNames:         []string{"layer.0", "layer.1", "layer.2"},
		AttentionHeadCount: 16,
	}
	e := NewONNXEngine("/tmp/model.onnx", cfg)
	got := e.Config()
	if got.HiddenDim != cfg.HiddenDim {
		t.Errorf("HiddenDim = %d, want %d", got.HiddenDim, cfg.HiddenDim)
	}
	if got.NumLayers != cfg.NumLayers {
		t.Errorf("NumLayers = %d, want %d", got.NumLayers, cfg.NumLayers)
	}
	if got.AttentionHeadCount != cfg.AttentionHeadCount {
		t.Errorf("AttentionHeadCount = %d, want %d", got.AttentionHeadCount, cfg.AttentionHeadCount)
	}
}

func createDummyONNXModel(hiddenDim int) ([]byte, error) {
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
					Dims:     []int64{10, int64(hiddenDim)},
					DataType: pbPtr(int32(1)), // FLOAT = 1
					Name:     pbPtr("embedding_weight"),
					FloatData: []float32{
						0.1, 0.2, 0.3, 0.4,
						0.5, 0.6, 0.7, 0.8,
						0.9, 1.0, 1.1, 1.2,
						1.3, 1.4, 1.5, 1.6,
						1.7, 1.8, 1.9, 2.0,
						2.1, 2.2, 2.3, 2.4,
						2.5, 2.6, 2.7, 2.8,
						2.9, 3.0, 3.1, 3.2,
						3.3, 3.4, 3.5, 3.6,
						3.7, 3.8, 3.9, 4.0,
					},
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
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}},
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
										{Value: &onnxpb.TensorShapeProto_Dimension_DimValue{DimValue: 3}},
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

func TestONNXEngine_EndToEnd(t *testing.T) {
	hiddenDim := 4
	modelBytes, err := createDummyONNXModel(hiddenDim)
	if err != nil {
		t.Fatalf("failed to create dummy ONNX model: %v", err)
	}

	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	if err := os.WriteFile(path, modelBytes, 0644); err != nil {
		t.Fatal(err)
	}

	cfg := ModelConfig{
		HiddenDim:          hiddenDim,
		NumLayers:          1,
		LayerNames:         []string{"hidden_states"},
		AttentionHeadCount: 1,
	}

	e := NewONNXEngine(path, cfg)
	defer func() { _ = e.Close() }()

	if err := e.Load(); err != nil {
		if strings.Contains(err.Error(), "onnxruntime") || strings.Contains(err.Error(), "shared library") {
			t.Skipf("skipping end-to-end test because ONNX shared library is not available: %v", err)
		}
		t.Fatalf("Load() failed: %v", err)
	}

	// 1 sample, 3 tokens
	input := []int64{0, 1, 2}
	shape := []int64{1, 3}

	// Test Forward
	out, err := e.Forward(input, shape)
	if err != nil {
		t.Fatalf("Forward() failed: %v", err)
	}

	// Expected output shape is [1, 3, 4] -> size 12
	if len(out) != 12 {
		t.Fatalf("len(out) = %d, want 12", len(out))
	}

	// Check if output is non-empty/non-zero
	hasNonZero := false
	for _, val := range out {
		if val != 0 {
			hasNonZero = true
			break
		}
	}
	if !hasNonZero {
		t.Error("Forward output was all zeros, expected gathered embedding values")
	}

	// Test GetIntermediate
	interOut, err := e.GetIntermediate("hidden_states", input, shape)
	if err != nil {
		t.Fatalf("GetIntermediate() failed: %v", err)
	}
	if len(interOut) != 12 {
		t.Fatalf("len(interOut) = %d, want 12", len(interOut))
	}

	// Test error paths
	// 1. Forward with mismatching input size / shape
	_, err = e.Forward([]int64{1, 2}, []int64{1, 3})
	if err == nil {
		t.Error("expected error for size/shape mismatch in Forward")
	}

	// 2. GetIntermediate with mismatching input size / shape
	_, err = e.GetIntermediate("hidden_states", []int64{1, 2}, []int64{1, 3})
	if err == nil {
		t.Error("expected error for size/shape mismatch in GetIntermediate")
	}

	// 3. GetIntermediate with non-existent layer name
	_, err = e.GetIntermediate("non_existent_layer", input, shape)
	if err == nil {
		t.Error("expected error for non-existent layer in GetIntermediate")
	}

	// 4. Negative token ID and 1D shape in Forward and GetIntermediate
	out1D, err := e.Forward([]int64{-1, 2}, []int64{2})
	if err != nil || len(out1D) != 8 {
		t.Errorf("Forward 1D with negative token failed: len=%d err=%v", len(out1D), err)
	}
	inter1D, err := e.GetIntermediate("hidden_states", []int64{-1}, []int64{1})
	if err != nil || len(inter1D) != 4 {
		t.Errorf("GetIntermediate 1D failed: len=%d err=%v", len(inter1D), err)
	}

	// 5. Empty shape and invalid dimension validation
	if _, err := e.Forward([]int64{1}, []int64{}); err == nil {
		t.Error("expected error for empty shape in Forward")
	}
	if _, err := e.Forward([]int64{1}, []int64{0}); err == nil {
		t.Error("expected error for zero dimension in Forward")
	}

	// 6. Session run error propagation
	if ps, ok := e.session.(*protoSession); ok {
		ps.runErr = os.ErrClosed
		if _, err := e.Forward(input, shape); err == nil {
			t.Error("expected error when session.Run fails")
		}
		ps.runErr = nil
	}
}

func TestONNXEngine_EdgeCases(t *testing.T) {
	// Empty ONNX file (valid protobuf, empty graph)
	dir := t.TempDir()
	emptyPath := filepath.Join(dir, "empty.onnx")
	if err := os.WriteFile(emptyPath, []byte{}, 0644); err != nil {
		t.Fatal(err)
	}
	engEmpty := NewONNXEngine(emptyPath, ModelConfig{HiddenDim: 4})
	if err := engEmpty.Load(); err == nil {
		t.Error("expected error for empty ONNX graph")
	}

	// Valid model with HiddenDim = 0 (invalid output tensor size)
	modelBytes, err := createDummyONNXModel(4)
	if err != nil {
		t.Fatal(err)
	}
	validPath := filepath.Join(dir, "valid.onnx")
	if err := os.WriteFile(validPath, modelBytes, 0644); err != nil {
		t.Fatal(err)
	}
	engZeroDim := NewONNXEngine(validPath, ModelConfig{HiddenDim: 0})
	if err := engZeroDim.Load(); err != nil {
		t.Fatalf("Load failed: %v", err)
	}
	defer func() { _ = engZeroDim.Close() }()

	if _, err := engZeroDim.Forward([]int64{1}, []int64{1}); err == nil {
		t.Error("expected error for zero HiddenDim in Forward")
	}
	if _, err := engZeroDim.GetIntermediate("hidden_states", []int64{1}, []int64{1}); err == nil {
		t.Error("expected error for zero HiddenDim in GetIntermediate")
	}

	// newProtoSession with nonexistent file
	if _, err := newProtoSession("/nonexistent/file.onnx", []string{"hidden_states"}); err == nil {
		t.Error("expected error from newProtoSession on missing file")
	}
}
