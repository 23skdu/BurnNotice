package converter

import (
	"bytes"
	"encoding/binary"
	"math"
	"os"
	"path/filepath"
	"strings"
	"testing"

	"burnnotice/internal/onnxpb"
	"google.golang.org/protobuf/proto"
)

func TestDTypeToONNX(t *testing.T) {
	cases := []struct {
		dtype string
		want  int32
	}{
		{"F32", 1}, {"float32", 1},
		{"F64", 11}, {"float64", 11},
		{"I32", 6}, {"int32", 6},
		{"I64", 7}, {"int64", 7},
		{"F16", 10}, {"float16", 10},
		{"BF16", 16}, {"bfloat16", 16},
		{"unknown", 1}, {"", 1},
	}
	for _, tc := range cases {
		if got := dtypeToONNX(tc.dtype); got != tc.want {
			t.Errorf("dtypeToONNX(%q) = %d, want %d", tc.dtype, got, tc.want)
		}
	}
}

func TestSerializeONNXModel_Success(t *testing.T) {
	tensors := []Tensor{
		{Name: "w0", DType: "F32", Shape: []uint64{2, 3}, Data: make([]byte, 24)},
		{Name: "w1", DType: "I64", Shape: []uint64{4}, Data: make([]byte, 32)},
	}
	data, err := serializeONNXModel(tensors, Options{OpsetVersion: 13})
	if err != nil {
		t.Fatalf("serializeONNXModel error: %v", err)
	}
	var model onnxpb.ModelProto
	if err := proto.Unmarshal(data, &model); err != nil {
		t.Fatalf("unmarshal error: %v", err)
	}
	if len(model.Graph.Initializer) != 2 {
		t.Fatalf("initializers = %d, want 2", len(model.Graph.Initializer))
	}
	if model.OpsetImport[0].GetVersion() != 13 {
		t.Errorf("opset = %d, want 13", model.OpsetImport[0].GetVersion())
	}
}

func TestSerializeONNXModel_DimensionOverflow(t *testing.T) {
	tensors := []Tensor{
		{Name: "huge", DType: "F32", Shape: []uint64{math.MaxInt64 + 1}, Data: nil},
	}
	_, err := serializeONNXModel(tensors, Options{})
	if err == nil || !strings.Contains(err.Error(), "exceeds int64 range") {
		t.Fatalf("expected int64 range error, got %v", err)
	}
}

func TestBuildONNXModel_WriteError(t *testing.T) {
	dir := t.TempDir()
	c := NewONNXConverter()
	// outputPath is an existing directory: WriteFile must fail.
	err := c.buildONNXModel([]Tensor{{Name: "w", DType: "F32", Shape: []uint64{1}, Data: make([]byte, 4)}}, dir, Options{})
	if err == nil || !strings.Contains(err.Error(), "write ONNX file") {
		t.Fatalf("expected write error, got %v", err)
	}
}

func TestONNXConverter_Convert_MkdirError(t *testing.T) {
	dir := t.TempDir()
	blocker := filepath.Join(dir, "blocker")
	if err := os.WriteFile(blocker, []byte("x"), 0600); err != nil {
		t.Fatal(err)
	}
	c := NewONNXConverter()
	err := c.Convert(dir, filepath.Join(blocker, "sub"), Options{})
	if err == nil || !strings.Contains(err.Error(), "create output dir") {
		t.Fatalf("expected create output dir error, got %v", err)
	}
}

func TestONNXConverter_Convert_ParseError(t *testing.T) {
	dir := t.TempDir()
	if err := os.WriteFile(filepath.Join(dir, "bad.safetensors"), []byte{1, 2, 3}, 0600); err != nil {
		t.Fatal(err)
	}
	c := NewONNXConverter()
	err := c.Convert(dir, filepath.Join(dir, "out"), Options{})
	if err == nil || !strings.Contains(err.Error(), "parse") {
		t.Fatalf("expected parse error, got %v", err)
	}
}

func TestValidateONNX_ReadError(t *testing.T) {
	dir := t.TempDir()
	// Stat succeeds on a directory but ReadFile fails.
	err := ValidateONNX(dir)
	if err == nil || !strings.Contains(err.Error(), "read ONNX file") {
		t.Fatalf("expected read error, got %v", err)
	}
}

func TestValidateONNX_InvalidProtobuf(t *testing.T) {
	path := filepath.Join(t.TempDir(), "bad.onnx")
	if err := os.WriteFile(path, []byte{0xff, 0xfe, 0xfd}, 0600); err != nil {
		t.Fatal(err)
	}
	err := ValidateONNX(path)
	if err == nil || !strings.Contains(err.Error(), "failed to unmarshal") {
		t.Fatalf("expected unmarshal error, got %v", err)
	}
}

func TestValidateONNX_NoGraph(t *testing.T) {
	path := filepath.Join(t.TempDir(), "nograph.onnx")
	data, err := proto.Marshal(&onnxpb.ModelProto{IrVersion: proto.Int64(8)})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	err = ValidateONNX(path)
	if err == nil || !strings.Contains(err.Error(), "no graph") {
		t.Fatalf("expected no-graph error, got %v", err)
	}
}

func TestValidateONNX_Success(t *testing.T) {
	path := filepath.Join(t.TempDir(), "ok.onnx")
	data, err := proto.Marshal(&onnxpb.ModelProto{IrVersion: proto.Int64(8), Graph: &onnxpb.GraphProto{Name: proto.String("g")}})
	if err != nil {
		t.Fatal(err)
	}
	if err := os.WriteFile(path, data, 0600); err != nil {
		t.Fatal(err)
	}
	if err := ValidateONNX(path); err != nil {
		t.Fatalf("ValidateONNX error: %v", err)
	}
}

func TestReadUint64LE_ShortReader(t *testing.T) {
	var val uint64
	err := readUint64LE(bytes.NewReader([]byte{1, 2, 3}), &val)
	if err == nil {
		t.Fatal("expected error for short reader")
	}
}

func TestReadUint64LE_Success(t *testing.T) {
	var val uint64
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 42)
	if err := readUint64LE(bytes.NewReader(buf), &val); err != nil {
		t.Fatalf("readUint64LE error: %v", err)
	}
	if val != 42 {
		t.Errorf("val = %d, want 42", val)
	}
}

func TestParseSafetensors_TruncatedHeaderSize(t *testing.T) {
	path := filepath.Join(t.TempDir(), "short.safetensors")
	if err := os.WriteFile(path, []byte{1, 2, 3}, 0600); err != nil {
		t.Fatal(err)
	}
	_, err := ParseSafetensors(path)
	if err == nil || !strings.Contains(err.Error(), "read header size") {
		t.Fatalf("expected header size error, got %v", err)
	}
}

func TestParseSafetensors_TruncatedHeader(t *testing.T) {
	path := filepath.Join(t.TempDir(), "trunc.safetensors")
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, 1024) // header larger than the file
	if err := os.WriteFile(path, buf, 0600); err != nil {
		t.Fatal(err)
	}
	_, err := ParseSafetensors(path)
	if err == nil || !strings.Contains(err.Error(), "read header") {
		t.Fatalf("expected header read error, got %v", err)
	}
}

func TestParseSafetensors_InvalidHeaderJSON(t *testing.T) {
	path := filepath.Join(t.TempDir(), "badjson.safetensors")
	header := []byte("not-json")
	buf := make([]byte, 8)
	binary.LittleEndian.PutUint64(buf, uint64(len(header)))
	buf = append(buf, header...)
	if err := os.WriteFile(path, buf, 0600); err != nil {
		t.Fatal(err)
	}
	_, err := ParseSafetensors(path)
	if err == nil || !strings.Contains(err.Error(), "parse header") {
		t.Fatalf("expected parse header error, got %v", err)
	}
}

func TestParseSafetensors_TensorDataTruncated(t *testing.T) {
	dir := t.TempDir()
	tensors := map[string]TensorInfo{
		"w": {DType: "F32", Shape: []uint64{4}, DataOffsets: []uint64{0, 16}},
	}
	path := createSafetensorsFile(t, dir, "model.safetensors", tensors)
	// Truncate the tensor data section.
	if err := os.Truncate(path, 8+64); err != nil {
		t.Fatal(err)
	}
	_, err := ParseSafetensors(path)
	if err == nil || !strings.Contains(err.Error(), "read tensor") {
		t.Fatalf("expected tensor read error, got %v", err)
	}
}

func TestCountTensors_Error(t *testing.T) {
	_, err := CountTensors([]string{filepath.Join(t.TempDir(), "missing.safetensors")})
	if err == nil {
		t.Fatal("expected error for missing file")
	}
}
