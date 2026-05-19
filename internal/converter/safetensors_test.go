package converter

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"os"
	"path/filepath"
	"testing"
)

func createSafetensorsFile(t *testing.T, dir string, name string, tensors map[string]TensorInfo) string {
	t.Helper()

	header, err := json.Marshal(tensors)
	if err != nil {
		t.Fatal(err)
	}

	// Pad header to 8-byte alignment
	padding := (8 - len(header)%8) % 8
	header = append(header, bytes.Repeat([]byte{0}, padding)...)

	// Calculate total data size
	var totalData uint64
	for _, info := range tensors {
		totalData += info.DataOffsets[1] - info.DataOffsets[0]
	}

	// Build file: [8-byte header size][header][padding][data]
	buf := make([]byte, 0, 8+len(header)+int(totalData))

	// Header size (little-endian uint64)
	sizeBuf := make([]byte, 8)
	binary.LittleEndian.PutUint64(sizeBuf, uint64(len(header)))
	buf = append(buf, sizeBuf...)
	buf = append(buf, header...)

	// Tensor data
	for _, info := range tensors {
		length := info.DataOffsets[1] - info.DataOffsets[0]
		buf = append(buf, make([]byte, length)...)
	}

	path := filepath.Join(dir, name)
	if err := os.WriteFile(path, buf, 0644); err != nil {
		t.Fatal(err)
	}
	return path
}

func TestParseSafetensors(t *testing.T) {
	dir := t.TempDir()

	tensors := map[string]TensorInfo{
		"model.embed_tokens.weight": {
			DType:    "F32",
			Shape:    []uint64{32000, 4096},
			DataOffsets: []uint64{0, 32000 * 4096 * 4},
		},
		"model.norm.weight": {
			DType:    "F32",
			Shape:    []uint64{4096},
			DataOffsets: []uint64{32000 * 4096 * 4, 32000*4096*4 + 4096*4},
		},
	}

	path := createSafetensorsFile(t, dir, "model.safetensors", tensors)

	result, err := ParseSafetensors(path)
	if err != nil {
		t.Fatalf("ParseSafetensors error: %v", err)
	}

	if len(result) != 2 {
		t.Fatalf("len(tensors) = %d, want 2", len(result))
	}

	// Check first tensor
	var embedTensor, normTensor *Tensor
	for i := range result {
		if result[i].Name == "model.embed_tokens.weight" {
			embedTensor = &result[i]
		}
		if result[i].Name == "model.norm.weight" {
			normTensor = &result[i]
		}
	}

	if embedTensor == nil {
		t.Fatal("embed_tokens.weight not found")
	}
	if embedTensor.DType != "F32" {
		t.Errorf("embed dtype = %q, want F32", embedTensor.DType)
	}
	if len(embedTensor.Shape) != 2 || embedTensor.Shape[0] != 32000 || embedTensor.Shape[1] != 4096 {
		t.Errorf("embed shape = %v, want [32000 4096]", embedTensor.Shape)
	}

	if normTensor == nil {
		t.Fatal("norm.weight not found")
	}
	if len(normTensor.Shape) != 1 || normTensor.Shape[0] != 4096 {
		t.Errorf("norm shape = %v, want [4096]", normTensor.Shape)
	}
}

func TestParseSafetensors_FileNotFound(t *testing.T) {
	_, err := ParseSafetensors("/nonexistent/model.safetensors")
	if err == nil {
		t.Error("expected error for nonexistent file")
	}
}

func TestParseSafetensors_EmptyTensors(t *testing.T) {
	dir := t.TempDir()
	tensors := map[string]TensorInfo{}
	path := createSafetensorsFile(t, dir, "empty.safetensors", tensors)

	result, err := ParseSafetensors(path)
	if err != nil {
		t.Fatalf("ParseSafetensors error: %v", err)
	}
	if len(result) != 0 {
		t.Errorf("len(tensors) = %d, want 0", len(result))
	}
}

func TestFindSafetensorsFiles(t *testing.T) {
	dir := t.TempDir()

	os.WriteFile(filepath.Join(dir, "model.safetensors"), []byte{}, 0644)
	os.WriteFile(filepath.Join(dir, "config.json"), []byte{}, 0644)
	os.WriteFile(filepath.Join(dir, "tokenizer.model"), []byte{}, 0644)

	files, err := FindSafetensorsFiles(dir)
	if err != nil {
		t.Fatalf("FindSafetensorsFiles error: %v", err)
	}

	if len(files) != 1 {
		t.Errorf("len(files) = %d, want 1", len(files))
	}
	if filepath.Base(files[0]) != "model.safetensors" {
		t.Errorf("file = %q, want model.safetensors", files[0])
	}
}

func TestFindSafetensorsFiles_DirNotFound(t *testing.T) {
	_, err := FindSafetensorsFiles("/nonexistent/dir")
	if err == nil {
		t.Error("expected error for nonexistent directory")
	}
}

func TestFindSafetensorsFiles_NoFiles(t *testing.T) {
	dir := t.TempDir()
	files, err := FindSafetensorsFiles(dir)
	if err != nil {
		t.Fatalf("FindSafetensorsFiles error: %v", err)
	}
	if len(files) != 0 {
		t.Errorf("len(files) = %d, want 0", len(files))
	}
}

func TestCountTensors(t *testing.T) {
	dir := t.TempDir()

	tensors1 := map[string]TensorInfo{
		"tensor1": {DType: "F32", Shape: []uint64{10}, DataOffsets: []uint64{0, 40}},
	}
	tensors2 := map[string]TensorInfo{
		"tensor2": {DType: "F32", Shape: []uint64{20}, DataOffsets: []uint64{0, 80}},
		"tensor3": {DType: "F32", Shape: []uint64{30}, DataOffsets: []uint64{0, 120}},
	}

	createSafetensorsFile(t, dir, "model-00001.safetensors", tensors1)
	createSafetensorsFile(t, dir, "model-00002.safetensors", tensors2)

	files, _ := FindSafetensorsFiles(dir)
	count, err := CountTensors(files)
	if err != nil {
		t.Fatalf("CountTensors error: %v", err)
	}
	if count != 3 {
		t.Errorf("count = %d, want 3", count)
	}
}

func TestGetTensorElementSize(t *testing.T) {
	tests := []struct {
		dtype string
		want  int
	}{
		{"F32", 4},
		{"FP32", 4},
		{"F16", 2},
		{"FP16", 2},
		{"BF16", 2},
		{"I32", 4},
		{"INT32", 4},
		{"I64", 8},
		{"INT64", 8},
		{"I8", 1},
		{"INT8", 1},
		{"U8", 1},
		{"UINT8", 1},
		{"BOOL", 1},
		{"UNKNOWN", 4},
	}

	for _, tt := range tests {
		t.Run(tt.dtype, func(t *testing.T) {
			got := GetTensorElementSize(tt.dtype)
			if got != tt.want {
				t.Errorf("GetTensorElementSize(%q) = %d, want %d", tt.dtype, got, tt.want)
			}
		})
	}
}

func TestTensorElementCount(t *testing.T) {
	tests := []struct {
		shape []uint64
		want  uint64
	}{
		{[]uint64{10}, 10},
		{[]uint64{10, 20}, 200},
		{[]uint64{32000, 4096}, 32000 * 4096},
		{[]uint64{1}, 1},
		{[]uint64{}, 1},
		{[]uint64{2, 3, 4}, 24},
	}

	for _, tt := range tests {
		t.Run("", func(t *testing.T) {
			got := TensorElementCount(tt.shape)
			if got != tt.want {
				t.Errorf("TensorElementCount(%v) = %d, want %d", tt.shape, got, tt.want)
			}
		})
	}
}

func TestONNXConverter_Convert_NoSafetensors(t *testing.T) {
	dir := t.TempDir()
	output := t.TempDir()

	c := NewONNXConverter()
	err := c.Convert(dir, output, Options{})
	if err == nil {
		t.Error("expected error when no safetensors files found")
	}
}

func TestONNXConverter_Convert_Success(t *testing.T) {
	dir := t.TempDir()
	output := t.TempDir()

	tensors := map[string]TensorInfo{
		"embed.weight": {DType: "F32", Shape: []uint64{100, 64}, DataOffsets: []uint64{0, 100 * 64 * 4}},
	}
	createSafetensorsFile(t, dir, "model.safetensors", tensors)

	c := NewONNXConverter()
	err := c.Convert(dir, output, Options{OpsetVersion: 17, DynamicAxes: true})
	if err != nil {
		t.Fatalf("Convert error: %v", err)
	}

	if _, err := os.Stat(filepath.Join(output, "model.onnx.meta")); err != nil {
		t.Error("meta file not created")
	}
}

func TestONNXConverter_Convert_DefaultOpset(t *testing.T) {
	dir := t.TempDir()
	output := t.TempDir()

	tensors := map[string]TensorInfo{
		"embed.weight": {DType: "F32", Shape: []uint64{100, 64}, DataOffsets: []uint64{0, 100 * 64 * 4}},
	}
	createSafetensorsFile(t, dir, "model.safetensors", tensors)

	c := NewONNXConverter()
	err := c.Convert(dir, output, Options{})
	if err != nil {
		t.Fatalf("Convert error: %v", err)
	}
}

func TestValidateONNX_FileNotFound(t *testing.T) {
	err := ValidateONNX("/nonexistent/model.onnx")
	if err == nil {
		t.Error("expected error for nonexistent ONNX file")
	}
}

func TestValidateONNX_Exists(t *testing.T) {
	dir := t.TempDir()
	path := filepath.Join(dir, "model.onnx")
	os.WriteFile(path, []byte("fake onnx"), 0644)

	err := ValidateONNX(path)
	if err != nil {
		t.Errorf("ValidateONNX error: %v", err)
	}
}
