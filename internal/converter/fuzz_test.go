package converter

import (
	"bytes"
	"encoding/binary"
	"encoding/json"
	"math"
	"os"
	"path/filepath"
	"testing"
)

func FuzzParseSafetensors(f *testing.F) {
	f.Add([]byte("test-tensor"), "F32", uint64(10), uint64(20))
	f.Add([]byte("weight"), "F16", uint64(100), uint64(200))
	f.Add([]byte("bias"), "I32", uint64(0), uint64(0))

	f.Fuzz(func(t *testing.T, name []byte, dtype string, start, end uint64) {
		if len(name) == 0 || len(name) > 256 {
			t.Skip()
		}
		for _, b := range name {
			if b < 32 || b > 126 {
				t.Skip()
			}
		}
		if start > end || end-start > 10000 {
			t.Skip()
		}
		if dtype == "" {
			t.Skip()
		}

		dir := t.TempDir()
		tensors := map[string]TensorInfo{
			string(name): {
				DType:       dtype,
				Shape:       []uint64{end - start},
				DataOffsets: []uint64{start, end},
			},
		}

		header, err := json.Marshal(tensors)
		if err != nil {
			t.Skip()
		}

		padding := (8 - len(header)%8) % 8
		header = append(header, bytes.Repeat([]byte{0}, padding)...)

		buf := make([]byte, 0, 8+len(header)+int(end))
		sizeBuf := make([]byte, 8)
		binary.LittleEndian.PutUint64(sizeBuf, uint64(len(header)))
		buf = append(buf, sizeBuf...)
		buf = append(buf, header...)
		buf = append(buf, make([]byte, end)...)

		path := filepath.Join(dir, "model.safetensors")
		if err := os.WriteFile(path, buf, 0644); err != nil {
			t.Skip()
		}

		result, err := ParseSafetensors(path)
		if err != nil {
			return
		}

		for _, tensor := range result {
			if tensor.DType == "" {
				t.Error("empty dtype")
			}
			if len(tensor.Shape) == 0 && tensor.Name != "" {
				t.Error("empty shape for named tensor")
			}
		}
	})
}

func FuzzGetTensorElementSize(f *testing.F) {
	dtypes := []string{"F32", "F16", "BF16", "I32", "I64", "I8", "U8", "BOOL", "FP32", "FP16", "INT32", "INT64", "INT8", "UINT8"}
	for _, dt := range dtypes {
		f.Add(dt)
	}

	f.Fuzz(func(t *testing.T, dtype string) {
		if len(dtype) > 32 {
			t.Skip()
		}
		for _, c := range dtype {
			if c < 32 || c > 126 {
				t.Skip()
			}
		}

		size := GetTensorElementSize(dtype)
		if size < 1 || size > 8 {
			t.Errorf("GetTensorElementSize(%q) = %d, want 1-8", dtype, size)
		}
	})
}

func FuzzTensorElementCount(f *testing.F) {
	f.Add(uint64(1))
	f.Add(uint64(100))
	f.Add(uint64(1000))

	f.Fuzz(func(t *testing.T, dim uint64) {
		if dim > 100000 {
			t.Skip()
		}

		count := TensorElementCount([]uint64{dim})
		if count != dim {
			t.Errorf("TensorElementCount([%d]) = %d, want %d", dim, count, dim)
		}

		count2 := TensorElementCount([]uint64{dim, 1})
		if count2 != dim {
			t.Errorf("TensorElementCount([%d, 1]) = %d, want %d", dim, count2, dim)
		}
	})
}

func FuzzParseHeader(f *testing.F) {
	f.Add(`{"tensor":{"dtype":"F32","shape":[10],"data_offsets":[0,40]}}`)
	f.Add(`{}`)
	f.Add(`not json`)

	f.Fuzz(func(t *testing.T, data string) {
		if len(data) > 10000 {
			t.Skip()
		}

		header, err := parseHeader([]byte(data))
		if err != nil {
			return
		}

		if header.Tensors == nil {
			t.Error("Tensors map is nil")
		}
		if header.Raw == nil {
			t.Error("Raw map is nil")
		}
	})
}

func FuzzFindSafetensorsFiles(f *testing.F) {
	f.Add("model.safetensors")
	f.Add("config.json")
	f.Add("model.bin")

	f.Fuzz(func(t *testing.T, filename string) {
		if len(filename) > 255 {
			t.Skip()
		}
		for _, c := range filename {
			if c == '/' || c == '\\' || c == 0 {
				t.Skip()
			}
		}

		dir := t.TempDir()
		path := filepath.Join(dir, filename)
		os.WriteFile(path, []byte{}, 0644)

		files, err := FindSafetensorsFiles(dir)
		if err != nil {
			t.Fatalf("FindSafetensorsFiles error: %v", err)
		}

		isSafetensors := filepath.Ext(filename) == ".safetensors"
		if isSafetensors && len(files) != 1 {
			t.Errorf("expected 1 file for %q, got %d", filename, len(files))
		}
		if !isSafetensors && len(files) != 0 {
			t.Errorf("expected 0 files for %q, got %d", filename, len(files))
		}
	})
}

func FuzzReadUint64LE(f *testing.F) {
	f.Add(uint64(0))
	f.Add(uint64(1))
	f.Add(uint64(18446744073709551615))
	f.Add(uint64(4294967295))

	f.Fuzz(func(t *testing.T, val uint64) {
		buf := make([]byte, 8)
		binary.LittleEndian.PutUint64(buf, val)

		var result uint64
		err := readUint64LE(bytes.NewReader(buf), &result)
		if err != nil {
			t.Fatalf("readUint64LE error: %v", err)
		}
		if result != val {
			t.Errorf("readUint64LE = %d, want %d", result, val)
		}
	})
}

func FuzzONNXConverter(f *testing.F) {
	f.Add("F32", uint64(10), uint64(50))
	f.Add("F16", uint64(0), uint64(100))

	f.Fuzz(func(t *testing.T, dtype string, start, end uint64) {
		if start > end || end-start > 10000 {
			t.Skip()
		}
		if dtype == "" {
			t.Skip()
		}
		if math.IsNaN(float64(start)) || math.IsInf(float64(start), 0) {
			t.Skip()
		}

		dir := t.TempDir()
		output := t.TempDir()

		tensors := map[string]TensorInfo{
			"test.weight": {
				DType:       dtype,
				Shape:       []uint64{end - start},
				DataOffsets: []uint64{start, end},
			},
		}

		header, err := json.Marshal(tensors)
		if err != nil {
			t.Skip()
		}

		padding := (8 - len(header)%8) % 8
		header = append(header, bytes.Repeat([]byte{0}, padding)...)

		buf := make([]byte, 0, 8+len(header)+int(end))
		sizeBuf := make([]byte, 8)
		binary.LittleEndian.PutUint64(sizeBuf, uint64(len(header)))
		buf = append(buf, sizeBuf...)
		buf = append(buf, header...)
		buf = append(buf, make([]byte, end)...)

		path := filepath.Join(dir, "model.safetensors")
		if err := os.WriteFile(path, buf, 0644); err != nil {
			t.Skip()
		}

		c := NewONNXConverter()
		err = c.Convert(dir, output, Options{OpsetVersion: 17, DynamicAxes: true})
		if err != nil {
			return
		}

		if _, err := os.Stat(filepath.Join(output, "model.onnx")); err != nil {
			t.Error("onnx file not created")
		}
	})
}
