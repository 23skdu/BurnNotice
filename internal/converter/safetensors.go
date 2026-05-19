package converter

import (
	"bytes"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
)

// SafetensorsHeader describes the tensor metadata in a safetensors file.
type SafetensorsHeader struct {
	Tensors map[string]TensorInfo `json:"-"`
	Raw     map[string]json.RawMessage
}

// TensorInfo holds metadata for a single tensor.
type TensorInfo struct {
	DType    string   `json:"dtype"`
	Shape    []uint64 `json:"shape"`
	DataOffsets []uint64 `json:"data_offsets"`
}

// Tensor represents a loaded tensor with its name and data.
type Tensor struct {
	Name   string
	DType  string
	Shape  []uint64
	Data   []byte
}

// ParseSafetensors reads a .safetensors file and returns all tensors.
func ParseSafetensors(path string) ([]Tensor, error) {
	// #nosec G304 -- CLI tool accepts user-provided model paths
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open safetensors: %w", err)
	}
	defer f.Close()

	// Read header size (first 8 bytes, little-endian uint64)
	var headerSize uint64
	if err := readUint64LE(f, &headerSize); err != nil {
		return nil, fmt.Errorf("read header size: %w", err)
	}

	// Read header JSON
	headerData := make([]byte, headerSize)
	if _, err := f.Read(headerData); err != nil {
		return nil, fmt.Errorf("read header: %w", err)
	}

	// Trim null padding from header
	headerData = bytes.TrimRight(headerData, "\x00")

	header, err := parseHeader(headerData)
	if err != nil {
		return nil, fmt.Errorf("parse header: %w", err)
	}

	// Read tensor data
	var tensors []Tensor
	for name, info := range header.Tensors {
		offset := info.DataOffsets[0]
		length := info.DataOffsets[1] - offset

		data := make([]byte, length)
		readOffset := 8 + headerSize + offset
		if readOffset > uint64(^uint64(0)>>1) {
			return nil, fmt.Errorf("read tensor %s: offset overflow", name)
		}
		// #nosec G115 -- overflow checked above
		if _, err := f.ReadAt(data, int64(readOffset)); err != nil {
			return nil, fmt.Errorf("read tensor %s: %w", name, err)
		}

		tensors = append(tensors, Tensor{
			Name:  name,
			DType: info.DType,
			Shape: info.Shape,
			Data:  data,
		})
	}

	return tensors, nil
}

func parseHeader(data []byte) (*SafetensorsHeader, error) {
	var raw map[string]json.RawMessage
	if err := json.Unmarshal(data, &raw); err != nil {
		return nil, err
	}

	header := &SafetensorsHeader{
		Tensors: make(map[string]TensorInfo),
		Raw:     raw,
	}

	for name, rawInfo := range raw {
		var info TensorInfo
		if err := json.Unmarshal(rawInfo, &info); err != nil {
			continue
		}
		if len(info.DataOffsets) == 2 {
			header.Tensors[name] = info
		}
	}

	return header, nil
}

// FindSafetensorsFiles returns all .safetensors files in a directory.
func FindSafetensorsFiles(dir string) ([]string, error) {
	entries, err := os.ReadDir(dir)
	if err != nil {
		return nil, fmt.Errorf("read dir: %w", err)
	}

	var files []string
	for _, e := range entries {
		if !e.IsDir() && filepath.Ext(e.Name()) == ".safetensors" {
			files = append(files, filepath.Join(dir, e.Name()))
		}
	}

	return files, nil
}

// CountTensors returns the number of tensors across all safetensors files.
func CountTensors(files []string) (int, error) {
	total := 0
	for _, f := range files {
		tensors, err := ParseSafetensors(f)
		if err != nil {
			return 0, err
		}
		total += len(tensors)
	}
	return total, nil
}

// GetTensorElementSize returns the byte size per element for a dtype.
func GetTensorElementSize(dtype string) int {
	switch dtype {
	case "F32", "FP32":
		return 4
	case "F16", "FP16":
		return 2
	case "BF16":
		return 2
	case "I32", "INT32":
		return 4
	case "I64", "INT64":
		return 8
	case "I8", "INT8":
		return 1
	case "U8", "UINT8":
		return 1
	case "BOOL":
		return 1
	default:
		return 4
	}
}

// TensorElementCount returns the total number of elements from a shape.
func TensorElementCount(shape []uint64) uint64 {
	count := uint64(1)
	for _, s := range shape {
		count *= s
	}
	return count
}
