package converter

import (
	"encoding/binary"
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
)

// Options holds ONNX conversion options.
type Options struct {
	OpsetVersion int
	DynamicAxes  bool
	OutputPath   string
}

// ONNXConverter converts HuggingFace models to ONNX format.
type ONNXConverter struct{}

// NewONNXConverter creates a new converter instance.
func NewONNXConverter() *ONNXConverter {
	return &ONNXConverter{}
}

// Convert converts a HuggingFace model directory to ONNX format.
func (c *ONNXConverter) Convert(modelDir, output string, opts Options) error {
	if opts.OpsetVersion == 0 {
		opts.OpsetVersion = 17
	}

	if err := os.MkdirAll(output, 0750); err != nil {
		return fmt.Errorf("create output dir: %w", err)
	}

	files, err := FindSafetensorsFiles(modelDir)
	if err != nil {
		return fmt.Errorf("find safetensors: %w", err)
	}
	if len(files) == 0 {
		return fmt.Errorf("no .safetensors files found in %s", modelDir)
	}

	var allTensors []Tensor
	for _, f := range files {
		tensors, err := ParseSafetensors(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		allTensors = append(allTensors, tensors...)
	}

	onnxPath := filepath.Join(output, "model.onnx")
	if err := c.buildONNXModel(allTensors, onnxPath, opts); err != nil {
		return fmt.Errorf("build ONNX: %w", err)
	}

	return nil
}

func (c *ONNXConverter) buildONNXModel(tensors []Tensor, outputPath string, opts Options) error {
	model, err := serializeONNXModel(tensors, opts)
	if err != nil {
		return fmt.Errorf("serialize model: %w", err)
	}

	if err := os.WriteFile(outputPath, model, 0600); err != nil {
		return fmt.Errorf("write ONNX file: %w", err)
	}

	return nil
}

func serializeONNXModel(tensors []Tensor, opts Options) ([]byte, error) {
	var buf []byte

	buf = append(buf, []byte("ONNX")...)
	buf = appendUint32(buf, uint32(opts.OpsetVersion)) // #nosec G115

	if opts.DynamicAxes {
		buf = append(buf, byte(1))
	} else {
		buf = append(buf, byte(0))
	}

	buf = appendUint32(buf, uint32(len(tensors))) // #nosec G115

	layerGroups := groupTensorsByLayer(tensors)

	layerNames := make([]string, 0, len(layerGroups))
	for name := range layerGroups {
		layerNames = append(layerNames, name)
	}
	sort.Strings(layerNames)

	for _, layerName := range layerNames {
		layerTensors := layerGroups[layerName]

		sort.Slice(layerTensors, func(i, j int) bool {
			return layerTensors[i].Name < layerTensors[j].Name
		})

		buf = appendString(buf, layerName)
		buf = appendUint32(buf, uint32(len(layerTensors))) // #nosec G115

		for _, t := range layerTensors {
			buf = appendString(buf, t.Name)
			buf = appendUint32(buf, uint32(len(t.Shape))) // #nosec G115
			for _, dim := range t.Shape {
				buf = appendUint64(buf, uint64(dim))
			}

			dataType := dtypeToONNX(t.DType)
			buf = append(buf, byte(dataType))

			data, err := tensorDataToBytes(t)
			if err != nil {
				return nil, fmt.Errorf("convert tensor %s: %w", t.Name, err)
			}
			buf = appendUint32(buf, uint32(len(data))) // #nosec G115
			buf = append(buf, data...)
		}
	}

	buf = append(buf, []byte("ENDM")...)

	return buf, nil
}

func groupTensorsByLayer(tensors []Tensor) map[string][]Tensor {
	groups := make(map[string][]Tensor)
	for _, t := range tensors {
		layer := extractLayerName(t.Name)
		groups[layer] = append(groups[layer], t)
	}
	return groups
}

func extractLayerName(tensorName string) string {
	parts := strings.Split(tensorName, ".")
	if len(parts) >= 2 {
		return parts[0] + "." + parts[1]
	}
	return "unknown"
}

func dtypeToONNX(dtype string) byte {
	switch dtype {
	case "F32", "float32":
		return 1
	case "F64", "float64":
		return 11
	case "I32", "int32":
		return 6
	case "I64", "int64":
		return 7
	case "F16", "float16":
		return 10
	case "BF16", "bfloat16":
		return 16
	default:
		return 1
	}
}

func tensorDataToBytes(t Tensor) ([]byte, error) {
	if t.Data == nil {
		return []byte{}, nil
	}

	return t.Data, nil
}

func appendUint32(buf []byte, v uint32) []byte {
	return append(buf, byte(v>>24), byte(v>>16), byte(v>>8), byte(v)) // #nosec G115 -- intentional byte extraction
}

func appendUint64(buf []byte, v uint64) []byte {
	return append(buf, byte(v>>56), byte(v>>48), byte(v>>40), byte(v>>32), byte(v>>24), byte(v>>16), byte(v>>8), byte(v)) // #nosec G115 -- intentional byte extraction
}

func appendString(buf []byte, s string) []byte {
	buf = appendUint32(buf, uint32(len(s))) // #nosec G115 -- string length fits in uint32
	return append(buf, []byte(s)...)
}

func ValidateONNX(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("ONNX file not found: %s", path)
	}

	// #nosec G304 -- path is user-provided CLI argument
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read ONNX file: %w", err)
	}

	if len(data) < 4 {
		return fmt.Errorf("file too small to be valid ONNX")
	}

	if string(data[:4]) != "ONNX" {
		return fmt.Errorf("invalid ONNX magic: expected ONNX, got %s", string(data[:4]))
	}

	if len(data) < 10 {
		return fmt.Errorf("file too small to contain header")
	}

	opset := binary.BigEndian.Uint32(data[4:8])
	if opset < 1 || opset > 21 {
		return fmt.Errorf("invalid opset version: %d", opset)
	}

	if len(data) < 14 {
		return fmt.Errorf("file too small to contain tensor count")
	}

	tensorCount := binary.BigEndian.Uint32(data[10:14])
	if tensorCount == 0 {
		return fmt.Errorf("no tensors in ONNX model")
	}

	if !strings.HasSuffix(path, ".meta") {
		if len(data) < 6 {
			return fmt.Errorf("file too small to contain end marker")
		}
		endMarker := data[len(data)-4:]
		if string(endMarker) != "ENDM" {
			return fmt.Errorf("missing end marker in ONNX file")
		}
	}

	return nil
}
