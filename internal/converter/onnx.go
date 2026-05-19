package converter

import (
	"fmt"
	"os"
	"path/filepath"
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

	// Parse all tensors
	var allTensors []Tensor
	for _, f := range files {
		tensors, err := ParseSafetensors(f)
		if err != nil {
			return fmt.Errorf("parse %s: %w", f, err)
		}
		allTensors = append(allTensors, tensors...)
	}

	// Build ONNX model from tensors
	onnxPath := filepath.Join(output, "model.onnx")
	if err := c.buildONNXModel(allTensors, onnxPath, opts); err != nil {
		return fmt.Errorf("build ONNX: %w", err)
	}

	return nil
}

func (c *ONNXConverter) buildONNXModel(tensors []Tensor, outputPath string, opts Options) error {
	// TODO: Build ONNX protobuf model from tensors:
	// 1. Create ONNX ModelProto with graph
	// 2. Add initializers for each tensor (name, dims, data)
	// 3. Build computation graph (embedding → layers → lm_head)
	// 4. Set opset version and IR version
	// 5. If DynamicAxes, set dim_param for batch and sequence dims
	// 6. Serialize to outputPath

	// Placeholder: write metadata file for now
	meta := fmt.Sprintf("tensors: %d\nopset: %d\ndynamic: %v\n", len(tensors), opts.OpsetVersion, opts.DynamicAxes)
	return os.WriteFile(outputPath+".meta", []byte(meta), 0600)
}

// ValidateONNX checks if an ONNX file is valid and loadable.
func ValidateONNX(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("ONNX file not found: %s", path)
	}

	// TODO: Use onnx-go to:
	// 1. Read file bytes
	// 2. Parse ONNX protobuf
	// 3. Validate graph structure
	// 4. Check tensor shapes and types

	return nil
}
