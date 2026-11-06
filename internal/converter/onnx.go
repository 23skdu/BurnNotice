package converter

import (
	"fmt"
	"os"
	"path/filepath"

	"burnnotice/internal/onnxpb"
	"google.golang.org/protobuf/proto"
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
	model := &onnxpb.ModelProto{
		IrVersion: proto.Int64(8),
		OpsetImport: []*onnxpb.OperatorSetIdProto{
			{
				Domain:  proto.String(""),
				Version: proto.Int64(int64(opts.OpsetVersion)),
			},
		},
		Graph: &onnxpb.GraphProto{
			Name:        proto.String("burnnotice_converted_graph"),
			Initializer: make([]*onnxpb.TensorProto, 0, len(tensors)),
		},
	}

	for _, t := range tensors {
		dataType := dtypeToONNX(t.DType)

		tensorProto := &onnxpb.TensorProto{
			Name:     proto.String(t.Name),
			DataType: proto.Int32(int32(dataType)),
			Dims:     make([]int64, len(t.Shape)),
			RawData:  t.Data,
		}
		for i, d := range t.Shape {
			/* #nosec G115 */
			tensorProto.Dims[i] = int64(d)
		}
		model.Graph.Initializer = append(model.Graph.Initializer, tensorProto)
	}

	return proto.Marshal(model)
}

func dtypeToONNX(dtype string) int32 {
	switch dtype {
	case "F32", "float32":
		return 1 // onnxpb.TensorProto_FLOAT
	case "F64", "float64":
		return 11 // onnxpb.TensorProto_DOUBLE
	case "I32", "int32":
		return 6 // onnxpb.TensorProto_INT32
	case "I64", "int64":
		return 7 // onnxpb.TensorProto_INT64
	case "F16", "float16":
		return 10 // onnxpb.TensorProto_FLOAT16
	case "BF16", "bfloat16":
		return 16 // onnxpb.TensorProto_BFLOAT16
	default:
		return 1
	}
}

func ValidateONNX(path string) error {
	if _, err := os.Stat(path); os.IsNotExist(err) {
		return fmt.Errorf("ONNX file not found: %s", path)
	}

	/* #nosec G304 */
	data, err := os.ReadFile(path)
	if err != nil {
		return fmt.Errorf("read ONNX file: %w", err)
	}

	var model onnxpb.ModelProto
	if err := proto.Unmarshal(data, &model); err != nil {
		return fmt.Errorf("failed to unmarshal ONNX: %w", err)
	}

	if model.Graph == nil {
		return fmt.Errorf("ONNX model has no graph")
	}

	return nil
}
