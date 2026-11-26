//go:build cgo

package engine

import (
	"fmt"
	"os"
	"sync"

	"github.com/yalue/onnxruntime_go"
)

var (
	onnxEnvInit sync.Once
	onnxEnvErr  error
)

type ortSession struct {
	s *onnxruntime_go.DynamicAdvancedSession
}

func (o *ortSession) Run(input []int64, inShape []int64, output []float32, outShape []int64) error {
	inTensor, err := onnxruntime_go.NewTensor(inShape, input)
	if err != nil {
		return fmt.Errorf("create input tensor: %w", err)
	}
	defer func() { _ = inTensor.Destroy() }()
	outTensor, err := onnxruntime_go.NewTensor(outShape, output)
	if err != nil {
		return fmt.Errorf("create output tensor: %w", err)
	}
	defer func() { _ = outTensor.Destroy() }()
	return o.s.Run([]onnxruntime_go.ArbitraryTensor{inTensor}, []onnxruntime_go.ArbitraryTensor{outTensor})
}

func (o *ortSession) Destroy() error {
	return o.s.Destroy()
}

// newORTSession initializes the ONNX Runtime environment once and creates a
// native inference session. It returns errORTUnavailable when the runtime
// environment cannot be initialized, so callers can fall back to protoSession.
func newORTSession(modelPath string, inputNames, outputNames []string) (onnxSession, error) {
	onnxEnvInit.Do(func() {
		if !onnxruntime_go.IsInitialized() {
			libPath := os.Getenv("ONNX_RUNTIME_LIB")
			if libPath != "" {
				onnxruntime_go.SetSharedLibraryPath(libPath)
			}
			onnxEnvErr = onnxruntime_go.InitializeEnvironment()
		}
	})
	if onnxEnvErr != nil {
		return nil, errORTUnavailable
	}
	session, err := onnxruntime_go.NewDynamicAdvancedSession(
		modelPath,
		inputNames,
		outputNames,
		nil,
	)
	if err != nil {
		return nil, fmt.Errorf("create session: %w", err)
	}
	return &ortSession{s: session}, nil
}
