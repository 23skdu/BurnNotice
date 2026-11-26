//go:build !cgo

package engine

// newORTSession is unavailable without cgo; createSession falls back to the
// pure-Go ONNX protobuf session.
func newORTSession(modelPath string, inputNames, outputNames []string) (onnxSession, error) {
	return nil, errORTUnavailable
}
