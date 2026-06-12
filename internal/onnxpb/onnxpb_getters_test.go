package onnxpb

import (
	"testing"
)

func TestONNXPB_Getters(t *testing.T) {
	// ModelProto
	m := &ModelProto{
		IrVersion:       ptr(int64(1)),
		ProducerName:    ptr("name"),
		ProducerVersion: ptr("1.0"),
		Domain:          ptr("domain"),
		ModelVersion:    ptr(int64(2)),
		DocString:       ptr("doc"),
		Graph:           &GraphProto{Name: ptr("graph")},
	}
	if m.GetIrVersion() != 1 {
		t.Errorf("GetIrVersion = %d, want 1", m.GetIrVersion())
	}
	if m.GetProducerName() != "name" {
		t.Errorf("GetProducerName = %q, want name", m.GetProducerName())
	}
	if m.GetProducerVersion() != "1.0" {
		t.Errorf("GetProducerVersion = %q, want 1.0", m.GetProducerVersion())
	}
	if m.GetDomain() != "domain" {
		t.Errorf("GetDomain = %q, want domain", m.GetDomain())
	}
	if m.GetModelVersion() != 2 {
		t.Errorf("GetModelVersion = %d, want 2", m.GetModelVersion())
	}
	if m.GetDocString() != "doc" {
		t.Errorf("GetDocString = %q, want doc", m.GetDocString())
	}
	if m.GetGraph() == nil || m.GetGraph().GetName() != "graph" {
		t.Error("failed to GetGraph")
	}
	if len(m.GetOpsetImport()) != 0 {
		t.Error("ops set import should be empty")
	}
	if len(m.GetMetadataProps()) != 0 {
		t.Error("metadata props should be empty")
	}
	if len(m.GetTrainingInfo()) != 0 {
		t.Error("training info should be empty")
	}

	// GraphProto
	g := &GraphProto{
		Name:      ptr("test-graph"),
		DocString: ptr("graph doc"),
	}
	if g.GetName() != "test-graph" {
		t.Errorf("GetName = %q, want test-graph", g.GetName())
	}
	if g.GetDocString() != "graph doc" {
		t.Errorf("GetDocString = %q, want graph doc", g.GetDocString())
	}
	if len(g.GetNode()) != 0 {
		t.Error("nodes should be empty")
	}
	if len(g.GetInitializer()) != 0 {
		t.Error("initializer should be empty")
	}
	if len(g.GetSparseInitializer()) != 0 {
		t.Error("sparse initializer should be empty")
	}
	if len(g.GetInput()) != 0 {
		t.Error("input should be empty")
	}
	if len(g.GetOutput()) != 0 {
		t.Error("output should be empty")
	}
	if len(g.GetValueInfo()) != 0 {
		t.Error("value info should be empty")
	}
	if g.GetQuantizationAnnotation() != nil {
		t.Error("quantization annotation should be nil")
	}

	// NodeProto
	node := &NodeProto{
		Name:      ptr("node1"),
		OpType:    ptr("Add"),
		Domain:    ptr("ai.onnx"),
		DocString: ptr("add node"),
	}
	if node.GetName() != "node1" {
		t.Errorf("GetName = %q, want node1", node.GetName())
	}
	if node.GetOpType() != "Add" {
		t.Errorf("GetOpType = %q, want Add", node.GetOpType())
	}
	if node.GetDomain() != "ai.onnx" {
		t.Errorf("GetDomain = %q, want ai.onnx", node.GetDomain())
	}
	if node.GetDocString() != "add node" {
		t.Errorf("GetDocString = %q, want add node", node.GetDocString())
	}
	if len(node.GetInput()) != 0 {
		t.Error("input should be empty")
	}
	if len(node.GetOutput()) != 0 {
		t.Error("output should be empty")
	}
	if len(node.GetAttribute()) != 0 {
		t.Error("attribute should be empty")
	}

	// TensorProto
	tensor := &TensorProto{
		Name:         ptr("weight"),
		DocString:    ptr("tensor doc"),
		DataType:     ptr(int32(1)),
		RawData:      []byte{1, 2, 3, 4},
		FloatData:    []float32{1.0},
		DoubleData:   []float64{1.0},
		Int64Data:    []int64{1},
		Int32Data:    []int32{1},
		StringData:   [][]byte{[]byte("str")},
		Uint64Data:   []uint64{1},
		DataLocation: TensorProto_DataLocation(0).Enum(),
	}
	if tensor.GetName() != "weight" {
		t.Errorf("GetName = %q, want weight", tensor.GetName())
	}
	if tensor.GetDocString() != "tensor doc" {
		t.Errorf("GetDocString = %q, want tensor doc", tensor.GetDocString())
	}
	if tensor.GetDataType() != 1 {
		t.Errorf("GetDataType = %d, want 1", tensor.GetDataType())
	}
	if len(tensor.GetRawData()) != 4 {
		t.Errorf("GetRawData len = %d, want 4", len(tensor.GetRawData()))
	}
	if len(tensor.GetFloatData()) != 1 {
		t.Error("float data should be 1")
	}
	if len(tensor.GetDoubleData()) != 1 {
		t.Error("double data should be 1")
	}
	if len(tensor.GetInt64Data()) != 1 {
		t.Error("int64 data should be 1")
	}
	if len(tensor.GetInt32Data()) != 1 {
		t.Error("int32 data should be 1")
	}
	if len(tensor.GetStringData()) != 1 {
		t.Error("string data should be 1")
	}
	if len(tensor.GetUint64Data()) != 1 {
		t.Error("uint64 data should be 1")
	}
	if tensor.GetDataLocation() != TensorProto_DEFAULT {
		t.Error("data location should be DEFAULT")
	}
	if len(tensor.GetDims()) != 0 {
		t.Error("dims should be empty")
	}
	if tensor.GetSegment() != nil {
		t.Error("segment should be nil")
	}
	if len(tensor.GetExternalData()) != 0 {
		t.Error("external data should be empty")
	}

	// ValueInfoProto
	valInfo := &ValueInfoProto{
		Name:      ptr("val"),
		DocString: ptr("val doc"),
		Type:      &TypeProto{},
	}
	if valInfo.GetName() != "val" {
		t.Errorf("GetName = %q, want val", valInfo.GetName())
	}
	if valInfo.GetDocString() != "val doc" {
		t.Errorf("GetDocString = %q, want val doc", valInfo.GetDocString())
	}
	if valInfo.GetType() == nil {
		t.Error("failed to GetType")
	}

	// AttributeProto
	attr := &AttributeProto{
		Name:         ptr("attr"),
		RefAttrName:  ptr("ref"),
		Type:         AttributeProto_INT.Enum(),
		F:            ptr(float32(1.5)),
		I:            ptr(int64(42)),
		S:            []byte("string"),
		T:            &TensorProto{Name: ptr("t")},
		G:            &GraphProto{Name: ptr("g")},
		SparseTensor: &SparseTensorProto{},
		Tp:           &TypeProto{},
	}
	if attr.GetName() != "attr" {
		t.Errorf("GetName = %q, want attr", attr.GetName())
	}
	if attr.GetRefAttrName() != "ref" {
		t.Errorf("GetRefAttrName = %q, want ref", attr.GetRefAttrName())
	}
	if attr.GetType() != AttributeProto_INT {
		t.Error("type should be INT")
	}
	if attr.GetF() != 1.5 {
		t.Errorf("GetF = %f, want 1.5", attr.GetF())
	}
	if attr.GetI() != 42 {
		t.Errorf("GetI = %d, want 42", attr.GetI())
	}
	if string(attr.GetS()) != "string" {
		t.Errorf("GetS = %q, want string", string(attr.GetS()))
	}
	if attr.GetT() == nil || attr.GetT().GetName() != "t" {
		t.Error("failed to GetT")
	}
	if attr.GetG() == nil || attr.GetG().GetName() != "g" {
		t.Error("failed to GetG")
	}
	if attr.GetSparseTensor() == nil {
		t.Error("failed to GetSparseTensor")
	}
	if attr.GetTp() == nil {
		t.Error("failed to GetTp")
	}

	// TypeProto
	tpTensor := &TypeProto{
		Value: &TypeProto_TensorType{TensorType: &TypeProto_Tensor{}},
	}
	tpSequence := &TypeProto{
		Value: &TypeProto_SequenceType{SequenceType: &TypeProto_Sequence{}},
	}
	tpMap := &TypeProto{
		Value: &TypeProto_MapType{MapType: &TypeProto_Map{}},
	}
	tpOptional := &TypeProto{
		Value: &TypeProto_OptionalType{OptionalType: &TypeProto_Optional{}},
	}
	tpSparse := &TypeProto{
		Value: &TypeProto_SparseTensorType{SparseTensorType: &TypeProto_SparseTensor{}},
	}
	if tpTensor.GetTensorType() == nil {
		t.Error("failed to GetTensorType")
	}
	if tpSequence.GetSequenceType() == nil {
		t.Error("failed to GetSequenceType")
	}
	if tpMap.GetMapType() == nil {
		t.Error("failed to GetMapType")
	}
	if tpOptional.GetOptionalType() == nil {
		t.Error("failed to GetOptionalType")
	}
	if tpSparse.GetSparseTensorType() == nil {
		t.Error("failed to GetSparseTensorType")
	}

	// TensorShapeProto
	tspVal := &TensorShapeProto{
		Dim: []*TensorShapeProto_Dimension{
			{
				Value:      &TensorShapeProto_Dimension_DimValue{DimValue: 8},
				Denotation: ptr("denot"),
			},
		},
	}
	tspParam := &TensorShapeProto{
		Dim: []*TensorShapeProto_Dimension{
			{
				Value:      &TensorShapeProto_Dimension_DimParam{DimParam: "batch"},
				Denotation: ptr("denot"),
			},
		},
	}

	if len(tspVal.GetDim()) != 1 {
		t.Fatal("failed to GetDim")
	}
	dimVal := tspVal.GetDim()[0]
	if dimVal.GetDimValue() != 8 {
		t.Errorf("GetDimValue = %d, want 8", dimVal.GetDimValue())
	}
	if dimVal.GetDenotation() != "denot" {
		t.Errorf("GetDenotation = %q, want denot", dimVal.GetDenotation())
	}

	dimParam := tspParam.GetDim()[0]
	if dimParam.GetDimParam() != "batch" {
		t.Errorf("GetDimParam = %q, want batch", dimParam.GetDimParam())
	}

	// OperatorSetIdProto
	opset := &OperatorSetIdProto{
		Domain:  ptr("ai.onnx"),
		Version: ptr(int64(17)),
	}
	if opset.GetDomain() != "ai.onnx" {
		t.Errorf("GetDomain = %q, want ai.onnx", opset.GetDomain())
	}
	if opset.GetVersion() != 17 {
		t.Errorf("GetVersion = %d, want 17", opset.GetVersion())
	}
}

func ptr[T any](v T) *T {
	return &v
}
