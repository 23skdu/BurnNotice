package onnxpb

import (
	"google.golang.org/protobuf/proto"
	"testing"
)

func TestONNXPB_MarshalUnmarshal(t *testing.T) {
	model := &ModelProto{
		IrVersion: proto.Int64(8),
		Graph: &GraphProto{
			Name: proto.String("test"),
		},
	}

	data, err := proto.Marshal(model)
	if err != nil {
		t.Fatalf("Marshal failed: %v", err)
	}

	var parsed ModelProto
	if err := proto.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("Unmarshal failed: %v", err)
	}

	if parsed.GetIrVersion() != 8 {
		t.Errorf("IrVersion = %d, want 8", parsed.GetIrVersion())
	}
	if parsed.GetGraph().GetName() != "test" {
		t.Errorf("Graph.Name = %q, want test", parsed.GetGraph().GetName())
	}
}
