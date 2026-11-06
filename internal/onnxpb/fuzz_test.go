package onnxpb

import (
	"testing"

	"google.golang.org/protobuf/proto"
)

func FuzzModelProtoUnmarshal(f *testing.F) {
	seedModel := &ModelProto{
		IrVersion:    proto.Int64(8),
		ProducerName: proto.String("burnnotice"),
		Graph: &GraphProto{
			Name: proto.String("test_graph"),
			Node: []*NodeProto{
				{
					Name:   proto.String("node_0"),
					OpType: proto.String("MatMul"),
					Input:  []string{"A", "B"},
					Output: []string{"Y"},
				},
			},
		},
	}
	if seedBytes, err := proto.Marshal(seedModel); err == nil {
		f.Add(seedBytes)
	}
	f.Add([]byte{})
	f.Add([]byte{0x08, 0x08, 0x12, 0x04, 0x74, 0x65, 0x73, 0x74})

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 65536 {
			return
		}
		var m ModelProto
		if err := proto.Unmarshal(data, &m); err != nil {
			return
		}
		// Touch getters on successfully unmarshaled proto to ensure no panics
		_ = m.GetIrVersion()
		_ = m.GetProducerName()
		if g := m.GetGraph(); g != nil {
			_ = g.GetName()
			for _, n := range g.GetNode() {
				_ = n.GetOpType()
			}
		}
		_, _ = proto.Marshal(&m)
	})
}
