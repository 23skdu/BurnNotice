package wasm

import (
	"math"
	"testing"
)

func FuzzEncodeDecodeF32(f *testing.F) {
	f.Add(float32(0.0), float32(1.5), float32(-42.25), float32(1e-6))
	f.Add(float32(-1.0), float32(3.14159), float32(1000.0), float32(-999.5))

	f.Fuzz(func(t *testing.T, a, b, c, d float32) {
		input := []float32{a, b, c, d}
		encoded := encodeF32(input)
		if len(encoded) != len(input)*4 {
			t.Fatalf("len(encoded) = %d, want %d", len(encoded), len(input)*4)
		}
		decoded := decodeF32(encoded)
		if len(decoded) != len(input) {
			t.Fatalf("len(decoded) = %d, want %d", len(decoded), len(input))
		}
		for i := range input {
			if math.Float32bits(decoded[i]) != math.Float32bits(input[i]) {
				t.Errorf("mismatch at %d: got %v, want %v", i, decoded[i], input[i])
			}
		}
	})
}
