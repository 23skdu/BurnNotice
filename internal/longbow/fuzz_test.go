package longbow

import (
	"context"
	"math"
	"testing"
)

func FuzzSerializeDeserializeFloats(f *testing.F) {
	f.Add([]byte{0, 0, 128, 63, 0, 0, 0, 64})
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12})
	f.Add([]byte{})

	f.Fuzz(func(t *testing.T, raw []byte) {
		floats := deserializeFloats(raw)
		roundTrip := serializeFloats(floats)
		floats2 := deserializeFloats(roundTrip)
		if len(floats) != len(floats2) {
			t.Fatalf("len mismatch: %d != %d", len(floats), len(floats2))
		}
		for i := range floats {
			if math.IsNaN(float64(floats[i])) && math.IsNaN(float64(floats2[i])) {
				continue
			}
			if floats[i] != floats2[i] {
				t.Fatalf("mismatch at %d: %v != %v", i, floats[i], floats2[i])
			}
		}
	})
}

func FuzzFallbackSearchAndMath(f *testing.F) {
	f.Add(float32(1.0), float32(0.0), float32(0.5), float32(0.5), 2)
	f.Add(float32(0.0), float32(0.0), float32(0.0), float32(0.0), 1)
	f.Add(float32(-1.5), float32(2.5), float32(3.0), float32(-0.5), 5)

	f.Fuzz(func(t *testing.T, a0, a1, b0, b1 float32, k int) {
		if math.IsNaN(float64(a0)) || math.IsNaN(float64(a1)) || math.IsNaN(float64(b0)) || math.IsNaN(float64(b1)) {
			return
		}
		if math.IsInf(float64(a0), 0) || math.IsInf(float64(a1), 0) || math.IsInf(float64(b0), 0) || math.IsInf(float64(b1), 0) {
			return
		}
		if k <= 0 || k > 100 {
			k = 2
		}

		vecA := []float32{a0, a1}
		vecB := []float32{b0, b1}

		_ = cosineSimilarity32(vecA, vecB)
		dist := euclideanDistance32(vecA, vecB)
		if dist < 0 {
			t.Fatalf("negative euclidean distance: %f", dist)
		}

		c := &Client{
			dim:          2,
			metric:       "cosine",
			fallbackMode: 1,
		}
		ctx := context.Background()
		if err := c.Ingest(ctx, [][]float32{vecA, vecB}, []uint64{1, 2}); err != nil {
			t.Fatalf("Ingest error: %v", err)
		}
		ids, dists, err := c.Search(ctx, vecA, k)
		if err != nil {
			t.Fatalf("Search error: %v", err)
		}
		if len(ids) != len(dists) {
			t.Fatalf("ids/dists len mismatch: %d != %d", len(ids), len(dists))
		}
	})
}
