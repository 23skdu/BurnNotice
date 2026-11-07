package sae

import (
	"math"
	"testing"
)

func FuzzSAEEncodeDecodeTopK(f *testing.F) {
	f.Add(1.0, -2.0, 0.5, 0.1, 2)
	f.Add(0.0, 0.0, -0.5, 3.0, 1)
	f.Add(-1.5, 2.5, 1.0, -1.0, 5)

	f.Fuzz(func(t *testing.T, x0, x1, b0, b1 float64, k int) {
		for _, v := range []float64{x0, x1, b0, b1} {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
				return
			}
		}
		if k < 0 || k > 16 {
			return
		}

		loader := &SAELoader{}
		err := loader.LoadFromMemory(&SAEWeights{
			EncoderWeights: [][]float64{{1, 0}, {0, 1}},
			DecoderWeights: [][]float64{{1, 0}, {0, 1}},
			EncoderBias:    []float64{b0, b1},
			DecoderBias:    []float64{0, 0},
			FeatureCount:   2,
			InputDim:       2,
		})
		if err != nil {
			t.Fatalf("LoadFromMemory failed: %v", err)
		}

		features, err := loader.Encode([]float64{x0, x1})
		if err != nil {
			t.Fatalf("Encode failed: %v", err)
		}
		for i, feat := range features {
			if feat < 0 {
				t.Errorf("feature[%d] = %f < 0 after ReLU", i, feat)
			}
		}

		_, err = loader.Decode(features)
		if err != nil {
			t.Fatalf("Decode failed: %v", err)
		}

		idxs, vals := loader.TopKFeatures(features, k)
		if len(idxs) != len(vals) {
			t.Errorf("TopKFeatures length mismatch: %d vs %d", len(idxs), len(vals))
		}
		_ = loader.Attribution(features)
	})
}

func FuzzNpyHeaderParser(f *testing.F) {
	f.Add([]byte("\x93NUMPY\x01\x00\x3c\x00{'descr': '<f4', 'fortran_order': False, 'shape': (2, 3), }"))
	f.Add([]byte("invalid"))

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) > 4096 {
			return
		}
		_, _, _ = readNpyHeader(data)
	})
}

func FuzzSAEFeatureAbsorptionAndAlignment(f *testing.F) {
	f.Add(1.0, 0.5, 0.2, 1.5)
	f.Add(2.0, 0.0, 0.5, 2.0)

	f.Fuzz(func(t *testing.T, x0, x1, y0, y1 float64) {
		for _, v := range []float64{x0, x1, y0, y1} {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
				return
			}
		}
		d1 := &SAELoader{}
		_ = d1.LoadFromMemory(&SAEWeights{
			Family:         SAELlamaScope,
			EncoderWeights: [][]float64{{1, 0}, {0, 1}},
			DecoderWeights: [][]float64{{1, 0}, {0, 1}},
			EncoderBias:    []float64{0, 0},
			DecoderBias:    []float64{0, 0},
			FeatureCount:   2,
			InputDim:       2,
		})
		d2 := &SAELoader{}
		_ = d2.LoadFromMemory(&SAEWeights{
			Family:         SAEQwenSAE,
			EncoderWeights: [][]float64{{1, 0}, {0, 1}},
			DecoderWeights: [][]float64{{1, 0}, {0, 1}},
			EncoderBias:    []float64{0, 0},
			DecoderBias:    []float64{0, 0},
			FeatureCount:   2,
			InputDim:       2,
		})
		acts := [][]float64{{x0, x1}, {y0, y1}}
		rep, err := AnalyzeFeatureAbsorption(d1, acts, map[string][]float64{"deception": {1, 0}})
		if err != nil {
			t.Fatalf("AnalyzeFeatureAbsorption failed: %v", err)
		}
		if rep.AbsorptionRate < 0 || rep.AbsorptionRate > 1.0 {
			t.Errorf("AbsorptionRate out of [0,1]: %f", rep.AbsorptionRate)
		}
		alignRes, err := AlignSAEDictionaries(d1, d2, acts, acts, 2)
		if err != nil {
			t.Fatalf("AlignSAEDictionaries failed: %v", err)
		}
		if alignRes.LinearCKA < -1e-9 || alignRes.LinearCKA > 1.0+1e-9 {
			t.Errorf("LinearCKA out of [0,1]: %f", alignRes.LinearCKA)
		}
	})
}
