package analysis

import (
	"encoding/binary"
	"math"
	"testing"
)

func decodeFloat64Slice(data []byte, maxLen int) []float64 {
	n := len(data) / 8
	if n > maxLen {
		n = maxLen
	}
	out := make([]float64, n)
	for i := 0; i < n; i++ {
		bits := binary.LittleEndian.Uint64(data[i*8 : (i+1)*8])
		out[i] = math.Float64frombits(bits)
	}
	return out
}

func FuzzProjectLatentSpace(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 16, 64}, uint8(2), false)
	f.Add([]byte{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16}, uint8(2), true)

	f.Fuzz(func(t *testing.T, raw []byte, dimRaw uint8, useBasis bool) {
		dim := int(dimRaw%8) + 2
		vals := decodeFloat64Slice(raw, 64)
		if len(vals) < dim*2 {
			return
		}
		numPts := len(vals) / dim
		points := make([]LatentPoint, numPts)
		for i := 0; i < numPts; i++ {
			cat := CategoryTruthful
			if i%2 == 1 {
				cat = CategoryDeceptive
			}
			points[i] = LatentPoint{
				ID:       "p",
				PromptID: "pr1",
				Token:    "tok",
				Layer:    i,
				Category: cat,
				Vector:   vals[i*dim : (i+1)*dim],
			}
		}

		var basisX, basisY []float64
		if useBasis {
			basisX = make([]float64, dim)
			basisY = make([]float64, dim)
			copy(basisX, vals[:dim])
			copy(basisY, vals[dim:2*dim])
		}

		proj, err := ProjectLatentSpace(points, basisX, basisY)
		if err != nil {
			return
		}
		for _, p := range proj.Points {
			if math.IsNaN(p.ProjectedX) || math.IsInf(p.ProjectedX, 0) || math.IsNaN(p.ProjectedY) || math.IsInf(p.ProjectedY, 0) {
				t.Fatalf("non-finite projected coordinates: (%v, %v)", p.ProjectedX, p.ProjectedY)
			}
		}
	})
}

func FuzzConceptConeAnalyzer(f *testing.F) {
	f.Add([]byte{0, 0, 0, 0, 0, 0, 240, 63, 0, 0, 0, 0, 0, 0, 0, 64, 0, 0, 0, 0, 0, 0, 8, 64, 0, 0, 0, 0, 0, 0, 16, 64}, uint8(2), uint8(2))
	f.Add([]byte{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15}, uint8(3), uint8(2))

	f.Fuzz(func(t *testing.T, raw []byte, numCatsRaw, dimRaw uint8) {
		numCats := int(numCatsRaw%5) + 2
		dim := int(dimRaw%6) + 2
		vals := decodeFloat64Slice(raw, 64)
		if len(vals) < numCats*dim {
			return
		}
		cats := make([]HarmCategory, numCats)
		for c := 0; c < numCats; c++ {
			vec := vals[c*dim : (c+1)*dim]
			cats[c] = HarmCategory{
				Name:        string(rune('A' + c)),
				Activations: [][]float64{vec},
			}
		}
		analyzer := NewConceptConeAnalyzer(cats)
		res := analyzer.Analyze()
		if res == nil {
			return
		}
		for k, sim := range res.CategoryPairwiseCosines {
			if math.IsNaN(sim) || math.IsInf(sim, 0) || sim < -1.0-1e-9 || sim > 1.0+1e-9 {
				t.Fatalf("invalid pairwise cosine for %s: %v", k, sim)
			}
		}
		sims := analyzer.PairwiseSimilarities()
		for i := range sims {
			for j := range sims[i] {
				v := sims[i][j]
				if math.IsNaN(v) || math.IsInf(v, 0) || v < -1.0-1e-9 || v > 1.0+1e-9 {
					t.Fatalf("invalid similarity matrix entry [%d][%d]: %v", i, j, v)
				}
			}
		}
	})
}

func FuzzKL(f *testing.F) {
	f.Add(1.0, 2.0, -1.0, 0.5, 1.5, -0.5)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(500.0, -500.0, 0.0, -500.0, 500.0, 0.0)

	f.Fuzz(func(t *testing.T, p0, p1, p2, q0, q1, q2 float64) {
		p := []float64{p0, p1, p2}
		q := []float64{q0, q1, q2}

		kl, err := ComputeKLDivergence(p, q)
		if err != nil {
			return
		}
		if math.IsNaN(kl) || math.IsInf(kl, -1) || kl < 0 {
			t.Fatalf("invalid KL divergence: %v", kl)
		}

		symKL, err := ComputeSymmetricKLDivergence(p, q)
		if err != nil {
			return
		}
		if math.IsNaN(symKL) || math.IsInf(symKL, -1) || symKL < 0 {
			t.Fatalf("invalid symmetric KL divergence: %v", symKL)
		}
	})
}

func FuzzCKA(f *testing.F) {
	f.Add(1.0, 2.0, -1.0, 0.5, 3.0, -2.0, 0.5, 1.5)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	f.Fuzz(func(t *testing.T, a0, a1, a2, a3, b0, b1, b2, b3 float64) {
		actA := [][]float64{{a0, a1}, {a2, a3}, {-a0, -a1}}
		actB := [][]float64{{b0, b1}, {b2, b3}, {-b0, -b1}}

		val := CKA(actA, actB)
		if math.IsNaN(val) || math.IsInf(val, 0) || val < 0.0 || val > 1.0 {
			t.Fatalf("CKA out of [0, 1] bounds: %v", val)
		}
	})
}

func FuzzAlignmentImprintDetector(f *testing.F) {
	f.Add(1.0, 0.0, 0.5, 0.9, 0.2, 0.4, 0.8, 0.3, 0.2)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)

	f.Fuzz(func(t *testing.T, x0, x1, x2, y0, y1, y2, z0, z1, z2 float64) {
		d := NewAlignmentImprintDetector()
		d.AddLayerDirection(0, []float64{x0, x1, x2})
		d.AddLayerDirection(1, []float64{y0, y1, y2})
		d.AddLayerDirection(2, []float64{z0, z1, z2})

		m := d.Metrics()
		if math.IsNaN(m.EffectiveRank) || math.IsNaN(m.SVDDecayRate) || math.IsNaN(m.CrossLayerCoherence) || math.IsNaN(m.DirectionStability) {
			t.Fatalf("Metrics returned NaN: %+v", m)
		}
		method := d.ClassifyAlignment()
		if method == "" {
			t.Fatal("ClassifyAlignment returned empty method")
		}
	})
}
