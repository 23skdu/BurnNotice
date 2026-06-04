package extract

import (
	"math"
	"testing"
)

func TestExtractWhitenedSVD_Shape(t *testing.T) {
	pos := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
		{0, 0, 1, 0},
		{0, 0, 0, 1},
	}
	neg := [][]float64{
		{-1, 0, 0, 0},
		{0, -1, 0, 0},
		{-1, -1, 0, 0},
		{0, 0, -1, 0},
		{0, 0, 0, -1},
	}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 2, -1, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok")
	}
	if len(dirs) != 2 {
		t.Fatalf("len(dirs) = %d, want 2", len(dirs))
	}
	if len(sv) != 2 {
		t.Fatalf("len(sv) = %d, want 2", len(sv))
	}
	for i, d := range dirs {
		if len(d) != 4 {
			t.Errorf("dirs[%d] dim = %d, want 4", i, len(d))
		}
		if n := Norm(d); math.Abs(n-1.0) > 1e-6 {
			t.Errorf("dirs[%d] norm = %f, want 1.0", i, n)
		}
	}
	if sv[0] < sv[1] {
		t.Errorf("singular values not descending: %v", sv)
	}
}

func TestExtractWhitenedSVD_KClamped(t *testing.T) {
	pos := [][]float64{{1, 0}, {0, 1}, {1, 1}}
	neg := [][]float64{{-1, 0}, {0, -1}, {-1, -1}}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 100, 0, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok")
	}
	if len(dirs) != 2 {
		t.Errorf("k=100 should clamp to min(rows, cols)=2, got %d", len(dirs))
	}
	if len(sv) != 2 {
		t.Errorf("k=100 should clamp sv to 2, got %d", len(sv))
	}
}

func TestExtractWhitenedSVD_InvalidInput(t *testing.T) {
	tests := []struct {
		name     string
		positive [][]float64
		negative [][]float64
		k        int
	}{
		{"nil positive", nil, [][]float64{{1}}, 1},
		{"nil negative", [][]float64{{1}}, nil, 1},
		{"empty positive", [][]float64{}, [][]float64{{1}}, 1},
		{"ragged", [][]float64{{1, 2}, {3}}, [][]float64{{1}, {2}}, 1},
		{"k=0", [][]float64{{1}, {2}}, [][]float64{{1}, {2}}, 0},
		{"k=-1", [][]float64{{1}, {2}}, [][]float64{{1}, {2}}, -1},
		{"dim mismatch", [][]float64{{1, 2}}, [][]float64{{3}}, 1},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			dirs, sv, ok := ExtractWhitenedSVD(tt.positive, tt.negative, tt.k, 0, 0)
			if ok {
				t.Errorf("expected !ok, got dirs=%v sv=%v", dirs, sv)
			}
		})
	}
}

func TestExtractWhitenedSVD_RankDeficient(t *testing.T) {
	const d = 30
	const n = 5
	rng := newSeededRNG(0xDEAD)

	pos := make([][]float64, n)
	neg := make([][]float64, n)
	for i := 0; i < n; i++ {
		pos[i] = make([]float64, d)
		neg[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			pos[i][j] = rng.NormFloat64() + 1.0
			neg[i][j] = rng.NormFloat64() - 1.0
		}
	}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 3, -1, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok on n<d input")
	}
	if len(dirs) != 3 {
		t.Fatalf("len(dirs) = %d, want 3", len(dirs))
	}
	for i, dv := range dirs {
		if n := Norm(dv); math.Abs(n-1.0) > 1e-6 {
			t.Errorf("dirs[%d] norm = %f, want 1.0", i, n)
		}
		for j, v := range dv {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("dirs[%d][%d] = %f, want finite (rank-deficient input)", i, j, v)
			}
		}
	}
	for i, s := range sv {
		if math.IsNaN(s) || math.IsInf(s, 0) {
			t.Errorf("sv[%d] = %f, want finite", i, s)
		}
	}
}

func TestExtractWhitenedSVD_RecoversInjectedSubspace(t *testing.T) {
	const d = 8
	const n = 500
	rng := newSeededRNG(0xABCD)

	trueDir := make([]float64, d)
	trueDir[0] = 1.0

	pos := make([][]float64, n)
	neg := make([][]float64, n)
	for i := 0; i < n; i++ {
		pos[i] = make([]float64, d)
		neg[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			pos[i][j] = rng.NormFloat64()
			neg[i][j] = rng.NormFloat64()
		}
	}
	for i := 0; i < n; i++ {
		pos[i][0] += 5.0
		neg[i][0] -= 5.0
	}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 1, -1, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok")
	}
	if len(dirs) != 1 {
		t.Fatalf("len(dirs) = %d, want 1", len(dirs))
	}

	cosine := math.Abs(Dot(dirs[0], trueDir))
	if cosine < 0.95 {
		t.Errorf("top direction should align with true dir; cosine = %f (want ≥ 0.95)", cosine)
	}

	if len(sv) < 2 {
		return
	}
	if sv[0] <= sv[1] {
		t.Errorf("singular values not descending: %v", sv)
	}
}

func TestExtractWhitenedSVD_IllConditionedRecovery(t *testing.T) {
	const d = 6
	const n = 800
	rng := newSeededRNG(0x1234)

	pos := make([][]float64, n)
	neg := make([][]float64, n)
	for i := 0; i < n; i++ {
		pos[i] = make([]float64, d)
		neg[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			scale := 1.0
			if j == 0 {
				scale = 100.0
			}
			pos[i][j] = rng.NormFloat64() * scale
			neg[i][j] = rng.NormFloat64() * scale
		}
	}
	for i := 0; i < n; i++ {
		pos[i][0] += 10.0
		neg[i][0] -= 10.0
	}

	dirs, _, ok := ExtractWhitenedSVD(pos, neg, 1, -1, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok")
	}
	if len(dirs) == 0 {
		t.Fatal("got 0 directions")
	}
	cosine := math.Abs(dirs[0][0])
	if cosine < 0.5 {
		t.Errorf("whitened SVD should still recover the strong first dim under ill-conditioning; |v[0]| = %f", cosine)
	}
}

func TestExtractWhitenedSVD_MultipleDirections(t *testing.T) {
	const d = 10
	const n = 600
	rng := newSeededRNG(0xCAFE)

	pos := make([][]float64, n)
	neg := make([][]float64, n)
	for i := 0; i < n; i++ {
		pos[i] = make([]float64, d)
		neg[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			pos[i][j] = rng.NormFloat64()
			neg[i][j] = rng.NormFloat64()
		}
	}
	for i := 0; i < n; i++ {
		pos[i][0] += 3.0
		neg[i][0] -= 3.0
		pos[i][1] += 3.0
		neg[i][1] -= 3.0
	}

	dirs, sv, ok := ExtractWhitenedSVD(pos, neg, 3, -1, 0)
	if !ok {
		t.Fatal("ExtractWhitenedSVD returned !ok")
	}
	if len(dirs) != 3 {
		t.Fatalf("len(dirs) = %d, want 3", len(dirs))
	}
	dominant1 := math.Abs(dirs[0][0]) + math.Abs(dirs[0][1])
	dominant2 := math.Abs(dirs[1][0]) + math.Abs(dirs[1][1])
	if dominant1 < 0.3 || dominant2 < 0.3 {
		t.Errorf("expected directions to align with the injected 0/1 axes, got magnitudes %f / %f", dominant1, dominant2)
	}
	if sv[0] < sv[1] || sv[1] < sv[2] {
		t.Errorf("singular values not strictly descending: %v", sv)
	}
}
