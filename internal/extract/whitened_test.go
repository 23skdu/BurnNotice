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

func TestAccuracy_ExtractDIM_And_PCA_NaNInfResilience(t *testing.T) {
	// Inject extreme NaN/+Inf/-Inf outliers into otherwise well-separated activations
	pos := [][]float64{
		{2.0, 0.0, 0.0},
		{2.1, 0.1, 0.0},
		{1.9, -0.1, 0.0},
		{math.Inf(1), math.NaN(), 0.0},
	}
	neg := [][]float64{
		{-2.0, 0.0, 0.0},
		{-2.1, 0.1, 0.0},
		{-1.9, -0.1, 0.0},
		{math.Inf(-1), 0.0, math.NaN()},
	}

	dimDir := ExtractDIM(copyMatrix(pos), copyMatrix(neg))
	if len(dimDir) != 3 {
		t.Fatalf("ExtractDIM len = %d, want 3", len(dimDir))
	}
	for i, v := range dimDir {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			t.Errorf("ExtractDIM[%d] is non-finite: %f", i, v)
		}
	}
	if math.Abs(Norm(dimDir)-1.0) > 1e-6 {
		t.Errorf("ExtractDIM norm = %f, want 1.0", Norm(dimDir))
	}

	pcaDirs, sv := ExtractPCA(copyMatrix(pos), copyMatrix(neg), 2)
	if len(pcaDirs) == 0 {
		t.Fatal("ExtractPCA returned empty directions on Winsorized NaN/Inf inputs")
	}
	for i, d := range pcaDirs {
		for j, v := range d {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("ExtractPCA[%d][%d] is non-finite: %f", i, j, v)
			}
		}
		if math.Abs(Norm(d)-1.0) > 1e-6 {
			t.Errorf("ExtractPCA[%d] norm = %f, want 1.0", i, Norm(d))
		}
	}
	for i, s := range sv {
		if math.IsNaN(s) || math.IsInf(s, 0) || s < 0 {
			t.Errorf("ExtractPCA singularValue[%d] invalid: %f", i, s)
		}
	}
}

func TestInvariant_DegenerateZeroNormAndIdenticalDistributions(t *testing.T) {
	// Zero-norm vectors in Normalize, Norm, Dot
	zero := []float64{0.0, 0.0, 0.0, 0.0}
	normZero := Normalize(zero)
	if Norm(normZero) != 0.0 {
		t.Errorf("Normalize(zero) norm = %f, want 0.0", Norm(normZero))
	}

	// Identical positive and negative matrices (zero difference matrix -> degenerate covariance)
	identical := [][]float64{
		{1.0, 2.0, 3.0},
		{1.0, 2.0, 3.0},
		{1.0, 2.0, 3.0},
	}

	// ExtractDIM on identical sets returns a finite zero-norm vector without NaN
	dimZero := ExtractDIM(copyMatrix(identical), copyMatrix(identical))
	for i, v := range dimZero {
		if math.IsNaN(v) || math.IsInf(v, 0) || v != 0.0 {
			t.Errorf("ExtractDIM(identical)[%d] = %f, want 0.0", i, v)
		}
	}

	// ExtractINLP on identical sets terminates cleanly without NaN or Inf
	inlpDirs := ExtractINLP(copyMatrix(identical), copyMatrix(identical), 2)
	for i, d := range inlpDirs {
		for j, v := range d {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				t.Errorf("ExtractINLP(identical)[%d][%d] is non-finite: %f", i, j, v)
			}
		}
	}

	// ExtractWhitenedSVD on zero-variance identical sets with eigFloor=0 returns ok=false cleanly
	_, _, ok := ExtractWhitenedSVD(copyMatrix(identical), copyMatrix(identical), 1, 0.0, 0.0)
	if ok {
		t.Error("expected ExtractWhitenedSVD to return ok=false on zero-variance data with eigFloor=0")
	}
}

func TestOracle_ExtractDIM_ExactAnalyticalDirection(t *testing.T) {
	pos := [][]float64{
		{4.0, 2.0, -1.0},
		{6.0, 4.0, -1.0},
	}
	neg := [][]float64{
		{2.0, -1.0, -1.0},
		{0.0, -3.0, -1.0},
	}
	// posMean = [5, 3, -1], negMean = [1, -2, -1]
	// diff = [4, 5, 0], norm = sqrt(41)
	got := ExtractDIM(pos, neg)
	norm41 := math.Sqrt(41.0)
	want := []float64{4.0 / norm41, 5.0 / norm41, 0.0}
	for i := range want {
		if math.Abs(got[i]-want[i]) > 1e-12 {
			t.Errorf("Oracle ExtractDIM[%d] = %.15f, want %.15f", i, got[i], want[i])
		}
	}
}
