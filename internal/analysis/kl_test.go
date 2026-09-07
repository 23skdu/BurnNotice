package analysis

import (
	"math"
	"testing"
)

func TestLogSoftmax(t *testing.T) {
	// 1. Normal values
	logits := []float64{1.0, 2.0, 3.0}
	logProbs, err := LogSoftmax(logits)
	if err != nil {
		t.Fatalf("LogSoftmax failed: %v", err)
	}

	// Sum of exp(logProbs) should be 1.0
	var sumProb float64
	for _, lp := range logProbs {
		sumProb += math.Exp(lp)
	}
	if math.Abs(sumProb-1.0) > 1e-6 {
		t.Errorf("sum(probs) = %f, want 1.0", sumProb)
	}

	// 2. Large numbers (overflow resistance)
	hugeLogits := []float64{1000.0, 1001.0, 1002.0}
	hugeLogProbs, err := LogSoftmax(hugeLogits)
	if err != nil {
		t.Fatalf("LogSoftmax with huge numbers failed: %v", err)
	}
	for i, lp := range hugeLogProbs {
		if math.IsNaN(lp) || math.IsInf(lp, 0) {
			t.Errorf("hugeLogProbs[%d] is NaN or Inf", i)
		}
	}

	// 3. Empty input
	_, err = LogSoftmax(nil)
	if err == nil {
		t.Error("expected error for empty logits")
	}
}

func TestComputeKLDivergence_Identical(t *testing.T) {
	logits := []float64{0.5, 1.2, -0.3, 2.8}
	kl, err := ComputeKLDivergence(logits, logits)
	if err != nil {
		t.Fatalf("ComputeKLDivergence failed: %v", err)
	}
	if kl != 0 {
		t.Errorf("KL between identical distributions = %e, want 0", kl)
	}
}

func TestComputeKLDivergence_KnownAnalytical(t *testing.T) {
	// Let P be uniform over 2 classes: logits [0, 0] -> p = [0.5, 0.5]
	// Let Q have logits [ln(0.8), ln(0.2)] -> q = [0.8, 0.2]
	pLogits := []float64{0.0, 0.0}
	qLogits := []float64{math.Log(0.8), math.Log(0.2)}

	kl, err := ComputeKLDivergence(pLogits, qLogits)
	if err != nil {
		t.Fatalf("ComputeKLDivergence failed: %v", err)
	}

	// Expected: 0.5 * ln(0.5/0.8) + 0.5 * ln(0.5/0.2) = 0.5 * ln(1.5625) ≈ 0.22314355
	expected := 0.5 * math.Log(1.5625)
	if math.Abs(kl-expected) > 1e-6 {
		t.Errorf("KL = %f, want %f", kl, expected)
	}
}

func TestComputeSymmetricKLDivergence(t *testing.T) {
	p := []float64{1.0, 0.0, -1.0}
	q := []float64{0.5, 0.5, 0.5}

	sym1, err := ComputeSymmetricKLDivergence(p, q)
	if err != nil {
		t.Fatalf("sym1 failed: %v", err)
	}
	sym2, err := ComputeSymmetricKLDivergence(q, p)
	if err != nil {
		t.Fatalf("sym2 failed: %v", err)
	}

	if math.Abs(sym1-sym2) > 1e-12 {
		t.Errorf("symmetric KL not symmetric: %f vs %f", sym1, sym2)
	}
	if sym1 <= 0 {
		t.Errorf("symmetric KL = %f, want > 0", sym1)
	}
}

func TestComputeBatchKLDivergence(t *testing.T) {
	batchP := [][]float64{
		{1.0, 2.0},
		{0.0, 0.0},
	}
	batchQ := [][]float64{
		{1.0, 2.0}, // identical -> KL = 0
		{1.0, 0.0}, // different -> KL > 0
	}

	tokenKLs, meanKL, err := ComputeBatchKLDivergence(batchP, batchQ)
	if err != nil {
		t.Fatalf("ComputeBatchKLDivergence failed: %v", err)
	}

	if len(tokenKLs) != 2 {
		t.Fatalf("len(tokenKLs) = %d, want 2", len(tokenKLs))
	}
	if tokenKLs[0] != 0 {
		t.Errorf("tokenKLs[0] = %f, want 0", tokenKLs[0])
	}
	if tokenKLs[1] <= 0 {
		t.Errorf("tokenKLs[1] = %f, want > 0", tokenKLs[1])
	}
	if meanKL <= 0 {
		t.Errorf("meanKL = %f, want > 0", meanKL)
	}
}

func TestComputeKLDivergence_Errors(t *testing.T) {
	// Empty inputs
	if _, err := ComputeKLDivergence(nil, []float64{1.0}); err == nil {
		t.Error("expected error for nil pLogits")
	}
	if _, err := ComputeKLDivergence([]float64{1.0}, nil); err == nil {
		t.Error("expected error for nil qLogits")
	}

	// Dimension mismatch
	if _, err := ComputeKLDivergence([]float64{1.0, 2.0}, []float64{1.0}); err == nil {
		t.Error("expected error for dimension mismatch")
	}

	// Batch mismatch
	b1 := [][]float64{{1.0}}
	b2 := [][]float64{{1.0}, {2.0}}
	if _, _, err := ComputeBatchKLDivergence(b1, b2); err == nil {
		t.Error("expected error for batch length mismatch")
	}
}

func TestProjectionKLDivergence_RegularizedVsExact(t *testing.T) {
	// Baseline original unsteered logits
	origLogits := []float64{3.0, 1.5, 0.2, -0.5}

	// Exact projection (alpha = 0 / 100% removal of component): alters logits significantly
	exactLogits := []float64{0.5, 1.5, 0.2, -0.5}

	// Regularized projection (alpha = 0.5 / ridge shrunk): preserves partial energy, reducing distortion
	regLogits := []float64{2.0, 1.5, 0.2, -0.5}

	klExact, err := ComputeKLDivergence(origLogits, exactLogits)
	if err != nil {
		t.Fatalf("unexpected error on exact KL: %v", err)
	}

	klReg, err := ComputeKLDivergence(origLogits, regLogits)
	if err != nil {
		t.Fatalf("unexpected error on regularized KL: %v", err)
	}

	// Regularized projection must have strictly lower KL divergence than exact projection
	if klReg >= klExact {
		t.Errorf("regularized projection should have lower KL divergence than exact projection: reg=%f vs exact=%f", klReg, klExact)
	}
}
