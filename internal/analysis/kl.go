package analysis

import (
	"errors"
	"fmt"
	"math"
)

// LogSoftmax computes numerically stable log-probabilities from unnormalized logits.
// It uses max-subtraction to avoid overflow:
//
//	log(p_i) = logits[i] - max(logits) - log(sum(exp(logits - max(logits))))
func LogSoftmax(logits []float64) ([]float64, error) {
	if len(logits) == 0 {
		return nil, errors.New("logits slice is empty")
	}

	maxLogit := logits[0]
	for _, v := range logits[1:] {
		if v > maxLogit {
			maxLogit = v
		}
	}

	var sumExp float64
	for _, v := range logits {
		sumExp += math.Exp(v - maxLogit)
	}

	logSumExp := maxLogit + math.Log(sumExp)

	logProbs := make([]float64, len(logits))
	for i, v := range logits {
		logProbs[i] = v - logSumExp
	}
	return logProbs, nil
}

// ComputeKLDivergence calculates forward Kullback-Leibler divergence D_KL(P || Q)
// from two sets of unnormalized logits representing distributions P and Q:
//
//	D_KL(P || Q) = sum_i P(i) * (log P(i) - log Q(i))
//
// Returns 0 if P and Q are identical, and >= 0 in all cases (Gibbs' inequality).
func ComputeKLDivergence(pLogits, qLogits []float64) (float64, error) {
	if len(pLogits) == 0 || len(qLogits) == 0 {
		return 0, errors.New("empty logits input")
	}
	if len(pLogits) != len(qLogits) {
		return 0, fmt.Errorf("logits dimension mismatch: %d vs %d", len(pLogits), len(qLogits))
	}

	logP, err := LogSoftmax(pLogits)
	if err != nil {
		return 0, err
	}
	logQ, err := LogSoftmax(qLogits)
	if err != nil {
		return 0, err
	}

	var kl float64
	for i := range logP {
		p := math.Exp(logP[i])
		if p > 0 {
			kl += p * (logP[i] - logQ[i])
		}
	}

	// Guard against tiny negative values due to floating-point rounding
	if kl < 0 && kl > -1e-12 {
		kl = 0
	}
	return kl, nil
}

// ComputeSymmetricKLDivergence computes the symmetric KL divergence (Jeffreys divergence):
//
//	D_sym(P, Q) = 0.5 * (D_KL(P || Q) + D_KL(Q || P))
func ComputeSymmetricKLDivergence(pLogits, qLogits []float64) (float64, error) {
	klPQ, err := ComputeKLDivergence(pLogits, qLogits)
	if err != nil {
		return 0, err
	}
	klQP, err := ComputeKLDivergence(qLogits, pLogits)
	if err != nil {
		return 0, err
	}
	return 0.5 * (klPQ + klQP), nil
}

// ComputeBatchKLDivergence computes the token-by-token KL divergence across a batch or sequence,
// returning the slice of per-token KL divergences and the mean KL divergence.
func ComputeBatchKLDivergence(pBatch, qBatch [][]float64) ([]float64, float64, error) {
	if len(pBatch) == 0 || len(qBatch) == 0 {
		return nil, 0, errors.New("empty batch input")
	}
	if len(pBatch) != len(qBatch) {
		return nil, 0, fmt.Errorf("batch length mismatch: %d vs %d", len(pBatch), len(qBatch))
	}

	tokenKLs := make([]float64, len(pBatch))
	var sumKL float64
	for i := range pBatch {
		kl, err := ComputeKLDivergence(pBatch[i], qBatch[i])
		if err != nil {
			return nil, 0, fmt.Errorf("token %d: %w", i, err)
		}
		tokenKLs[i] = kl
		sumKL += kl
	}

	meanKL := sumKL / float64(len(pBatch))
	return tokenKLs, meanKL, nil
}
