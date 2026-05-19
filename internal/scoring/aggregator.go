package scoring

import (
	"math"
	"sort"
)

// TokenScore holds the score for a single token.
type TokenScore struct {
	Token  string
	Scores map[string]float64
}

// ResponseMetrics holds aggregated metrics for a response.
type ResponseMetrics struct {
	MeanScores map[string]float64
	MaxScores  map[string]float64
	AUC        map[string]float64
	TokenCount int
}

// Aggregator aggregates per-token scores into per-response metrics.
type Aggregator struct {
	tokenScores []TokenScore
}

// NewAggregator creates a new score aggregator.
func NewAggregator() *Aggregator {
	return &Aggregator{
		tokenScores: make([]TokenScore, 0),
	}
}

// AddToken adds a token's scores to the aggregator.
func (a *Aggregator) AddToken(token string, scores map[string]float64) {
	scoresCopy := make(map[string]float64, len(scores))
	for k, v := range scores {
		scoresCopy[k] = v
	}
	a.tokenScores = append(a.tokenScores, TokenScore{
		Token:  token,
		Scores: scoresCopy,
	})
}

// Compute computes aggregated metrics for all added tokens.
func (a *Aggregator) Compute() ResponseMetrics {
	metrics := ResponseMetrics{
		MeanScores: make(map[string]float64),
		MaxScores:  make(map[string]float64),
		AUC:        make(map[string]float64),
		TokenCount: len(a.tokenScores),
	}

	if len(a.tokenScores) == 0 {
		return metrics
	}

	metricNames := make(map[string]bool)
	for _, ts := range a.tokenScores {
		for name := range ts.Scores {
			metricNames[name] = true
		}
	}

	for name := range metricNames {
		var sum float64
		maxVal := -math.MaxFloat64
		values := make([]float64, 0, len(a.tokenScores))

		for _, ts := range a.tokenScores {
			v, ok := ts.Scores[name]
			if !ok {
				continue
			}
			sum += v
			values = append(values, v)
			if v > maxVal {
				maxVal = v
			}
		}

		if len(values) > 0 {
			metrics.MeanScores[name] = sum / float64(len(values))
			metrics.MaxScores[name] = maxVal
			metrics.AUC[name] = computeAUC(values)
		}
	}

	return metrics
}

// Reset clears all token scores.
func (a *Aggregator) Reset() {
	a.tokenScores = a.tokenScores[:0]
}

// TokenCount returns the number of tokens added.
func (a *Aggregator) TokenCount() int {
	return len(a.tokenScores)
}

func computeAUC(values []float64) float64 {
	if len(values) < 2 {
		return 0.0
	}

	sorted := make([]float64, len(values))
	copy(sorted, values)
	sort.Float64s(sorted)

	var auc float64
	n := len(sorted)
	for i := 1; i < n; i++ {
		width := sorted[i] - sorted[i-1]
		height := (sorted[i] + sorted[i-1]) / 2.0
		auc += width * height
	}

	return auc / float64(n)
}
