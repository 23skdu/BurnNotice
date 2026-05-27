package scoring

import (
	"math"
	"math/rand"
	"testing"
)

func BenchmarkScorer_Score(b *testing.B) {
	s := NewScorer(nil)
	s.RegisterDirection("deception", randomVec(4096))
	s.RegisterDirection("truth", randomVec(4096))
	s.RegisterDirection("conflict", randomVec(4096))

	activation := randomVec(4096)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.Score(activation)
	}
}

func BenchmarkScorer_SubspaceScore(b *testing.B) {
	s := NewScorer(nil)
	s.RegisterDirection("deception", randomVec(4096))
	s.RegisterDirection("truth", randomVec(4096))

	basis := [][]float64{
		randomVec(4096),
		randomVec(4096),
	}

	activation := randomVec(4096)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		s.ScoreSubspace(activation, basis)
	}
}

func BenchmarkNormalize(b *testing.B) {
	v := randomVec(4096)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		norm := vecNorm(v)
		result := make([]float64, len(v))
		for j := range v {
			result[j] = v[j] / norm
		}
		_ = result
	}
}

func BenchmarkDot(b *testing.B) {
	a := randomVec(4096)
	b2 := randomVec(4096)

	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		var sum float64
		for j := range a {
			sum += a[j] * b2[j]
		}
		_ = sum
	}
}

func BenchmarkSigmoid(b *testing.B) {
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		_ = 1.0 / (1.0 + math.Exp(-float64(i)))
	}
}

func randomVec(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64()*2 - 1
	}
	return v
}
