package math

import (
	"math"
	"math/rand"
)

type seededRNG struct {
	*rand.Rand
}

func newSeededRNG(seed int64) *seededRNG {
	return &seededRNG{rand.New(rand.NewSource(seed))}
}

func (s *seededRNG) NormFloat64() float64 {
	return s.Rand.NormFloat64()
}

// cholesky returns the lower-triangular Cholesky factor L such that L·Lᵀ = m.
// m must be symmetric positive-definite. The upper triangle is treated as
// garbage. Used only for tests.
func cholesky(m [][]float64, d int) [][]float64 {
	l := make([][]float64, d)
	for i := range l {
		l[i] = make([]float64, d)
	}
	for i := 0; i < d; i++ {
		for j := 0; j <= i; j++ {
			sum := m[i][j]
			for k := 0; k < j; k++ {
				sum -= l[i][k] * l[j][k]
			}
			if i == j {
				if sum <= 0 {
					panic("cholesky: matrix is not positive definite")
				}
				l[i][j] = math.Sqrt(sum)
			} else {
				l[i][j] = sum / l[j][j]
			}
		}
	}
	return l
}
