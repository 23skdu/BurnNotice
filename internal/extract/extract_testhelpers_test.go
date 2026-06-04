package extract

import (
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
