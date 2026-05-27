package metrics

import (
	"sync/atomic"
)

// AdaptiveSampler determines whether an event should be sampled.
// It uses a simple atomic counter to sample 1 in every N events.
type AdaptiveSampler struct {
	counter   atomic.Uint64
	sampleN   atomic.Uint64
}

// NewAdaptiveSampler creates a new sampler that samples 1 in every n events.
func NewAdaptiveSampler(n uint64) *AdaptiveSampler {
	if n == 0 {
		n = 1
	}
	s := &AdaptiveSampler{}
	s.sampleN.Store(n)
	return s
}

// ShouldSample returns true if the current event should be sampled.
func (s *AdaptiveSampler) ShouldSample() bool {
	n := s.sampleN.Load()
	if n <= 1 {
		return true
	}
	val := s.counter.Add(1)
	return val%n == 0
}

// Weight returns the current sampling weight (N).
func (s *AdaptiveSampler) Weight() float64 {
	return float64(s.sampleN.Load())
}

// SetSampleRate updates the sampling rate to 1 in N.
func (s *AdaptiveSampler) SetSampleRate(n uint64) {
	if n == 0 {
		n = 1
	}
	s.sampleN.Store(n)
}

// DefaultSampler is a globally accessible sampler configured for 1 in 100 events.
var DefaultSampler = NewAdaptiveSampler(100)
