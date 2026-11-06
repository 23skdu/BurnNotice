package metrics

import "testing"

func TestAdaptiveSampler(t *testing.T) {
	s := NewAdaptiveSampler(1)
	if !s.ShouldSample() {
		t.Error("expected sample on n=1")
	}

	s.SetSampleRate(3)
	// count=0 (initially), ShouldSample calls Add(1) -> returns 1 -> 1%3 != 0 (false)
	// next -> 2%3 != 0 (false)
	// next -> 3%3 == 0 (true)

	if s.ShouldSample() {
		t.Error("expected no sample on 1st call for n=3")
	}
	if s.ShouldSample() {
		t.Error("expected no sample on 2nd call for n=3")
	}
	if !s.ShouldSample() {
		t.Error("expected sample on 3rd call for n=3")
	}

	if s.Weight() != 3.0 {
		t.Errorf("expected weight 3.0, got %f", s.Weight())
	}

	s.SetSampleRate(0) // should fallback to 1
	if !s.ShouldSample() {
		t.Error("expected sample on n=0 -> n=1")
	}
}
