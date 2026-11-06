package main

import (
	"math"
	"testing"
)

func FuzzParseModelSize(f *testing.F) {
	f.Add("0.5B", false, false)
	f.Add("7B", false, false)
	f.Add("13B", false, false)
	f.Add("70b", true, false)
	f.Add("350M", false, true)
	f.Add("1000000000", false, false)
	f.Add("", false, false)
	f.Add("invalid", true, true)

	f.Fuzz(func(t *testing.T, s string, flagChanged, flagValue bool) {
		if len(s) > 128 {
			return
		}
		params, ok := parseModelSize(s)
		if ok && (math.IsNaN(params) || math.IsInf(params, 0)) {
			return
		}

		large := isLargeModel(s)
		if ok && (params > largeModelThreshold) != large {
			t.Errorf("isLargeModel(%q) = %v, want %v (params=%g)", s, large, params > largeModelThreshold, params)
		}

		got := shouldUseWhitenedSVD(flagChanged, flagValue, s)
		if flagChanged && got != flagValue {
			t.Errorf("shouldUseWhitenedSVD(true, %v, %q) = %v, want %v", flagValue, s, got, flagValue)
		}
		if !flagChanged && got != large {
			t.Errorf("shouldUseWhitenedSVD(false, %v, %q) = %v, want %v", flagValue, s, got, large)
		}
	})
}
