package main

import (
	"strconv"
	"strings"
)

// largeModelThreshold is the parameter count above which whitened SVD
// extraction is the default for the `burnnotice profile` command. This
// matches the size cutoff called out in docs/roadmap.md.
const largeModelThreshold = 7e9

// parseModelSize converts a human-readable model-size string like "0.5B",
// "7B", "13B", "70B", or "175000M" into a raw parameter count. The unit
// suffix is case-insensitive and supports B (billion) and M (million).
// Returns (0, false) if the string cannot be parsed.
func parseModelSize(s string) (float64, bool) {
	s = strings.TrimSpace(s)
	if s == "" {
		return 0, false
	}
	suffix := s[len(s)-1]
	numStr := s[:len(s)-1]
	num, err := strconv.ParseFloat(numStr, 64)
	if err != nil {
		return 0, false
	}
	switch suffix {
	case 'B', 'b':
		return num * 1e9, true
	case 'M', 'm':
		return num * 1e6, true
	default:
		// No unit suffix: treat as raw parameter count.
		raw, err := strconv.ParseFloat(s, 64)
		if err != nil {
			return 0, false
		}
		return raw, true
	}
}

// isLargeModel reports whether the given model-size string represents a model
// above the largeModelThreshold (>7B params). Unparseable inputs are treated
// as small (returns false) so the safer DIM default is chosen.
func isLargeModel(modelSize string) bool {
	params, ok := parseModelSize(modelSize)
	if !ok {
		return false
	}
	return params > largeModelThreshold
}

// shouldUseWhitenedSVD decides whether the profile CLI should default to the
// whitened-SVD extraction method. The decision is:
//
//  1. If the user explicitly set --whitened-svd, honor the flag value.
//  2. Otherwise, default to true for models strictly larger than 7B
//     parameters (per docs/roadmap.md), and false otherwise.
func shouldUseWhitenedSVD(flagChanged, flagValue bool, modelSize string) bool {
	if flagChanged {
		return flagValue
	}
	return isLargeModel(modelSize)
}
