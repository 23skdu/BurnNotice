package math

// Winsorize caps extreme values in activations at the given quantile thresholds.
// lower and upper are quantiles (e.g. 0.005 and 0.995).
// NaN/Inf inputs are clamped to safe values before winsorization.
func Winsorize(activations []float64, lower, upper float64) []float64 {
	n := len(activations)
	if n == 0 {
		return nil
	}

	clean := ClampToFloat64(activations)

	sorted := make([]float64, n)
	copy(sorted, clean)
	sortFloat64s(sorted)

	loIdx := int(float64(n) * lower)
	if loIdx < 0 {
		loIdx = 0
	}
	if loIdx >= n {
		loIdx = n - 1
	}
	hiIdx := int(float64(n)*upper) - 1
	if hiIdx >= n {
		hiIdx = n - 1
	}
	if hiIdx < loIdx {
		hiIdx = loIdx
	}

	loVal := sorted[loIdx]
	hiVal := sorted[hiIdx]

	out := make([]float64, n)
	for i, v := range clean {
		switch {
		case v < loVal:
			out[i] = loVal
		case v > hiVal:
			out[i] = hiVal
		default:
			out[i] = v
		}
	}
	return out
}

// WinsorizeInPlace modifies the slice in place to avoid allocation.
// NaN/Inf inputs are clamped to safe values before winsorization.
func WinsorizeInPlace(activations []float64, lower, upper float64) {
	n := len(activations)
	if n == 0 {
		return
	}

	clean := ClampToFloat64(activations)

	sorted := make([]float64, n)
	copy(sorted, clean)
	sortFloat64s(sorted)

	loIdx := int(float64(n) * lower)
	if loIdx < 0 {
		loIdx = 0
	}
	if loIdx >= n {
		loIdx = n - 1
	}
	hiIdx := int(float64(n)*upper) - 1
	if hiIdx >= n {
		hiIdx = n - 1
	}
	if hiIdx < loIdx {
		hiIdx = loIdx
	}

	loVal := sorted[loIdx]
	hiVal := sorted[hiIdx]

	for i, v := range clean {
		switch {
		case v < loVal:
			activations[i] = loVal
		case v > hiVal:
			activations[i] = hiVal
		default:
			activations[i] = v
		}
	}
}

// WinsorizeMatrixInPlace winsorizes each column of a matrix independently.
// data is a row-major matrix where each row is a sample and each column is a feature.
func WinsorizeMatrixInPlace(data [][]float64, lower, upper float64) {
	if len(data) == 0 {
		return
	}
	cols := len(data[0])
	rows := len(data)
	for c := 0; c < cols; c++ {
		col := make([]float64, rows)
		for r := 0; r < rows; r++ {
			col[r] = data[r][c]
		}
		WinsorizeInPlace(col, lower, upper)
		for r := 0; r < rows; r++ {
			data[r][c] = col[r]
		}
	}
}

// IsStable returns false if any value is NaN or Inf.
func IsStable(v []float64) bool {
	for _, x := range v {
		if isNaN(x) || isInf(x) {
			return false
		}
	}
	return true
}

// ClampToFloat64 safely clamps extreme values to float64 range.
func ClampToFloat64(v []float64) []float64 {
	out := make([]float64, len(v))
	for i, x := range v {
		switch {
		case isNaN(x):
			out[i] = 0
		case isInf(x):
			if x > 0 {
				out[i] = maxFloat64
			} else {
				out[i] = -maxFloat64
			}
		default:
			out[i] = x
		}
	}
	return out
}
