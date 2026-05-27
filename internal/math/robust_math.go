package math

import "math"

// Float64 math helpers — wrappers to avoid build tag issues with internal package naming.

func sortFloat64s(a []float64) {
	for i := 1; i < len(a); i++ {
		for j := i; j > 0 && a[j] < a[j-1]; j-- {
			a[j], a[j-1] = a[j-1], a[j]
		}
	}
}

func isNaN(f float64) bool {
	return f != f
}

func isInf(f float64) bool {
	return math.IsInf(f, 0)
}

const maxFloat64 = 1e10
