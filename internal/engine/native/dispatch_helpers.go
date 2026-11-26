//go:build cgo

package native

// #nosec G115 -- callers bounds-check Dimension and numDirections to [1, math.MaxInt32]
func toInt32(x int) int32 {
	return int32(x)
}
