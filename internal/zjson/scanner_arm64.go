//go:build arm64

package zjson

// findQuoteOrEscape scans for the next quote or backslash byte.
// Returns the index of the first match, or len(s) if none found.
//
// Note: ARM64 NEON SIMD implementation is deferred due to Go assembler
// complexity. The fallback byte-by-byte scan is used for now.
// NEON optimization can be added later via cgo or go:linkname.
func findQuoteOrEscape(s []byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '"' || s[i] == '\\' {
			return i
		}
	}
	return len(s)
}
