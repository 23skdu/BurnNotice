//go:build !arm64

package zjson

// findQuoteOrEscape scans for the next quote or backslash byte.
// Returns the index of the first match, or len(s) if none found.
func findQuoteOrEscape(s []byte) int {
	for i := 0; i < len(s); i++ {
		if s[i] == '"' || s[i] == '\\' {
			return i
		}
	}
	return len(s)
}
