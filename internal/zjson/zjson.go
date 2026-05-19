// Package zjson provides a zero-allocation JSON parser/encoder optimized for
// ARM64 Darwin with NEON SIMD acceleration.
//
// Inspired by bytedance/sonic's techniques but implemented from scratch:
//   - sync.Pool buffer reuse with size gating
//   - Two-tier string handling (fast path references input, slow path uses pooled buffer)
//   - JsonState value descriptor for tokenization
//   - ARM64 NEON SIMD quote/escape detection
//   - Lazy AST (offset-based, on-demand parsing)
package zjson

// ValueType represents the type of a JSON value.
type ValueType uint8

const (
	VNull ValueType = iota
	VBool
	VString
	VNumber
	VObject
	VArray
)

// Config controls parser/encoder behavior.
type Config struct {
	// CopyString forces string copies even on the fast path.
	// Default (false) references the input buffer for zero allocation.
	CopyString bool

	// MaxBufferSize caps the pooled buffer size returned to sync.Pool.
	// Prevents a single large parse from permanently bloating all pooled buffers.
	MaxBufferSize int
}

// DefaultConfig is the recommended configuration for most use cases.
var DefaultConfig = Config{
	CopyString:    false,
	MaxBufferSize: 1 << 20, // 1 MB
}

// ValidConfig validates and normalizes a Config.
func (c Config) Valid() Config {
	if c.MaxBufferSize <= 0 {
		c.MaxBufferSize = 1 << 20
	}
	return c
}
