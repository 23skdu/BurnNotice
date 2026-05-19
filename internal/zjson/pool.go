package zjson

import (
	"bytes"
	"sync"
)

const (
	// DefaultBufSize is the initial buffer size for pooled buffers.
	DefaultBufSize = 4096

	// DefaultDigitBufSize is the size of the pooled number parsing buffer.
	// 800 bytes covers the maximum digits needed for float64.
	DefaultDigitBufSize = 800
)

// bytePool pools reusable byte slices for encoding/decoding.
var bytePool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, DefaultBufSize)
		return &b
	},
}

// digitPool pools fixed-size buffers for number parsing.
var digitPool = sync.Pool{
	New: func() any {
		b := make([]byte, 0, DefaultDigitBufSize)
		return &b
	},
}

// GetBuf retrieves a pooled byte slice.
func GetBuf() *[]byte {
	return bytePool.Get().(*[]byte)
}

// PutBuf returns a byte slice to the pool if its capacity is reasonable.
func PutBuf(b *[]byte, maxAllowed int) {
	if CanSizeReuse(cap(*b), maxAllowed) {
		*b = (*b)[:0]
		bytePool.Put(b)
	}
}

// GetDigitBuf retrieves a pooled digit buffer for number parsing.
func GetDigitBuf() *[]byte {
	return digitPool.Get().(*[]byte)
}

// PutDigitBuf returns a digit buffer to the pool.
func PutDigitBuf(b *[]byte) {
	*b = (*b)[:0]
	digitPool.Put(b)
}

// BufferPool wraps a pooled buffer with size gating for safe reuse.
type BufferPool struct {
	pool       *sync.Pool
	maxAllowed int
}

// NewBufferPool creates a new BufferPool with the given max allowed size.
func NewBufferPool(initialSize, maxAllowed int) *BufferPool {
	return &BufferPool{
		pool: &sync.Pool{
			New: func() any {
				b := make([]byte, 0, initialSize)
				return &b
			},
		},
		maxAllowed: maxAllowed,
	}
}

// Get retrieves a buffer from the pool.
func (bp *BufferPool) Get() *[]byte {
	return bp.pool.Get().(*[]byte)
}

// Put returns a buffer to the pool if size is acceptable.
func (bp *BufferPool) Put(b *[]byte) {
	if CanSizeReuse(cap(*b), bp.maxAllowed) {
		*b = (*b)[:0]
		bp.pool.Put(b)
	}
}

// BytesPool provides pooled bytes.Buffer instances.
var bytesBufferPool = sync.Pool{
	New: func() any {
		return bytes.NewBuffer(make([]byte, 0, DefaultBufSize))
	},
}

// GetBytesBuffer retrieves a pooled bytes.Buffer.
func GetBytesBuffer() *bytes.Buffer {
	return bytesBufferPool.Get().(*bytes.Buffer)
}

// PutBytesBuffer returns a bytes.Buffer to the pool.
func PutBytesBuffer(buf *bytes.Buffer) {
	if buf.Cap() <= DefaultBufSize*4 {
		buf.Reset()
		bytesBufferPool.Put(buf)
	}
}
