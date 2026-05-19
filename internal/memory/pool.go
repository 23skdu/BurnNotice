package memory

import (
	"sync"
	"sync/atomic"
)

// Pool manages activation buffer allocation with reuse.
type Pool struct {
	mu         sync.Mutex
	buffers    map[int][]*[]float64
	totalAlloc atomic.Int64
	totalReuse atomic.Int64
	maxUsage   int
}

// NewPool creates a buffer pool.
func NewPool(maxUsage int) *Pool {
	return &Pool{
		buffers:  make(map[int][]*[]float64),
		maxUsage: maxUsage,
	}
}

// Get returns a buffer of the requested size.
func (p *Pool) Get(size int) []float64 {
	p.mu.Lock()
	defer p.mu.Unlock()

	if pool, ok := p.buffers[size]; ok && len(pool) > 0 {
		buf := pool[len(pool)-1]
		p.buffers[size] = pool[:len(pool)-1]
		p.totalReuse.Add(1)
		return *buf
	}

	p.totalAlloc.Add(1)
	buf := make([]float64, size)
	return buf
}

// Put returns a buffer to the pool for reuse.
func (p *Pool) Put(buf []float64) {
	p.mu.Lock()
	defer p.mu.Unlock()

	size := len(buf)
	p.buffers[size] = append(p.buffers[size], &buf)
}

// Stats returns pool statistics.
func (p *Pool) Stats() PoolStats {
	p.mu.Lock()
	defer p.mu.Unlock()

	totalBuffered := 0
	bufferedSize := 0
	for size, pool := range p.buffers {
		totalBuffered += len(pool)
		bufferedSize += size * len(pool)
	}

	return PoolStats{
		TotalAlloc:   p.totalAlloc.Load(),
		TotalReuse:   p.totalReuse.Load(),
		CurrentUsage: bufferedSize,
		BufferCount:  totalBuffered,
		MaxUsage:     p.maxUsage,
		Pressure:     float64(bufferedSize) / float64(p.maxUsage),
	}
}

// PoolStats holds pool statistics.
type PoolStats struct {
	TotalAlloc   int64
	TotalReuse   int64
	CurrentUsage int
	BufferCount  int
	MaxUsage     int
	Pressure     float64
}

// ShouldGC returns true if memory pressure exceeds 80%.
func (p *Pool) ShouldGC() bool {
	p.mu.Lock()
	defer p.mu.Unlock()
	if p.maxUsage <= 0 {
		return false
	}

	bufferedSize := 0
	for size, pool := range p.buffers {
		bufferedSize += size * len(pool)
	}
	return float64(bufferedSize)/float64(p.maxUsage) > 0.8
}

// Clear removes all pooled buffers.
func (p *Pool) Clear() {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.buffers = make(map[int][]*[]float64)
}
