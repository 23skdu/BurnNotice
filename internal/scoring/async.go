package scoring

import (
	"context"
	"sync"
	"sync/atomic"
	"time"
)

type pooledActivation struct {
	data   []float64
	pooled *[]float64
}

// AsyncScorer handles non-blocking activation scoring with a synchronous bypass fallback.
type AsyncScorer struct {
	ringBuffer   chan pooledActivation
	capacity     int
	bypassActive int32 // Atomic flag (0 = Async, 1 = Synchronous Bypass Active)
	latencyLimit time.Duration
	bypassRatio  float64
	scorer       *Scorer
	onViolation  func(map[string]float64)
	latencyTrack int64 // Atomic nanoseconds processing latency
	bufPool      sync.Pool
}

// NewAsyncScorer creates a non-blocking scorer with capacity and safety limits.
func NewAsyncScorer(scorer *Scorer, capacity int, onViolation func(map[string]float64)) *AsyncScorer {
	if capacity <= 0 {
		capacity = 1000
	}
	return &AsyncScorer{
		ringBuffer:   make(chan pooledActivation, capacity),
		capacity:     capacity,
		bypassRatio:  0.85,                    // 85% depth threshold
		latencyLimit: 2500 * time.Microsecond, // 2.5ms latency limit
		scorer:       scorer,
		onViolation:  onViolation,
	}
}

func (as *AsyncScorer) acquireBuffer(n int) ([]float64, *[]float64) {
	if raw := as.bufPool.Get(); raw != nil {
		ptr := raw.(*[]float64)
		if cap(*ptr) >= n {
			s := (*ptr)[:n]
			return s, ptr
		}
	}
	buf := make([]float64, n)
	return buf, &buf
}

func (as *AsyncScorer) releaseBuffer(ptr *[]float64) {
	if ptr != nil {
		as.bufPool.Put(ptr)
	}
}

// PushActivation schedules an activation vector for scoring or bypasses synchronously if overloaded.
func (as *AsyncScorer) PushActivation(activation []float64) bool {
	return as.pushInternal(activation, false)
}

// PushActivationPooled copies the activation into a reusable pooled slice before async enqueueing,
// avoiding caller heap escapes and reducing GC pressure in high-throughput pipelines.
func (as *AsyncScorer) PushActivationPooled(activation []float64) bool {
	return as.pushInternal(activation, true)
}

func (as *AsyncScorer) pushInternal(activation []float64, copyToPool bool) bool {
	depth := len(as.ringBuffer)
	ratio := float64(depth) / float64(as.capacity)

	latencyNS := atomic.LoadInt64(&as.latencyTrack)
	latencyVal := time.Duration(latencyNS)

	// Trigger emergency synchronous bypass if queue depth > 85% or latency > 2.5ms
	if ratio >= as.bypassRatio || latencyVal > as.latencyLimit {
		atomic.StoreInt32(&as.bypassActive, 1)
		// Run synchronously immediately
		scores := as.scorer.Score(activation)
		if as.onViolation != nil {
			as.onViolation(scores)
		}
		return true // Synced bypass executed
	}

	item := pooledActivation{data: activation}
	if copyToPool {
		buf, ptr := as.acquireBuffer(len(activation))
		copy(buf, activation)
		item = pooledActivation{data: buf, pooled: ptr}
	}

	// Normal asynchronous enqueue
	select {
	case as.ringBuffer <- item:
		return false
	default:
		as.releaseBuffer(item.pooled)
		// Fallback to sync if queue is physically full
		atomic.StoreInt32(&as.bypassActive, 1)
		scores := as.scorer.Score(activation)
		if as.onViolation != nil {
			as.onViolation(scores)
		}
		return true
	}
}

// Start spawns background consumer goroutines processing scored activations.
func (as *AsyncScorer) Start(ctx context.Context, workers int) {
	for i := 0; i < workers; i++ {
		go func() {
			for {
				select {
				case <-ctx.Done():
					return
				case item, ok := <-as.ringBuffer:
					if !ok {
						return
					}
					start := time.Now()
					scores := as.scorer.Score(item.data)
					elapsed := time.Since(start)
					as.releaseBuffer(item.pooled)

					// Update latency tracker atomically
					atomic.StoreInt64(&as.latencyTrack, elapsed.Nanoseconds())

					if as.onViolation != nil {
						as.onViolation(scores)
					}

					// Dynamically reset bypass state if queue clears and latency is safe
					if len(as.ringBuffer) == 0 && elapsed < as.latencyLimit {
						atomic.StoreInt32(&as.bypassActive, 0)
					}
				}
			}
		}()
	}
}

// IsBypassActive returns true if the emergency synchronous bypass is active.
func (as *AsyncScorer) IsBypassActive() bool {
	return atomic.LoadInt32(&as.bypassActive) == 1
}

// QueueDepth returns the current number of elements waiting in the buffer.
func (as *AsyncScorer) QueueDepth() int {
	return len(as.ringBuffer)
}
