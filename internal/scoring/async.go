package scoring

import (
	"context"
	"sync/atomic"
	"time"
)

// AsyncScorer handles non-blocking activation scoring with a synchronous bypass fallback.
type AsyncScorer struct {
	ringBuffer   chan []float64
	capacity     int
	bypassActive int32 // Atomic flag (0 = Async, 1 = Synchronous Bypass Active)
	latencyLimit time.Duration
	bypassRatio  float64
	scorer       *Scorer
	onViolation  func(map[string]float64)
	latencyTrack int64 // Atomic nanoseconds processing latency
}

// NewAsyncScorer creates a non-blocking scorer with capacity and safety limits.
func NewAsyncScorer(scorer *Scorer, capacity int, onViolation func(map[string]float64)) *AsyncScorer {
	if capacity <= 0 {
		capacity = 1000
	}
	return &AsyncScorer{
		ringBuffer:   make(chan []float64, capacity),
		capacity:     capacity,
		bypassRatio:  0.85,                    // 85% depth threshold
		latencyLimit: 2500 * time.Microsecond, // 2.5ms latency limit
		scorer:       scorer,
		onViolation:  onViolation,
	}
}

// PushActivation schedules an activation vector for scoring or bypasses synchronously if overloaded.
func (as *AsyncScorer) PushActivation(activation []float64) bool {
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

	// Normal asynchronous enqueue
	select {
	case as.ringBuffer <- activation:
		return false
	default:
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
				case act, ok := <-as.ringBuffer:
					if !ok {
						return
					}
					start := time.Now()
					scores := as.scorer.Score(act)
					elapsed := time.Since(start)

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
