package scoring

import (
	"context"
	"sync"
	"sync/atomic"
	"testing"
	"time"
)

func TestAsyncScorer_ConcurrentPooledRingBuffer(t *testing.T) {
	s := NewScorer(nil)
	s.RegisterDirection("truth", []float64{1, 0, 0, 0})
	s.RegisterDirection("deception", []float64{-1, 0, 0, 0})

	var violations atomic.Int64
	as := NewAsyncScorer(s, 64, func(scores map[string]float64) {
		if len(scores) > 0 {
			violations.Add(1)
		}
	})

	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	as.Start(ctx, 4)

	var wg sync.WaitGroup
	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			act := []float64{float64(id + 1), 0.5, -0.2, 0.1}
			for i := 0; i < 100; i++ {
				_ = as.PushActivationPooled(act)
				_ = as.IsBypassActive()
				_ = as.QueueDepth()
			}
		}(w)
	}
	wg.Wait()

	deadline := time.Now().Add(500 * time.Millisecond)
	for as.QueueDepth() > 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}
	if violations.Load() == 0 {
		t.Errorf("expected scored activations to invoke callback")
	}
}
