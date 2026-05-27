package scoring

import (
	"context"
	"sync/atomic"
	"testing"
	"time"
)

func TestAsyncScorer_ExecutionAndBypass(t *testing.T) {
	scorer := NewScorer(nil)
	dir := make([]float64, 4096)
	dir[0] = 1.0
	scorer.RegisterDirection("deception:0", dir)

	var scoreCallbackCount int32
	onViolation := func(scores map[string]float64) {
		atomic.AddInt32(&scoreCallbackCount, 1)
	}

	t.Run("Normal Async Flow", func(t *testing.T) {
		atomic.StoreInt32(&scoreCallbackCount, 0)
		asyncScorer := NewAsyncScorer(scorer, 100, onViolation)

		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()

		asyncScorer.Start(ctx, 2)

		act := make([]float64, 4096)
		act[0] = 0.5

		// Queue up activations
		for i := 0; i < 5; i++ {
			bypassed := asyncScorer.PushActivation(act)
			if bypassed {
				t.Errorf("expected normal async enqueue, but synchronous bypass triggered")
			}
		}

		// Wait a brief moment for async workers to digest
		time.Sleep(100 * time.Millisecond)

		count := atomic.LoadInt32(&scoreCallbackCount)
		if count != 5 {
			t.Errorf("expected 5 callbacks processed, got %d", count)
		}
		if asyncScorer.IsBypassActive() {
			t.Errorf("expected bypass to remain inactive under low load")
		}
	})

	t.Run("Synchronous Bypass Trigger on Queue Depth", func(t *testing.T) {
		atomic.StoreInt32(&scoreCallbackCount, 0)
		// Small buffer size to easily cross 85% depth
		asyncScorer := NewAsyncScorer(scorer, 10, onViolation)

		act := make([]float64, 4096)
		act[0] = 0.5

		// Push 10 items (90% capacity, exceeding 85% bypass threshold)
		bypassedCount := 0
		for i := 0; i < 10; i++ {
			bypassed := asyncScorer.PushActivation(act)
			if bypassed {
				bypassedCount++
			}
		}

		if bypassedCount == 0 {
			t.Errorf("expected synchronous bypass to trigger on queue depth > 85%%")
		}
		if !asyncScorer.IsBypassActive() {
			t.Errorf("expected bypassActive flag to be true")
		}
	})
}
