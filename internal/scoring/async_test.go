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

func TestAsyncScorerConcurrentRace(t *testing.T) {
	scorer := NewScorer(nil)
	dir1 := make([]float64, 128)
	dir1[0] = 1.0
	dir1[1] = -0.5
	dir2 := make([]float64, 128)
	dir2[1] = 1.0
	scorer.RegisterDirection("deception:0", dir1)
	scorer.RegisterDirection("truth:0", dir2)

	var violationCallbacks int64
	onViolation := func(scores map[string]float64) {
		atomic.AddInt64(&violationCallbacks, 1)
	}

	asyncScorer := NewAsyncScorer(scorer, 16, onViolation)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()

	asyncScorer.Start(ctx, 4)

	const producers = 8
	const iterations = 100
	done := make(chan struct{}, producers+2)

	for p := 0; p < producers; p++ {
		go func(id int) {
			defer func() { done <- struct{}{} }()
			act := make([]float64, 128)
			act[0] = float64(id+1) * 0.1
			act[1] = 0.25
			for i := 0; i < iterations; i++ {
				if i%2 == 0 {
					_ = asyncScorer.PushActivation(act)
				} else {
					_ = asyncScorer.PushActivationPooled(act)
				}
			}
		}(p)
	}

	// Concurrent state readers (IsBypassActive & QueueDepth)
	go func() {
		defer func() { done <- struct{}{} }()
		for i := 0; i < iterations*2; i++ {
			_ = asyncScorer.IsBypassActive()
			depth := asyncScorer.QueueDepth()
			if depth < 0 || depth > 16 {
				t.Errorf("invalid queue depth: %d", depth)
			}
		}
	}()

	// Concurrent Score & GetScore reader
	go func() {
		defer func() { done <- struct{}{} }()
		for i := 0; i < iterations; i++ {
			_ = scorer.GetScore("deception:0")
		}
	}()

	for i := 0; i < producers+2; i++ {
		<-done
	}

	// Wait briefly for queue drain
	deadline := time.Now().Add(500 * time.Millisecond)
	for asyncScorer.QueueDepth() > 0 && time.Now().Before(deadline) {
		time.Sleep(5 * time.Millisecond)
	}

	if atomic.LoadInt64(&violationCallbacks) == 0 {
		t.Errorf("expected violation callbacks > 0 under concurrent load")
	}
}

func TestAsyncScorer_DefaultCapacityAndReset(t *testing.T) {
	scorer := NewScorer(nil)
	scorer.RegisterDirection("d", []float64{1, 0})

	// Test default capacity <= 0 fallback
	as := NewAsyncScorer(scorer, 0, nil)
	if as.capacity != 1000 {
		t.Errorf("expected default capacity 1000, got %d", as.capacity)
	}
}
