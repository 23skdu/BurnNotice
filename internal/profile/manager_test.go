package profile

import (
	"context"
	"errors"
	"testing"
	"time"
)

func TestNewProfileManager(t *testing.T) {
	pm := NewManager(4)

	if pm.workers != 4 {
		t.Errorf("workers = %d, want 4", pm.workers)
	}
	if len(pm.jobs) != 0 {
		t.Error("jobs should be empty initially")
	}
}

func TestProfileManager_SubmitAndGet(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	pm.Start(func(job *Job) error {
		return nil
	})

	job := &Job{ID: "job1", ModelID: "model1", Total: 10}
	pm.Submit(job)

	time.Sleep(100 * time.Millisecond)

	retrieved, ok := pm.GetJob("job1")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Status != StatusComplete {
		t.Errorf("Status = %s, want complete", retrieved.Status)
	}
	if retrieved.Progress != 1.0 {
		t.Errorf("Progress = %f, want 1.0", retrieved.Progress)
	}
}

func TestProfileManager_JobFails(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	expectedErr := errors.New("handler error")
	pm.Start(func(job *Job) error {
		return expectedErr
	})

	job := &Job{ID: "job2", ModelID: "model1", Total: 5}
	pm.Submit(job)

	time.Sleep(100 * time.Millisecond)

	retrieved, ok := pm.GetJob("job2")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Status != StatusFailed {
		t.Errorf("Status = %s, want failed", retrieved.Status)
	}
	if retrieved.Err == nil {
		t.Error("expected error on failed job")
	}
}

func TestProfileManager_ListJobs(t *testing.T) {
	pm := NewManager(2)
	defer pm.Stop()

	pm.Start(func(job *Job) error {
		return nil
	})

	pm.Submit(&Job{ID: "a", ModelID: "m1", Total: 1})
	pm.Submit(&Job{ID: "b", ModelID: "m1", Total: 1})
	pm.Submit(&Job{ID: "c", ModelID: "m1", Total: 1})

	time.Sleep(100 * time.Millisecond)

	jobs := pm.ListJobs()
	if len(jobs) != 3 {
		t.Errorf("len(jobs) = %d, want 3", len(jobs))
	}
}

func TestProfileManager_CancelJob(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	blocker := make(chan struct{})
	pm.Start(func(job *Job) error {
		<-job.Context().Done()
		return job.Context().Err()
	})

	job := &Job{ID: "cancel1", ModelID: "m1", Total: 10}
	pm.Submit(job)

	time.Sleep(50 * time.Millisecond)

	job.Cancel()

	time.Sleep(100 * time.Millisecond)

	retrieved, ok := pm.GetJob("cancel1")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Status != StatusFailed {
		t.Errorf("Status = %s, want failed after cancel", retrieved.Status)
	}
	_ = blocker
}

func TestProfileManager_JobContext(t *testing.T) {
	job := &Job{ID: "ctx1", ModelID: "m1"}

	ctx := job.Context()
	if ctx == nil {
		t.Fatal("context should not be nil")
	}

	select {
	case <-ctx.Done():
		t.Error("context should not be done initially")
	default:
		// context is active
	}

	job.Cancel()

	select {
	case <-ctx.Done():
		// context was cancelled as expected
	case <-time.After(time.Second):
		t.Fatal("context was not cancelled within timeout")
	}
}

func TestProfileManager_JobContextIdempotent(t *testing.T) {
	job := &Job{ID: "ctx2"}

	ctx1 := job.Context()
	ctx2 := job.Context()

	if ctx1 != ctx2 {
		t.Error("Context() should return the same context on repeated calls")
	}
}

func TestProfileManager_ProgressTracking(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	pm.Start(func(job *Job) error {
		for i := 0; i < 5; i++ {
			job.Completed = i + 1
			job.Progress = float64(i+1) / 5.0
			time.Sleep(10 * time.Millisecond)
		}
		return nil
	})

	job := &Job{ID: "progress1", ModelID: "m1", Total: 5}
	pm.Submit(job)

	time.Sleep(200 * time.Millisecond)

	retrieved, ok := pm.GetJob("progress1")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Completed != 5 {
		t.Errorf("Completed = %d, want 5", retrieved.Completed)
	}
	if retrieved.Progress != 1.0 {
		t.Errorf("Progress = %f, want 1.0", retrieved.Progress)
	}
}

func TestProfileManager_PanicRecovery(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	pm.Start(func(job *Job) error {
		panic("handler panic")
	})

	job := &Job{ID: "panic1", ModelID: "m1", Total: 1}
	pm.Submit(job)

	time.Sleep(100 * time.Millisecond)

	retrieved, ok := pm.GetJob("panic1")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Status != StatusFailed {
		t.Errorf("Status = %s, want failed after panic", retrieved.Status)
	}
	if retrieved.Err == nil {
		t.Error("expected error after panic")
	}
}

func TestProfileManager_MultipleWorkers(t *testing.T) {
	pm := NewManager(4)
	defer pm.Stop()

	pm.Start(func(job *Job) error {
		time.Sleep(50 * time.Millisecond)
		return nil
	})

	for i := 0; i < 8; i++ {
		pm.Submit(&Job{ID: "multi", ModelID: "m1", Total: 1})
	}

	time.Sleep(500 * time.Millisecond)

	retrieved, ok := pm.GetJob("multi")
	if !ok {
		t.Fatal("job not found")
	}
	if retrieved.Status != StatusComplete {
		t.Errorf("Status = %s, want complete", retrieved.Status)
	}
}

func TestProfileManager_GetJobNotFound(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	_, ok := pm.GetJob("nonexistent")
	if ok {
		t.Error("should return false for nonexistent job")
	}
}

func TestProfileManager_JobQueuedStatus(t *testing.T) {
	pm := NewManager(1)
	defer pm.Stop()

	blocker := make(chan struct{})
	pm.Start(func(job *Job) error {
		<-blocker
		return nil
	})

	job1 := &Job{ID: "q1", ModelID: "m1", Total: 1}
	pm.Submit(job1)

	time.Sleep(50 * time.Millisecond)

	job2 := &Job{ID: "q2", ModelID: "m1", Total: 1}
	pm.Submit(job2)

	time.Sleep(50 * time.Millisecond)

	retrieved, _ := pm.GetJob("q2")
	if retrieved.Status != StatusQueued {
		t.Errorf("Status = %s, want queued (worker blocked on job1)", retrieved.Status)
	}

	close(blocker)
	time.Sleep(100 * time.Millisecond)
}

func TestProfileManager_ContextCancellation(t *testing.T) {
	job := &Job{ID: "cancel_ctx", ModelID: "m1"}

	ctx := job.Context()
	job.Cancel()

	select {
	case <-ctx.Done():
		if ctx.Err() != context.Canceled {
			t.Errorf("ctx.Err() = %v, want context.Canceled", ctx.Err())
		}
	case <-time.After(time.Second):
		t.Fatal("context was not cancelled within timeout")
	}
}
