package scheduler

import (
	"testing"
)

func TestSmallJobsParallel(t *testing.T) {
	s := NewScheduler(10000)

	s.Submit(&Job{ID: "job1", ModelSize: 100, Priority: 1})
	s.Submit(&Job{ID: "job2", ModelSize: 100, Priority: 1})
	s.Submit(&Job{ID: "job3", ModelSize: 100, Priority: 1})

	running := s.Schedule()
	if len(running) != 3 {
		t.Fatalf("len(running) = %d, want 3", len(running))
	}
}

func TestLargeJobExclusive(t *testing.T) {
	s := NewScheduler(1000)

	s.Submit(&Job{ID: "large", ModelSize: 200, Priority: 1})
	s.Submit(&Job{ID: "small", ModelSize: 200, Priority: 1})

	running := s.Schedule()
	if len(running) != 2 {
		t.Fatalf("len(running) = %d, want 2", len(running))
	}
}

func TestPreemption(t *testing.T) {
	s := NewScheduler(1000)

	s.Submit(&Job{ID: "low", ModelSize: 200000000, Priority: 1})
	s.Schedule()

	high := &Job{ID: "high", ModelSize: 200000000, Priority: 5}
	high.MemoryReq = high.ModelSize * 4

	preempted, err := s.Preempt(high)
	if err != nil {
		t.Fatalf("Preempt error: %v", err)
	}
	if preempted == nil {
		t.Fatal("expected preempted job")
	}
	if preempted.ID != "low" {
		t.Errorf("preempted job = %s, want low", preempted.ID)
	}
}

func TestPriorityOrdering(t *testing.T) {
	s := NewScheduler(1000)

	s.Submit(&Job{ID: "low", ModelSize: 100, Priority: 1})
	s.Submit(&Job{ID: "high", ModelSize: 100, Priority: 10})
	s.Submit(&Job{ID: "med", ModelSize: 100, Priority: 5})

	running := s.Schedule()
	if len(running) != 3 {
		t.Fatalf("len(running) = %d, want 3", len(running))
	}

	if running[0].ID != "high" {
		t.Errorf("first job = %s, want high", running[0].ID)
	}
}

func TestCompleteFreesMemory(t *testing.T) {
	s := NewScheduler(1000)

	s.Submit(&Job{ID: "job1", ModelSize: 500, Priority: 1})
	s.Schedule()

	availBefore := s.AvailableMemory()
	_ = s.Complete("job1")
	availAfter := s.AvailableMemory()

	if availAfter <= availBefore {
		t.Fatal("available memory should increase after complete")
	}
}

func TestAvailableMemory(t *testing.T) {
	s := NewScheduler(1000)

	if s.AvailableMemory() != 1000*1024*1024 {
		t.Fatal("initial available memory incorrect")
	}
}

func TestComplete_Nonexistent(t *testing.T) {
	s := NewScheduler(1000)
	err := s.Complete("nonexistent")
	if err == nil {
		t.Fatal("expected error completing non-running job")
	}
}

func TestPreempt_EnoughMemory(t *testing.T) {
	s := NewScheduler(1000)
	high := &Job{ID: "high", ModelSize: 50, Priority: 5}
	high.MemoryReq = 200
	
	preempted, err := s.Preempt(high)
	if err != nil {
		t.Fatalf("Preempt failed: %v", err)
	}
	if preempted != nil {
		t.Errorf("expected no preemption when memory is sufficient, got %s", preempted.ID)
	}
}

func TestPreempt_NoRunningJobs(t *testing.T) {
	s := NewScheduler(100) // 100MB GPU Memory
	high := &Job{ID: "high", ModelSize: 150 * 1024 * 1024, Priority: 5}
	high.MemoryReq = high.ModelSize * 4
	
	_, err := s.Preempt(high)
	if err == nil {
		t.Fatal("expected error preempting with no running jobs")
	}
}

func TestPreempt_NoLowerPriorityJobs(t *testing.T) {
	s := NewScheduler(100) // 100MB GPU Memory
	
	// Submit a high priority job and run it
	s.Submit(&Job{ID: "high-1", ModelSize: 20 * 1024 * 1024, Priority: 10}) // 80MB
	s.Schedule()
	
	// Attempt to preempt with a lower priority job
	lower := &Job{ID: "lower", ModelSize: 20 * 1024 * 1024, Priority: 5} // 80MB
	lower.MemoryReq = lower.ModelSize * 4
	
	_, err := s.Preempt(lower)
	if err == nil {
		t.Fatal("expected error preempting with no lower priority jobs")
	}
}

func TestGetRunningJobsAndQueue(t *testing.T) {
	s := NewScheduler(100) // 100MB GPU Memory
	
	s.Submit(&Job{ID: "q1", ModelSize: 20 * 1024 * 1024, Priority: 1}) // 80MB
	s.Submit(&Job{ID: "q2", ModelSize: 15 * 1024 * 1024, Priority: 2}) // 60MB (higher priority)
	
	running := s.Schedule()
	if len(running) != 1 {
		t.Fatalf("len(running) = %d, want 1", len(running))
	}
	
	rJobs := s.GetRunningJobs()
	if len(rJobs) != 1 {
		t.Errorf("GetRunningJobs len = %d, want 1", len(rJobs))
	}
	if rJobs[0].ID != "q2" { // q2 has higher priority, so scheduled first
		t.Errorf("running job ID = %s, want q2", rJobs[0].ID)
	}
	
	queue := s.GetQueue()
	if len(queue) != 1 {
		t.Errorf("GetQueue len = %d, want 1", len(queue))
	}
	if queue[0].ID != "q1" {
		t.Errorf("queued job ID = %s, want q1", queue[0].ID)
	}
}
