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
