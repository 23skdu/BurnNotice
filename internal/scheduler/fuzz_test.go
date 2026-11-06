package scheduler

import (
	"testing"
)

func FuzzSchedulerSubmitAndPreempt(f *testing.F) {
	f.Add(1000, "job-1", int64(100), 5, int64(200), 10)
	f.Add(100, "job-2", int64(50*1024*1024), 1, int64(25*1024*1024), 8)
	f.Add(1, "", int64(0), -1, int64(1024), 0)

	f.Fuzz(func(t *testing.T, gpuMemMB int, id string, size1 int64, prio1 int, size2 int64, prio2 int) {
		if gpuMemMB <= 0 || gpuMemMB > 100000 {
			gpuMemMB = 1000
		}
		if size1 < 0 || size1 > 1e12 {
			size1 = 1024
		}
		if size2 < 0 || size2 > 1e12 {
			size2 = 1024
		}

		s := NewScheduler(gpuMemMB)
		s.Submit(&Job{ID: id + "-1", ModelSize: size1, Priority: prio1})
		s.Submit(&Job{ID: id + "-2", ModelSize: size2, Priority: prio2})

		running := s.Schedule()
		if s.AvailableMemory() < 0 {
			t.Fatalf("negative available memory: %d", s.AvailableMemory())
		}

		high := &Job{
			ID:        id + "-high",
			ModelSize: size1,
			Priority:  prio1 + 5,
			MemoryReq: estimateMemory(size1),
		}
		_, _ = s.Preempt(high)

		for _, j := range running {
			_ = s.Complete(j.ID)
		}
	})
}
