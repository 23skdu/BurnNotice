package scheduler

import (
	"fmt"
	"sync"
	"testing"
)

func TestScheduler_ConcurrentSubmitScheduleComplete(t *testing.T) {
	s := NewScheduler(2048)
	var wg sync.WaitGroup

	for w := 0; w < 8; w++ {
		wg.Add(1)
		go func(worker int) {
			defer wg.Done()
			for i := 0; i < 25; i++ {
				id := fmt.Sprintf("job-%d-%d", worker, i)
				s.Submit(&Job{
					ID:        id,
					ModelSize: 1024 * 1024,
					Priority:  (worker + i) % 10,
				})
				ran := s.Schedule()
				_ = s.GetQueue()
				_ = s.GetRunningJobs()
				_ = s.AvailableMemory()
				for _, r := range ran {
					_ = s.Complete(r.ID)
				}
			}
		}(w)
	}

	wg.Wait()
}
