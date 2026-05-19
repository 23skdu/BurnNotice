package scheduler

import (
	"fmt"
	"sort"
	"sync"
	"time"
)

// Job represents a profiling job.
type Job struct {
	ID         string
	ModelSize  int64
	Priority   int
	Status     string
	MemoryReq  int64
	StartTime  time.Time
	cancelCh   chan struct{}
}

// Scheduler manages GPU-aware job scheduling.
type Scheduler struct {
	mu            sync.RWMutex
	gpuMemory     int64
	usedMemory    int64
	queue         []*Job
	running       map[string]*Job
	preemptCh     chan string
}

// NewScheduler creates a scheduler with the given GPU memory.
func NewScheduler(gpuMemoryMB int) *Scheduler {
	return &Scheduler{
		gpuMemory: int64(gpuMemoryMB) * 1024 * 1024,
		running:   make(map[string]*Job),
		preemptCh: make(chan string, 1),
	}
}

// Submit adds a job to the queue.
func (s *Scheduler) Submit(job *Job) {
	s.mu.Lock()
	defer s.mu.Unlock()

	job.MemoryReq = estimateMemory(job.ModelSize)
	job.Status = "queued"
	job.cancelCh = make(chan struct{})
	s.queue = append(s.queue, job)

	sort.Slice(s.queue, func(i, j int) bool {
		return s.queue[i].Priority > s.queue[j].Priority
	})
}

// Schedule attempts to run queued jobs.
func (s *Scheduler) Schedule() []*Job {
	s.mu.Lock()
	defer s.mu.Unlock()

	var toRun []*Job
	remaining := make([]*Job, 0, len(s.queue))

	for _, job := range s.queue {
		if s.usedMemory+job.MemoryReq <= s.gpuMemory {
			job.Status = "running"
			job.StartTime = time.Now()
			s.running[job.ID] = job
			s.usedMemory += job.MemoryReq
			toRun = append(toRun, job)
		} else {
			remaining = append(remaining, job)
		}
	}

	s.queue = remaining
	return toRun
}

// Complete marks a job as complete and frees memory.
func (s *Scheduler) Complete(jobID string) error {
	s.mu.Lock()
	defer s.mu.Unlock()

	job, ok := s.running[jobID]
	if !ok {
		return fmt.Errorf("job %s not running", jobID)
	}

	job.Status = "complete"
	s.usedMemory -= job.MemoryReq
	delete(s.running, jobID)
	return nil
}

// Preempt pauses a lower-priority job for a higher-priority one.
func (s *Scheduler) Preempt(highPriorityJob *Job) (*Job, error) {
	s.mu.Lock()
	defer s.mu.Unlock()

	if s.usedMemory+highPriorityJob.MemoryReq <= s.gpuMemory {
		return nil, nil
	}

	var lowestPriority *Job
	for _, job := range s.running {
		if lowestPriority == nil || job.Priority < lowestPriority.Priority {
			lowestPriority = job
		}
	}

	if lowestPriority == nil {
		return nil, fmt.Errorf("no running jobs to preempt")
	}

	if lowestPriority.Priority >= highPriorityJob.Priority {
		return nil, fmt.Errorf("no lower-priority job to preempt")
	}

	lowestPriority.Status = "preempted"
	s.usedMemory -= lowestPriority.MemoryReq
	delete(s.running, lowestPriority.ID)
	close(lowestPriority.cancelCh)

	return lowestPriority, nil
}

// GetRunningJobs returns all currently running jobs.
func (s *Scheduler) GetRunningJobs() []*Job {
	s.mu.RLock()
	defer s.mu.RUnlock()

	jobs := make([]*Job, 0, len(s.running))
	for _, job := range s.running {
		jobs = append(jobs, job)
	}
	return jobs
}

// GetQueue returns the job queue.
func (s *Scheduler) GetQueue() []*Job {
	s.mu.RLock()
	defer s.mu.RUnlock()

	result := make([]*Job, len(s.queue))
	copy(result, s.queue)
	return result
}

// AvailableMemory returns the remaining GPU memory.
func (s *Scheduler) AvailableMemory() int64 {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.gpuMemory - s.usedMemory
}

func estimateMemory(modelSizeBytes int64) int64 {
	return modelSizeBytes * 4
}
