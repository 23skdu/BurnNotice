package profile

import (
	"context"
	"fmt"
	"sync"
	"time"
)

// JobStatus represents the current state of a profiling job.
type JobStatus string

const (
	StatusQueued      JobStatus = "queued"
	StatusRunning     JobStatus = "running"
	StatusComplete    JobStatus = "complete"
	StatusFailed      JobStatus = "failed"
	StatusCheckpointed JobStatus = "checkpointed"
)

// ProfileJob represents a single profiling task.
type ProfileJob struct {
	ID        string
	ModelID   string
	Status    JobStatus
	Progress  float64
	Total     int
	Completed int
	Err       error
	ctx       context.Context
	cancel    context.CancelFunc
	mu        sync.Mutex
}

// Cancel stops the job.
func (j *ProfileJob) Cancel() {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.cancel != nil {
		j.cancel()
	}
}

// Context returns the job's context.
func (j *ProfileJob) Context() context.Context {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.ctx == nil {
		j.ctx, j.cancel = context.WithCancel(context.Background())
	}
	return j.ctx
}

// ProfileManager manages profiling jobs with a worker pool.
type ProfileManager struct {
	jobs    map[string]*ProfileJob
	jobChan chan *ProfileJob
	workers int
	mu      sync.RWMutex
	wg      sync.WaitGroup
}

// NewProfileManager creates a manager with the specified worker count.
func NewProfileManager(workers int) *ProfileManager {
	pm := &ProfileManager{
		jobs:    make(map[string]*ProfileJob),
		jobChan: make(chan *ProfileJob, workers*2),
		workers: workers,
	}
	return pm
}

// Start spawns worker goroutines.
func (pm *ProfileManager) Start(handler func(*ProfileJob) error) {
	for i := 0; i < pm.workers; i++ {
		pm.wg.Add(1)
		go func() {
			defer pm.wg.Done()
			for job := range pm.jobChan {
				pm.executeJob(job, handler)
			}
		}()
	}
}

// Submit enqueues a profiling job.
func (pm *ProfileManager) Submit(job *ProfileJob) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	job.Status = StatusQueued
	job.Progress = 0
	pm.jobs[job.ID] = job
	pm.jobChan <- job
}

// GetJob returns a job by ID.
func (pm *ProfileManager) GetJob(id string) (*ProfileJob, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	job, ok := pm.jobs[id]
	return job, ok
}

// ListJobs returns all jobs.
func (pm *ProfileManager) ListJobs() []*ProfileJob {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	result := make([]*ProfileJob, 0, len(pm.jobs))
	for _, j := range pm.jobs {
		result = append(result, j)
	}
	return result
}

// Stop shuts down the worker pool.
func (pm *ProfileManager) Stop() {
	close(pm.jobChan)
	pm.wg.Wait()
}

func (pm *ProfileManager) executeJob(job *ProfileJob, handler func(*ProfileJob) error) {
	pm.mu.Lock()
	job.Status = StatusRunning
	pm.mu.Unlock()

	start := time.Now()
	defer func() {
		if r := recover(); r != nil {
			pm.mu.Lock()
			job.Status = StatusFailed
			job.Err = fmt.Errorf("panic: %v", r)
			pm.mu.Unlock()
		}
	}()

	if err := handler(job); err != nil {
		pm.mu.Lock()
		job.Status = StatusFailed
		job.Err = err
		job.Progress = 0
		pm.mu.Unlock()
		return
	}

	pm.mu.Lock()
	job.Status = StatusComplete
	job.Progress = 1.0
	job.Completed = job.Total
	_ = start // elapsed time available for logging
	pm.mu.Unlock()
}
