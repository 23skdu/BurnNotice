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
	// StatusQueued means the job is waiting to be processed.
	StatusQueued       JobStatus = "queued"
	StatusRunning      JobStatus = "running"
	StatusComplete     JobStatus = "complete"
	StatusFailed       JobStatus = "failed"
	StatusCheckpointed JobStatus = "checkpointed"
)

// ProfileJob represents a single profiling task.
type Job struct {
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
func (j *Job) Cancel() {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.cancel != nil {
		j.cancel()
	}
}

// Context returns the job's context.
func (j *Job) Context() context.Context {
	j.mu.Lock()
	defer j.mu.Unlock()
	if j.ctx == nil {
		j.ctx, j.cancel = context.WithCancel(context.Background())
	}
	return j.ctx
}

// ProfileManager manages profiling jobs with a worker pool.
type Manager struct {
	jobs    map[string]*Job
	jobChan chan *Job
	workers int
	mu      sync.RWMutex
	wg      sync.WaitGroup
}

// NewProfileManager creates a manager with the specified worker count.
func NewManager(workers int) *Manager {
	pm := &Manager{
		jobs:    make(map[string]*Job),
		jobChan: make(chan *Job, workers*2),
		workers: workers,
	}
	return pm
}

// Start spawns worker goroutines.
func (pm *Manager) Start(handler func(*Job) error) {
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
func (pm *Manager) Submit(job *Job) {
	pm.mu.Lock()
	defer pm.mu.Unlock()

	job.Status = StatusQueued
	job.Progress = 0
	pm.jobs[job.ID] = job
	pm.jobChan <- job
}

// GetJob returns a job by ID.
func (pm *Manager) GetJob(id string) (*Job, bool) {
	pm.mu.RLock()
	defer pm.mu.RUnlock()
	job, ok := pm.jobs[id]
	return job, ok
}

// ListJobs returns all jobs.
func (pm *Manager) ListJobs() []*Job {
	pm.mu.RLock()
	defer pm.mu.RUnlock()

	result := make([]*Job, 0, len(pm.jobs))
	for _, j := range pm.jobs {
		result = append(result, j)
	}
	return result
}

// Stop shuts down the worker pool.
func (pm *Manager) Stop() {
	close(pm.jobChan)
	pm.wg.Wait()
}

func (pm *Manager) executeJob(job *Job, handler func(*Job) error) {
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
