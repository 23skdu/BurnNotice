package compliance

import (
	"fmt"
	"sync"
	"sync/atomic"
	"time"
)

// ApprovalStatus represents the state of an approval request.
type ApprovalStatus string

const (
	StatusPending  ApprovalStatus = "pending"
	StatusApproved ApprovalStatus = "approved"
	StatusRejected ApprovalStatus = "rejected"
	StatusTimedOut ApprovalStatus = "timed_out"
)

// ApprovalRequest represents a human-in-the-loop approval request.
type ApprovalRequest struct {
	ID        string
	SessionID string
	Reason    string
	Status    ApprovalStatus
	CreatedAt time.Time
	ReviewedAt *time.Time
	Reviewer   string
	RejectReason string
}

// ApprovalWorkflow manages approval requests with timeout.
type ApprovalWorkflow struct {
	mu            sync.RWMutex
	requests      map[string]*ApprovalRequest
	defaultTimeout time.Duration
	ticker        *time.Ticker
	stopCh        chan struct{}
}

// NewApprovalWorkflow creates a new approval workflow.
func NewApprovalWorkflow(defaultTimeout time.Duration) *ApprovalWorkflow {
	if defaultTimeout <= 0 {
		defaultTimeout = 5 * time.Minute
	}
	aw := &ApprovalWorkflow{
		requests:       make(map[string]*ApprovalRequest),
		defaultTimeout: defaultTimeout,
		stopCh:         make(chan struct{}),
	}
	aw.startTimeoutChecker()
	return aw
}

func (aw *ApprovalWorkflow) startTimeoutChecker() {
	tickerInterval := aw.defaultTimeout / 10
	if tickerInterval < 50*time.Millisecond {
		tickerInterval = 50 * time.Millisecond
	}
	aw.ticker = time.NewTicker(tickerInterval)
	go func() {
		for {
			select {
			case <-aw.ticker.C:
				aw.checkTimeouts()
			case <-aw.stopCh:
				aw.ticker.Stop()
				return
			}
		}
	}()
}

func (aw *ApprovalWorkflow) checkTimeouts() {
	aw.mu.Lock()
	defer aw.mu.Unlock()

	now := time.Now()
	for _, req := range aw.requests {
		if req.Status == StatusPending && now.Sub(req.CreatedAt) > aw.defaultTimeout {
			req.Status = StatusTimedOut
		}
	}
}

var reqCounter uint64

// RequestApproval creates a new approval request.
func (aw *ApprovalWorkflow) RequestApproval(sessionID, reason string) string {
	aw.mu.Lock()
	defer aw.mu.Unlock()

	count := atomic.AddUint64(&reqCounter, 1)
	id := fmt.Sprintf("approval-%d-%d", time.Now().UnixNano(), count)
	aw.requests[id] = &ApprovalRequest{
		ID:        id,
		SessionID: sessionID,
		Reason:    reason,
		Status:    StatusPending,
		CreatedAt: time.Now(),
	}
	return id
}

// Approve marks a request as approved.
func (aw *ApprovalWorkflow) Approve(requestID, reviewer string) error {
	aw.mu.Lock()
	defer aw.mu.Unlock()

	req, ok := aw.requests[requestID]
	if !ok {
		return fmt.Errorf("request %s not found", requestID)
	}
	if req.Status != StatusPending {
		return fmt.Errorf("request %s is not pending (status: %s)", requestID, req.Status)
	}

	req.Status = StatusApproved
	req.Reviewer = reviewer
	now := time.Now()
	req.ReviewedAt = &now
	return nil
}

// Reject marks a request as rejected.
func (aw *ApprovalWorkflow) Reject(requestID, reviewer, reason string) error {
	aw.mu.Lock()
	defer aw.mu.Unlock()

	req, ok := aw.requests[requestID]
	if !ok {
		return fmt.Errorf("request %s not found", requestID)
	}
	if req.Status != StatusPending {
		return fmt.Errorf("request %s is not pending (status: %s)", requestID, req.Status)
	}

	req.Status = StatusRejected
	req.Reviewer = reviewer
	req.RejectReason = reason
	now := time.Now()
	req.ReviewedAt = &now
	return nil
}

// GetRequest returns an approval request by ID.
func (aw *ApprovalWorkflow) GetRequest(requestID string) (*ApprovalRequest, bool) {
	aw.mu.RLock()
	defer aw.mu.RUnlock()

	req, ok := aw.requests[requestID]
	return req, ok
}

// GetPendingRequests returns all pending approval requests.
func (aw *ApprovalWorkflow) GetPendingRequests() []*ApprovalRequest {
	aw.mu.RLock()
	defer aw.mu.RUnlock()

	var pending []*ApprovalRequest
	for _, req := range aw.requests {
		if req.Status == StatusPending {
			pending = append(pending, req)
		}
	}
	return pending
}

// Stop stops the timeout checker.
func (aw *ApprovalWorkflow) Stop() {
	close(aw.stopCh)
}
