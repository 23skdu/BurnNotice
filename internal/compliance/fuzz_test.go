package compliance

import (
	"math"
	"testing"
	"time"
)

func FuzzApprovalWorkflow(f *testing.F) {
	f.Add("session-1", "reason-1", int64(60000))
	f.Add("", "", int64(0))

	f.Fuzz(func(t *testing.T, sessionID, reason string, timeoutMs int64) {
		if len(sessionID) > 100 || len(reason) > 100 {
			return
		}
		if timeoutMs < 0 || timeoutMs > 600000 {
			return
		}
		timeout := time.Duration(timeoutMs) * time.Millisecond
		if timeout == 0 {
			timeout = time.Minute
		}
		aw := NewApprovalWorkflow(timeout)
		defer aw.Stop()

		id := aw.RequestApproval(sessionID, reason)
		if id == "" {
			t.Error("expected non-empty request ID")
		}

		req, ok := aw.GetRequest(id)
		if !ok || req == nil {
			t.Error("expected request to exist")
		}

		pending := aw.GetPendingRequests()
		_ = pending

		if err := aw.Approve(id, "reviewer"); err != nil {
			_ = err
		}
		if err := aw.Reject(id, "reviewer", "reason"); err != nil {
			_ = err
		}
	})
}

func FuzzCapabilityEvaluator(f *testing.F) {
	f.Add(0.5, 0.3, 0.8)

	f.Fuzz(func(t *testing.T, score, threshold, weight float64) {
		if math.IsNaN(score) || math.IsInf(score, 0) ||
			math.IsNaN(threshold) || math.IsInf(threshold, 0) ||
			math.IsNaN(weight) || math.IsInf(weight, 0) {
			return
		}
		ce := NewCapabilityEvaluator()
		_ = ce
	})
}
