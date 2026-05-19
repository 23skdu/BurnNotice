package compliance

import (
	"os"
	"testing"
	"time"

	"burnnotice/internal/ledger"
)

func TestGenerateReport(t *testing.T) {
	tmp := t.TempDir() + "/report_test.db"
	defer os.Remove(tmp)

	backend, _ := ledger.NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-001",
		Metric:    "deception_score",
		Score:     0.85,
		Threshold: 0.7,
		Action:    "steer",
	})
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-002",
		Metric:    "deception_score",
		Score:     0.9,
		Threshold: 0.7,
		Action:    "block",
	})

	reporter := NewReporter(backend)
	report, err := reporter.GenerateReport(now.Add(-1*time.Hour), now.Add(1*time.Hour))
	if err != nil {
		t.Fatalf("GenerateReport error: %v", err)
	}

	if report.TotalAlerts != 2 {
		t.Errorf("TotalAlerts = %d, want 2", report.TotalAlerts)
	}
	if report.AlertsByMetric["deception_score"] != 2 {
		t.Errorf("deception_score count = %d, want 2", report.AlertsByMetric["deception_score"])
	}
}

func TestEUAIActReport(t *testing.T) {
	tmp := t.TempDir() + "/eu_test.db"
	defer os.Remove(tmp)

	backend, _ := ledger.NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-001",
		Metric:    "deception_score",
		Score:     0.9,
		Threshold: 0.7,
		Action:    "block",
	})
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-002",
		Metric:    "deception_score",
		Score:     0.6,
		Threshold: 0.7,
		Action:    "log",
	})

	reporter := NewReporter(backend)
	report, err := reporter.GenerateReport(now.Add(-1*time.Hour), now.Add(1*time.Hour))
	if err != nil {
		t.Fatalf("GenerateReport error: %v", err)
	}

	if report.EUAIAct.HighRiskAlerts == 0 {
		t.Fatal("expected high risk alerts")
	}
}

func TestNISTAIRMFReport(t *testing.T) {
	tmp := t.TempDir() + "/nist_test.db"
	defer os.Remove(tmp)

	backend, _ := ledger.NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	now := time.Now()
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-001",
		Metric:    "deception_score",
		Score:     0.8,
		Threshold: 0.7,
		Action:    "block",
	})
	backend.RecordAlert(ledger.Alert{
		Timestamp: now,
		SessionID: "sess-002",
		Metric:    "deception_score",
		Score:     0.6,
		Threshold: 0.7,
		Action:    "log",
	})

	reporter := NewReporter(backend)
	report, err := reporter.GenerateReport(now.Add(-1*time.Hour), now.Add(1*time.Hour))
	if err != nil {
		t.Fatalf("GenerateReport error: %v", err)
	}

	if report.NISTAIRMF.Govern == 0 {
		t.Fatal("expected Govern alerts")
	}
	if report.NISTAIRMF.Measure == 0 {
		t.Fatal("expected Measure alerts")
	}
}

func TestExportReport(t *testing.T) {
	tmp := t.TempDir() + "/export_test.db"
	defer os.Remove(tmp)

	backend, _ := ledger.NewBackend("sqlite", tmp)
	defer backend.Close()
	backend.Init()

	reporter := NewReporter(backend)
	report := &Report{
		GeneratedAt:     time.Now(),
		TotalAlerts:     5,
		ComplianceScore: 0.85,
	}

	exported, err := reporter.ExportReport(report)
	if err != nil {
		t.Fatalf("ExportReport error: %v", err)
	}
	if exported == "" {
		t.Fatal("exported string is empty")
	}
}

func TestApprovalWorkflow(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	id := aw.RequestApproval("sess-001", "High deception score detected")
	if id == "" {
		t.Fatal("request ID is empty")
	}

	req, ok := aw.GetRequest(id)
	if !ok {
		t.Fatal("request not found")
	}
	if req.Status != StatusPending {
		t.Errorf("status = %s, want pending", req.Status)
	}
}

func TestApprove(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	id := aw.RequestApproval("sess-001", "Test reason")

	err := aw.Approve(id, "reviewer-1")
	if err != nil {
		t.Fatalf("Approve error: %v", err)
	}

	req, _ := aw.GetRequest(id)
	if req.Status != StatusApproved {
		t.Errorf("status = %s, want approved", req.Status)
	}
	if req.Reviewer != "reviewer-1" {
		t.Errorf("reviewer = %s, want reviewer-1", req.Reviewer)
	}
}

func TestReject(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	id := aw.RequestApproval("sess-001", "Test reason")

	err := aw.Reject(id, "reviewer-1", "Not compliant")
	if err != nil {
		t.Fatalf("Reject error: %v", err)
	}

	req, _ := aw.GetRequest(id)
	if req.Status != StatusRejected {
		t.Errorf("status = %s, want rejected", req.Status)
	}
	if req.RejectReason != "Not compliant" {
		t.Errorf("reject reason = %s, want Not compliant", req.RejectReason)
	}
}

func TestApprovalNotFound(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	err := aw.Approve("nonexistent", "reviewer")
	if err == nil {
		t.Fatal("expected error for nonexistent request")
	}
}

func TestApprovalNotPending(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	id := aw.RequestApproval("sess-001", "Test reason")
	_ = aw.Approve(id, "reviewer")

	err := aw.Approve(id, "reviewer-2")
	if err == nil {
		t.Fatal("expected error for non-pending request")
	}
}

func TestGetPendingRequests(t *testing.T) {
	aw := NewApprovalWorkflow(1 * time.Minute)
	defer aw.Stop()

	aw.RequestApproval("sess-001", "Reason 1")
	aw.RequestApproval("sess-002", "Reason 2")

	time.Sleep(50 * time.Millisecond)

	pending := aw.GetPendingRequests()
	if len(pending) != 2 {
		t.Fatalf("len(pending) = %d, want 2", len(pending))
	}
}

func TestApprovalTimeout(t *testing.T) {
	aw := NewApprovalWorkflow(100 * time.Millisecond)
	defer aw.Stop()

	id := aw.RequestApproval("sess-001", "Test reason")

	time.Sleep(500 * time.Millisecond)

	req, ok := aw.GetRequest(id)
	if !ok {
		t.Fatal("request not found")
	}
	if req.Status != StatusTimedOut {
		t.Errorf("status = %s, want timed_out", req.Status)
	}
}

func TestDefaultTimeout(t *testing.T) {
	aw := NewApprovalWorkflow(0)
	defer aw.Stop()

	if aw.defaultTimeout != 5*time.Minute {
		t.Errorf("defaultTimeout = %v, want 5m", aw.defaultTimeout)
	}
}
