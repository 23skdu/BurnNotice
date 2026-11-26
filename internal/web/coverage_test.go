package web

import (
	"context"
	"net/http"
	"net/http/httptest"
	"testing"

	"burnnotice/internal/compliance"
	"burnnotice/internal/ledger"
	"burnnotice/internal/policy"
	"burnnotice/internal/registry"
	"burnnotice/internal/scheduler"
	"burnnotice/internal/scoring"
)

func TestHandlersNilDependencies(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	cases := []struct {
		method string
		path   string
	}{
		{http.MethodGet, "/"},
		{http.MethodGet, "/api/dashboard/summary"},
		{http.MethodGet, "/api/model/x/latent"},
		{http.MethodGet, "/api/compliance/report"},
		{http.MethodGet, "/api/compliance/report?format=pdf"},
		{http.MethodGet, "/api/approvals"},
		{http.MethodPost, "/api/approvals/abc/approve"},
		{http.MethodPost, "/api/approvals/abc/reject"},
		{http.MethodGet, "/api/profiles"},
		{http.MethodGet, "/api/alerts/export"},
	}
	for _, tc := range cases {
		req := httptest.NewRequest(tc.method, tc.path, nil)
		w := httptest.NewRecorder()
		// Route through the handlers directly.
		switch tc.path {
		case "/":
			h.Dashboard()(w, req)
		case "/api/dashboard/summary":
			h.DashboardSummary()(w, req)
		case "/api/model/x/latent":
			h.LatentSpaceData()(w, req)
		case "/api/compliance/report":
			h.ComplianceReportHandler()(w, req)
		case "/api/compliance/report?format=pdf":
			h.ComplianceReportHandler()(w, req)
		case "/api/approvals":
			h.ApprovalsHandler()(w, req)
		case "/api/approvals/abc/approve":
			h.ApproveRequestHandler(true)(w, req)
		case "/api/approvals/abc/reject":
			h.ApproveRequestHandler(false)(w, req)
		case "/api/profiles":
			h.ProfilesHandler()(w, req)
		case "/api/alerts/export":
			h.ExportAlertsCSVHandler()(w, req)
		}
		_ = w
	}

	// Direct nil-dependency calls for handlers not reached by the table above.
	h2 := NewHandlers(NewSSEHub())
	r := httptest.NewRequest("GET", "/api/compliance/report", nil)
	wr := httptest.NewRecorder()
	h2.ComplianceReportHandler()(wr, r)
	r2 := httptest.NewRequest("GET", "/api/model/x/latent/data", nil)
	r2.SetPathValue("id", "x")
	wr2 := httptest.NewRecorder()
	h2.LatentSpaceData()(wr2, r2)
}

func TestHandlersComplianceAndApprovalsErrors(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	// Compliance configured but report generation uses a real reporter.
	ledg, err := ledger.NewBackend("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := ledg.Init(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ledg.Close() }()
	h.Compliance = compliance.NewReporter(ledg)

	req := httptest.NewRequest(http.MethodGet, "/api/compliance/report", nil)
	w := httptest.NewRecorder()
	h.ComplianceReportHandler()(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("compliance report code = %d, want 200", w.Code)
	}

	// PDF export path.
	req2 := httptest.NewRequest(http.MethodGet, "/api/compliance/report?format=pdf", nil)
	w2 := httptest.NewRecorder()
	h.ComplianceReportHandler()(w2, req2)
	if w2.Code != http.StatusOK {
		t.Errorf("compliance pdf code = %d, want 200 (got %s)", w2.Code, w2.Body.String())
	}
}

func TestHandlersLatentSpaceDataBranches(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	// No registry: nil branch.
	req := httptest.NewRequest(http.MethodGet, "/api/model/x/latent", nil)
	w := httptest.NewRecorder()
	h.LatentSpaceData()(w, req)
	// Just ensure no panic; status may vary.
	_ = w.Code
}

func TestHandlersExportAlertsCSVNilLedger(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	req := httptest.NewRequest(http.MethodGet, "/api/alerts/export", nil)
	w := httptest.NewRecorder()
	h.ExportAlertsCSVHandler()(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("code = %d, want 503", w.Code)
	}
}

func TestHandlersDashboardSummaryBranches(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	// With scheduler nil, active/total fall back to stored values (0).
	h.ActiveModels = 2
	h.TotalProfiles = 5
	req := httptest.NewRequest(http.MethodGet, "/api/dashboard/summary", nil)
	w := httptest.NewRecorder()
	h.DashboardSummary()(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("code = %d", w.Code)
	}
}

func TestHandlersApprovalsNilBranch(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	for _, approve := range []bool{true, false} {
		req := httptest.NewRequest(http.MethodPost, "/api/approvals/abc/approve", nil)
		w := httptest.NewRecorder()
		h.ApproveRequestHandler(approve)(w, req)
		if w.Code != http.StatusServiceUnavailable {
			t.Errorf("approve=%v code = %d, want 503", approve, w.Code)
		}
	}
	// ApprovalsHandler nil branch.
	req := httptest.NewRequest(http.MethodGet, "/api/approvals", nil)
	w := httptest.NewRecorder()
	h.ApprovalsHandler()(w, req)
	if w.Code != http.StatusServiceUnavailable {
		t.Errorf("code = %d, want 503", w.Code)
	}
}

func TestHandlersProfilesNilBranch(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	req := httptest.NewRequest(http.MethodGet, "/api/profiles", nil)
	w := httptest.NewRecorder()
	h.ProfilesHandler()(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("code = %d", w.Code)
	}
}

func TestHandlersWriteErrors(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	h.ActiveModels = 2
	h.TotalProfiles = 5

	w := &errorResponseWriter{}
	req := httptest.NewRequest("GET", "/api/dashboard/summary", nil)
	h.DashboardSummary()(w, req)

	// LatentSpaceData write path (success branch returns after Write).
	req2 := httptest.NewRequest("GET", "/api/model/x/latent/data", nil)
	req2.SetPathValue("id", "x")
	w2 := &errorResponseWriter{}
	h.LatentSpaceData()(w2, req2)
}

func TestHandlersComplianceWritePaths(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	ledg, err := ledger.NewBackend("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := ledg.Init(); err != nil {
		t.Fatal(err)
	}
	defer func() { _ = ledg.Close() }()
	h.Compliance = compliance.NewReporter(ledg)

	// JSON write error.
	w := &errorResponseWriter{}
	req := httptest.NewRequest("GET", "/api/compliance/report", nil)
	h.ComplianceReportHandler()(w, req)

	// PDF write error.
	w2 := &errorResponseWriter{}
	req2 := httptest.NewRequest("GET", "/api/compliance/report?format=pdf", nil)
	h.ComplianceReportHandler()(w2, req2)
}

func TestHandlersDashboardSummaryDeps(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	// ActiveModels=0, TotalProfiles=0 with deps set -> exercises scheduler/registry branches.
	sched := scheduler.NewScheduler(1024)
	h.Scheduler = sched
	reg := registry.NewRegistry(t.TempDir())
	h.Registry = reg
	ks, _ := policy.NewKillSwitch(nil)
	h.KillSwitch = ks
	sc := scoring.NewScorer(nil)
	as := scoring.NewAsyncScorer(sc, 32, nil)
	ctx, cancel := context.WithCancel(context.Background())
	defer cancel()
	as.Start(ctx, 1)
	h.AsyncScorer = as

	req := httptest.NewRequest("GET", "/api/dashboard/summary", nil)
	w := httptest.NewRecorder()
	h.DashboardSummary().ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Errorf("code = %d", w.Code)
	}
}

func TestHandlersDashboardDeps(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	h.Scheduler = scheduler.NewScheduler(1024)
	h.Registry = registry.NewRegistry(t.TempDir())
	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()
	h.Dashboard()(w, req)
}

func TestHandlersComplianceClosedLedger(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	ledg, err := ledger.NewBackend("sqlite", ":memory:")
	if err != nil {
		t.Fatal(err)
	}
	if err := ledg.Init(); err != nil {
		t.Fatal(err)
	}
	h.Compliance = compliance.NewReporter(ledg)
	_ = ledg.Close() // force GenerateReport to fail

	req := httptest.NewRequest("GET", "/api/compliance/report", nil)
	w := httptest.NewRecorder()
	h.ComplianceReportHandler()(w, req)
	if w.Code != http.StatusInternalServerError {
		t.Errorf("code = %d, want 500", w.Code)
	}
}
