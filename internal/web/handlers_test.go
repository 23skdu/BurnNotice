package web

import (
	"context"
	"errors"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/a-h/templ"
	templruntime "github.com/a-h/templ/runtime"
)

func TestMiddleware_LoggingAndRequestID(t *testing.T) {
	handler := Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	reqID := w.Header().Get("X-Request-ID")
	if reqID == "" {
		t.Error("X-Request-ID header should be set")
	}
}

func TestMiddleware_PanicRecovery(t *testing.T) {
	handler := Middleware(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		panic("test panic")
	}))

	req := httptest.NewRequest("GET", "/panic", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusInternalServerError {
		t.Errorf("status = %d, want 500", w.Code)
	}
}

func TestHandlers_Dashboard(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	h.ActiveModels = 3
	h.TotalProfiles = 5

	req := httptest.NewRequest("GET", "/", nil)
	w := httptest.NewRecorder()

	h.Dashboard().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "BurnNotice Dashboard", "Active Models", "3", "Total Profiles", "5") {
		t.Error("dashboard HTML missing expected content")
	}
}

func TestHandlers_DashboardSummary(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)
	h.ActiveModels = 2
	h.TotalProfiles = 10

	req := httptest.NewRequest("GET", "/api/summary", nil)
	w := httptest.NewRecorder()

	h.DashboardSummary().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	ct := w.Header().Get("Content-Type")
	if ct != "application/json" {
		t.Errorf("Content-Type = %s, want application/json", ct)
	}

	body := w.Body.String()
	if !containsAll(body, `"active_models":2`, `"total_profiles":10`, `"status":"healthy"`) {
		t.Errorf("summary JSON missing expected fields: %s", body)
	}
}

func TestHandlers_Profiler(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/profiler", nil)
	w := httptest.NewRecorder()

	h.Profiler().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Profiler Status", "Model ID", "Status", "Progress") {
		t.Error("profiler HTML missing expected content")
	}
}

func TestHandlers_ModelProfile(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("GET /model/{id}", h.ModelProfile())

	req := httptest.NewRequest("GET", "/model/test-model", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Model Profile:", "test-model", "Deception", "Truth") {
		t.Error("model profile HTML missing expected content")
	}
}

func TestHandlers_Monitor(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("GET /monitor/{session}", h.Monitor())

	req := httptest.NewRequest("GET", "/monitor/sess-123", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Live Monitor:", "sess-123", "Deception", "Conflict", "Truth", "Compliance") {
		t.Error("monitor HTML missing expected content")
	}
}

func TestHandlers_Heatmap(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("GET /model/{id}/heatmap", h.Heatmap())

	req := httptest.NewRequest("GET", "/model/test-model/heatmap", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Layer x Token Heatmap", "test-model", "<svg") {
		t.Error("heatmap HTML missing expected content")
	}
}

func TestHandlers_Subspace(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("GET /model/{id}/subspace", h.Subspace())

	req := httptest.NewRequest("GET", "/model/test-model/subspace", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Subspace Projection", "test-model", "<svg", "Truth", "Deception") {
		t.Error("subspace HTML missing expected content")
	}
}

func TestHandlers_Circuits(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("GET /model/{id}/circuits", h.Circuits())

	req := httptest.NewRequest("GET", "/model/test-model/circuits", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Neural Circuits", "test-model", "<svg") {
		t.Error("circuits HTML missing expected content")
	}
}

func TestHandlers_Policies(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/policies", nil)
	w := httptest.NewRecorder()

	h.Policies().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Policy Configuration", "Add Policy", "deception_threshold") {
		t.Error("policies HTML missing expected content")
	}
}

func TestHandlers_SavePolicy_GetMethod(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/api/policies", nil)
	w := httptest.NewRecorder()

	h.SavePolicy().ServeHTTP(w, req)

	if w.Code != http.StatusMethodNotAllowed {
		t.Errorf("status = %d, want 405", w.Code)
	}
}

func TestHandlers_SavePolicy_PostMethod(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("POST", "/api/policies", nil)
	w := httptest.NewRecorder()

	h.SavePolicy().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Policy saved successfully") {
		t.Error("save policy response missing success message")
	}
}

func TestHandlers_Alerts(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/alerts", nil)
	w := httptest.NewRecorder()

	h.Alerts().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Alert History", "Timestamp", "Metric", "Score", "Threshold", "Action") {
		t.Error("alerts HTML missing expected content")
	}
}

func TestHandlers_CancelJob(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	mux.Handle("POST /job/{id}/cancel", h.CancelJob())

	req := httptest.NewRequest("POST", "/job/job-123/cancel", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Job job-123 cancelled") {
		t.Error("cancel job response missing expected message")
	}
}

func TestHandlers_RegisterRoutes(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux, "static")

	routes := []struct {
		method string
		path   string
	}{
		{"GET", "/"},
		{"GET", "/api/summary"},
		{"GET", "/profiler"},
		{"GET", "/model/test"},
		{"GET", "/monitor/sess-1"},
		{"GET", "/model/test/heatmap"},
		{"GET", "/model/test/subspace"},
		{"GET", "/model/test/circuits"},
		{"GET", "/policies"},
		{"GET", "/alerts"},
	}

	for _, r := range routes {
		req := httptest.NewRequest(r.method, r.path, nil)
		w := httptest.NewRecorder()
		mux.ServeHTTP(w, req)

		if w.Code == http.StatusNotFound {
			t.Errorf("route %s %s returned 404", r.method, r.path)
		}
	}
}

func TestHandlers_StaticFiles(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	mux := http.NewServeMux()
	h.RegisterRoutes(mux, "static")

	req := httptest.NewRequest("GET", "/static/css/dashboard.css", nil)
	w := httptest.NewRecorder()
	mux.ServeHTTP(w, req)

	if w.Code != http.StatusOK && w.Code != http.StatusNotFound {
		t.Errorf("static file status = %d, want 200 or 404", w.Code)
	}
}

func containsAll(s string, substrs ...string) bool {
	for _, sub := range substrs {
		if !strings.Contains(s, sub) {
			return false
		}
	}
	return true
}

type errorResponseWriter struct {
	header http.Header
}

func (e *errorResponseWriter) Header() http.Header {
	if e.header == nil {
		e.header = make(http.Header)
	}
	return e.header
}

func (e *errorResponseWriter) Write(b []byte) (int, error) {
	return 0, http.ErrBodyNotAllowed
}

func (e *errorResponseWriter) WriteHeader(statusCode int) {}

func TestHandlers_DashboardSummary_Error(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/api/summary", nil)
	w := &errorResponseWriter{}

	h.DashboardSummary().ServeHTTP(w, req)
}

func TestHandlers_LatentSpace(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/model/llama-3-8b/latent", nil)
	req.SetPathValue("id", "llama-3-8b")
	w := httptest.NewRecorder()

	h.LatentSpace().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, "Semantic Latent Space", "llama-3-8b", "Centroid Separation", "Trajectory Curvature") {
		t.Errorf("latent space HTML missing expected elements")
	}
}

func TestHandlers_LatentSpaceSVG(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/api/model/llama-3-8b/latent/svg", nil)
	req.SetPathValue("id", "llama-3-8b")
	w := httptest.NewRecorder()

	h.LatentSpaceSVG().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	contentType := w.Header().Get("Content-Type")
	if !strings.Contains(contentType, "image/svg+xml") {
		t.Errorf("expected Content-Type image/svg+xml, got %s", contentType)
	}

	body := w.Body.String()
	if !strings.Contains(body, "<svg") || !strings.Contains(body, "</svg>") {
		t.Errorf("expected valid SVG output")
	}
}

func TestHandlers_LatentSpaceData(t *testing.T) {
	hub := NewSSEHub()
	h := NewHandlers(hub)

	req := httptest.NewRequest("GET", "/api/model/llama-3-8b/latent/data", nil)
	req.SetPathValue("id", "llama-3-8b")
	w := httptest.NewRecorder()

	h.LatentSpaceData().ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want 200", w.Code)
	}

	body := w.Body.String()
	if !containsAll(body, `"model_id"`, `"variance_explained"`, `"status":"projected"`) {
		t.Errorf("latent data JSON missing expected elements: %s", body)
	}
}

type failAfterNWriter struct {
	remaining int
}

func (f *failAfterNWriter) Write(p []byte) (int, error) {
	if f.remaining <= 0 {
		return 0, errors.New("write error")
	}
	f.remaining--
	return len(p), nil
}

func (f *failAfterNWriter) WriteString(s string) (int, error) {
	if f.remaining <= 0 {
		return 0, errors.New("write string error")
	}
	f.remaining--
	return len(s), nil
}

func TestTemplates_CancelledContextAndWriteErrors(t *testing.T) {
	origSize := templruntime.DefaultBufferSize
	templruntime.DefaultBufferSize = 1
	defer func() {
		templruntime.DefaultBufferSize = origSize
	}()

	components := []templ.Component{
		layout("Test Title"),
		dashboardHTML(10, 20),
		profilerHTML(),
		modelProfileHTML("llama-3-8b"),
		monitorHTML("sess-1"),
		heatmapHTML("llama-3-8b"),
		subspaceHTML("llama-3-8b"),
		circuitsHTML("llama-3-8b"),
		policiesHTML(),
		alertsHTML(),
		latentSpaceHTML("llama-3-8b"),
	}

	cancelledCtx, cancel := context.WithCancel(context.Background())
	cancel()

	for _, comp := range components {
		if err := comp.Render(cancelledCtx, httptest.NewRecorder()); err == nil {
			t.Error("expected error rendering component with cancelled context")
		}
		// Render directly to non-buffer writer that fails on flush/write
		_ = comp.Render(context.Background(), &failAfterNWriter{remaining: 0})

		// Render using 1-byte *templruntime.Buffer failing at each write step
		for step := 0; step <= 25; step++ {
			buf := new(templruntime.Buffer)
			buf.Reset(&failAfterNWriter{remaining: step})
			_ = comp.Render(context.Background(), buf)
		}
	}
}
