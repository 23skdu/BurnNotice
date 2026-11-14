package main

import (
	"net/http"
	"net/http/httptest"
	"os"
	"path/filepath"
	"strings"
	"testing"
	"time"
)

func TestBuildDashboardApp_AllEndpoints(t *testing.T) {
	dir := t.TempDir()
	policyPath := filepath.Join(dir, "policies.yaml")
	if err := os.WriteFile(policyPath, []byte("policies:\n  - name: p1\n    metric: deception\n    threshold: 0.7\n    action: steer\n"), 0600); err != nil {
		t.Fatal(err)
	}
	ckptPath := filepath.Join(dir, "ckpt.json")

	t.Setenv("BURNNOTICE_PROFILES_DIR", dir)
	t.Setenv("BURNNOTICE_POLICY_FILE", policyPath)
	t.Setenv("BURNNOTICE_CHECKPOINT_PATH", ckptPath)
	t.Setenv("BURNNOTICE_METRIC_SAMPLE_RATE", "1")
	t.Setenv("BURNNOTICE_LEDGER_DSN", ":memory:")

	app := buildDashboardApp()
	defer app.closeFn()

	getRoutes := []string{
		"/",
		"/api/summary",
		"/metrics",
		"/api/compliance/report",
		"/api/compliance/report?format=pdf",
		"/api/approvals",
		"/api/profiles",
		"/api/jobs",
		"/api/alerts/export",
		"/api/model/test-model/latent/data",
		"/api/model/test-model/latent/svg",
	}
	for _, path := range getRoutes {
		req := httptest.NewRequest(http.MethodGet, path, nil)
		w := httptest.NewRecorder()
		app.handler.ServeHTTP(w, req)
		if w.Code != http.StatusOK {
			t.Errorf("GET %s status = %d, want 200", path, w.Code)
		}
	}

	postReq := httptest.NewRequest(http.MethodPost, "/api/score", strings.NewReader(`{"session_id":"test","prompt":"hello","activation":[0.1,0.2,0.3,0.4]}`))
	postW := httptest.NewRecorder()
	app.handler.ServeHTTP(postW, postReq)
	if postW.Code != http.StatusOK {
		t.Errorf("POST /api/score status = %d, want 200", postW.Code)
	}
}

func TestMainDashboard(t *testing.T) {
	// Start the server in a goroutine
	_ = os.Setenv("PORT", "8081") // Use a different port
	defer func() { _ = os.Unsetenv("PORT") }()

	go func() {
		// Mock os.Exit or panic to prevent actual fatal on failure
		defer func() {
			_ = recover()
		}()
		main()
	}()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)
}
