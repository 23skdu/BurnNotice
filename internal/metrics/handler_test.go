package metrics

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
)

func TestHandler_ServesPrometheusMetrics(t *testing.T) {
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", w.Code, http.StatusOK)
	}

	body := w.Body.String()
	if !strings.Contains(body, "# HELP") {
		t.Error("response body does not contain Prometheus HELP lines")
	}
	if !strings.Contains(body, "# TYPE") {
		t.Error("response body does not contain Prometheus TYPE lines")
	}
	if !strings.Contains(body, "burnnotice_") {
		t.Error("response body does not contain burnnotice_ metric prefix")
	}
}

func TestHandler_ContentType(t *testing.T) {
	handler := Handler()
	req := httptest.NewRequest("GET", "/metrics", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	ct := w.Header().Get("Content-Type")
	if ct == "" {
		t.Error("Content-Type header is empty, want prometheus text format")
	}
}

func TestHandler_AllMethods(t *testing.T) {
	for _, method := range []string{"GET", "POST", "HEAD"} {
		handler := Handler()
		req := httptest.NewRequest(method, "/metrics", nil)
		w := httptest.NewRecorder()

		handler.ServeHTTP(w, req)

		if w.Code != http.StatusOK {
			t.Errorf("%s /metrics returned %d, want 200", method, w.Code)
		}
	}
}

func TestHTTPMiddleware(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
		_, _ = w.Write([]byte("ok"))
	})

	handler := HTTPMiddleware(inner)

	req := httptest.NewRequest("GET", "/test", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Errorf("status = %d, want %d", w.Code, http.StatusOK)
	}
	if w.Body.String() != "ok" {
		t.Errorf("body = %q, want %q", w.Body.String(), "ok")
	}
}

func TestHTTPMiddleware_RecordsDuration(t *testing.T) {
	inner := http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	})

	handler := HTTPMiddleware(inner)

	req := httptest.NewRequest("GET", "/test-path", nil)
	w := httptest.NewRecorder()

	handler.ServeHTTP(w, req)

	h := Handler()
	mreq := httptest.NewRequest("GET", "/metrics", nil)
	mw := httptest.NewRecorder()
	h.ServeHTTP(mw, mreq)

	body := mw.Body.String()
	if !strings.Contains(body, "burnnotice_http_request_duration_seconds") {
		t.Error("metrics output should contain request duration metric after middleware call")
	}
}
