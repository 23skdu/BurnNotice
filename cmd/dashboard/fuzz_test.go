package main

import (
	"context"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"burnnotice/internal/web"
)

func FuzzDashboardHandler(f *testing.F) {
	f.Add("GET", "/api/alerts")
	f.Add("GET", "/api/stats")
	f.Add("GET", "/health")
	f.Add("POST", "/unknown")

	f.Fuzz(func(t *testing.T, method, path string) {
		if len(method) == 0 || len(method) > 16 || len(path) == 0 || len(path) > 256 {
			return
		}
		if path[0] != '/' {
			path = "/" + path
		}
		if strings.HasPrefix(path, "/events") {
			return
		}

		ctx, cancel := context.WithCancel(context.Background())
		cancel()
		req, err := http.NewRequestWithContext(ctx, method, path, nil)
		if err != nil {
			return
		}

		hub := web.NewSSEHub()
		handlers := web.NewHandlers(hub)
		mux := http.NewServeMux()
		handlers.RegisterRoutes(mux, "static")
		handler := web.Middleware(mux)

		rr := httptest.NewRecorder()
		handler.ServeHTTP(rr, req)
		if rr.Code == 0 {
			t.Error("expected non-zero HTTP status code")
		}
	})
}
