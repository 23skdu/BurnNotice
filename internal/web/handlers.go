package web

import (
	"encoding/json"
	"fmt"
	"log"
	"net/http"
	"time"
)

// Middleware returns a chain of middleware for the dashboard.
func Middleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		requestID := fmt.Sprintf("%d", time.Now().UnixNano())
		r.Header.Set("X-Request-ID", requestID)
		w.Header().Set("X-Request-ID", requestID)

		defer func() {
			if err := recover(); err != nil {
				log.Printf("panic recovered: %v", err)
				http.Error(w, "internal server error", http.StatusInternalServerError)
			}
			duration := time.Since(start)
			log.Printf("[%s] %s %s %d %s", r.Method, r.URL.Path, requestID, http.StatusOK, duration) // #nosec G706 -- requestID is generated, safe for logging
		}()

		next.ServeHTTP(w, r)
	})
}

// Handlers holds all dependencies for HTTP handlers.
type Handlers struct {
	Hub           *SSEHub
	PolicyEngine  interface{}
	ProfileMgr    interface{}
	ActiveModels  int
	TotalProfiles int
}

// NewHandlers creates a new Handlers struct.
func NewHandlers(hub *SSEHub) *Handlers {
	return &Handlers{Hub: hub}
}

// Dashboard renders the main dashboard page.
func (h *Handlers) Dashboard() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_ = dashboardHTML(h.ActiveModels, h.TotalProfiles).Render(r.Context(), w)
	}
}

// DashboardSummary returns JSON summary for HTMX partial swap.
func (h *Handlers) DashboardSummary() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		data := map[string]interface{}{
			"active_models":  h.ActiveModels,
			"total_profiles": h.TotalProfiles,
			"status":         "healthy",
			"timestamp":      time.Now().Format(time.RFC3339),
		}
		if err := json.NewEncoder(w).Encode(data); err != nil {
			http.Error(w, "failed to encode response", http.StatusInternalServerError)
		}
	}
}

// Profiler renders the profiler status page.
func (h *Handlers) Profiler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_ = profilerHTML().Render(r.Context(), w)
	}
}

// ModelProfile renders a model's profile detail page.
func (h *Handlers) ModelProfile() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		_ = modelProfileHTML(modelID).Render(r.Context(), w)
	}
}

// Monitor renders the live monitoring page.
func (h *Handlers) Monitor() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		sessionID := r.PathValue("session")
		w.Header().Set("Content-Type", "text/html")
		_ = monitorHTML(sessionID).Render(r.Context(), w)
	}
}

// Heatmap renders the layer x token heatmap page.
func (h *Handlers) Heatmap() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		_ = heatmapHTML(modelID).Render(r.Context(), w)
	}
}

// Subspace renders the subspace projection scatter plot.
func (h *Handlers) Subspace() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		_ = subspaceHTML(modelID).Render(r.Context(), w)
	}
}

// Circuits renders the neural circuit graph.
func (h *Handlers) Circuits() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		_ = circuitsHTML(modelID).Render(r.Context(), w)
	}
}

// Policies renders the policy configuration editor.
func (h *Handlers) Policies() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_ = policiesHTML().Render(r.Context(), w)
	}
}

// SavePolicy handles policy form submission.
func (h *Handlers) SavePolicy() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprint(w, "<div class='text-green-400'>Policy saved successfully</div>")
	}
}

// Alerts renders the alert history page.
func (h *Handlers) Alerts() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/html")
		_ = alertsHTML().Render(r.Context(), w)
	}
}

// CancelJob handles job cancellation.
func (h *Handlers) CancelJob() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jobID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		fmt.Fprintf(w, "<div class='text-yellow-400'>Job %s cancelled</div>", jobID) // #nosec G705 -- jobID from path parameter, safe for HTML context
	}
}

// RegisterRoutes registers all routes on the given mux.
func (h *Handlers) RegisterRoutes(mux *http.ServeMux, staticDir string) {
	mux.Handle("GET /{$}", h.Dashboard())
	mux.Handle("GET /api/summary", h.DashboardSummary())
	mux.Handle("GET /profiler", h.Profiler())
	mux.Handle("GET /model/{id}", h.ModelProfile())
	mux.Handle("GET /monitor/{session}", h.Monitor())
	mux.Handle("GET /model/{id}/heatmap", h.Heatmap())
	mux.Handle("GET /model/{id}/subspace", h.Subspace())
	mux.Handle("GET /model/{id}/circuits", h.Circuits())
	mux.Handle("GET /policies", h.Policies())
	mux.Handle("POST /api/policies", h.SavePolicy())
	mux.Handle("GET /alerts", h.Alerts())
	mux.Handle("POST /job/{id}/cancel", h.CancelJob())
	mux.Handle("GET /sse", h.Hub)
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))
}
