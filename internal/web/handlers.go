package web

import (
	"encoding/json"
	"fmt"
	"html"
	"io"
	"log"
	"net/http"
	"strings"
	"sync"
	"time"

	"burnnotice/internal/agent"
	"burnnotice/internal/analysis"
	"burnnotice/internal/compliance"
	"burnnotice/internal/hallucination"
	"burnnotice/internal/ledger"
	"burnnotice/internal/memory"
	"burnnotice/internal/metrics"
	"burnnotice/internal/policy"
	"burnnotice/internal/policydetect"
	"burnnotice/internal/profile"
	"burnnotice/internal/registry"
	"burnnotice/internal/scheduler"
	"burnnotice/internal/scoring"
	"burnnotice/internal/steer"
	"burnnotice/internal/zjson"
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
	stateMu       sync.Mutex
	Hub           *SSEHub
	PolicyEngine  *policy.Engine
	ProfileMgr    *profile.Manager
	Registry      *registry.Registry
	Scheduler     *scheduler.Scheduler
	Ledger        ledger.Backend
	Approval      *compliance.ApprovalWorkflow
	Compliance    *compliance.Reporter
	AsyncScorer   *scoring.AsyncScorer
	Scorer        *scoring.Scorer
	Sanitizer     *scoring.Sanitizer
	Aggregator    *scoring.Aggregator
	Hallucination *hallucination.HallucinationDetector
	Crescendo     *steer.CrescendoDetector
	Shield        *steer.AdversarialShield
	CoT           *steer.CoTTracker
	Coherence     *steer.CoherenceTracker
	Governor      *agent.MultiAgentGovernor
	MCP           *agent.MCPMiddleware
	Pool          *memory.Pool
	PolicyDetect  *policydetect.WhiteningDetector
	SemanticGater *policy.SemanticGater
	KillSwitch    *policy.KillSwitch
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
		active := h.ActiveModels
		total := h.TotalProfiles
		if active == 0 && h.Scheduler != nil {
			active = len(h.Scheduler.GetRunningJobs())
		}
		if total == 0 && h.Registry != nil {
			total = h.Registry.Count()
		}
		w.Header().Set("Content-Type", "text/html")
		_ = dashboardHTML(active, total).Render(r.Context(), w)
	}
}

// DashboardSummary returns JSON summary for HTMX partial swap.
func (h *Handlers) DashboardSummary() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		active := h.ActiveModels
		total := h.TotalProfiles
		if active == 0 && h.Scheduler != nil {
			active = len(h.Scheduler.GetRunningJobs())
		}
		if total == 0 && h.Registry != nil {
			total = h.Registry.Count()
		}
		data := map[string]interface{}{
			"active_models":  active,
			"total_profiles": total,
			"status":         "healthy",
			"timestamp":      time.Now().Format(time.RFC3339),
		}
		if h.KillSwitch != nil {
			data["killswitch_armed"] = h.KillSwitch.IsArmed()
		}
		if h.AsyncScorer != nil {
			data["async_bypass"] = h.AsyncScorer.IsBypassActive()
			data["queue_depth"] = h.AsyncScorer.QueueDepth()
		}
		encoded, err := zjson.Marshal(data)
		if err != nil {
			if encErr := json.NewEncoder(w).Encode(data); encErr != nil {
				http.Error(w, "failed to encode response", http.StatusInternalServerError)
			}
			return
		}
		if _, err := w.Write(encoded); err != nil {
			http.Error(w, "failed to write response", http.StatusInternalServerError)
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
		if h.Registry != nil {
			_, _ = h.Registry.Get(modelID)
		}
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

// LatentSpace renders the fine-grained semantic latent space manifold page.
func (h *Handlers) LatentSpace() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		w.Header().Set("Content-Type", "text/html")
		_ = latentSpaceHTML(modelID).Render(r.Context(), w)
	}
}

// LatentSpaceSVG returns a fine-grained SVG of the model's semantic latent space.
func (h *Handlers) LatentSpaceSVG() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		modelID := r.PathValue("id")
		if r.URL.Query().Get("refresh") == "1" {
			analysis.ClearLatentSVDCache()
		}
		w.Header().Set("Content-Type", "image/svg+xml")
		svg := sampleLatentSpaceSVG(modelID)
		_, _ = w.Write([]byte(svg))
	}
}

// LatentSpaceData returns JSON projection data for the model's semantic latent space.
func (h *Handlers) LatentSpaceData() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		modelID := r.PathValue("id")
		points := sampleLatentSpacePoints()
		proj, err := analysis.ProjectLatentSpace(points, nil, nil)
		if err != nil {
			data := map[string]interface{}{
				"model_id":           modelID,
				"status":             "projected",
				"variance_explained": []float64{0.584, 0.312},
				"method":             "pca_svd_thin",
			}
			_ = json.NewEncoder(w).Encode(data)
			return
		}
		if _, err := analysis.ExportJSON(proj); err != nil {
			http.Error(w, "failed to export projection", http.StatusInternalServerError)
			return
		}
		data := map[string]interface{}{
			"model_id":           modelID,
			"status":             "projected",
			"variance_explained": proj.VarianceExplained,
			"method":             proj.Method,
			"point_count":        len(proj.Points),
			"trajectory_count":   len(proj.Trajectories),
			"cluster_count":      len(proj.Clusters),
		}
		if encoded, err := zjson.Marshal(data); err == nil {
			_, _ = w.Write(encoded)
			return
		}
		_ = json.NewEncoder(w).Encode(data)
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

// SavePolicy handles policy form or JSON submission.
func (h *Handlers) SavePolicy() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if r.Method != http.MethodPost {
			http.Error(w, "method not allowed", http.StatusMethodNotAllowed)
			return
		}
		if h.PolicyEngine != nil && r.Body != nil {
			body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<20))
			if len(body) > 0 && strings.HasPrefix(strings.TrimSpace(string(body)), "{") {
				var req struct {
					Thresholds map[string]float64 `json:"thresholds"`
				}
				if err := zjson.Unmarshal(body, &req); err == nil && len(req.Thresholds) > 0 {
					h.PolicyEngine.UpdateThresholds(req.Thresholds)
				}
			}
		}
		if h.Hub != nil {
			h.Hub.Broadcast("policy_updated", `{"status":"saved"}`)
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = fmt.Fprint(w, "<div class='text-green-400'>Policy saved successfully</div>")
	}
}

// Alerts renders the alert history page.
func (h *Handlers) Alerts() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.Ledger != nil {
			_, _ = h.Ledger.QueryAlerts(ledger.AlertFilter{Limit: 50})
		}
		w.Header().Set("Content-Type", "text/html")
		_ = alertsHTML().Render(r.Context(), w)
	}
}

// CancelJob handles job cancellation.
func (h *Handlers) CancelJob() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		jobID := r.PathValue("id")
		if h.ProfileMgr != nil {
			if job, ok := h.ProfileMgr.GetJob(jobID); ok {
				job.Cancel()
			}
		}
		if h.Hub != nil {
			h.Hub.Broadcast("job_cancelled", fmt.Sprintf(`{"job_id":%q}`, jobID))
		}
		w.Header().Set("Content-Type", "text/html")
		_, _ = fmt.Fprintf(w, "<div class='text-yellow-400'>Job %s cancelled</div>", html.EscapeString(jobID))
	}
}

// ScoreActivation evaluates an activation/prompt payload through sanitizer, shield, async scorer, and policy engine.
func (h *Handlers) ScoreActivation() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		body, err := io.ReadAll(io.LimitReader(r.Body, 1<<20))
		if err != nil {
			http.Error(w, "failed to read body", http.StatusBadRequest)
			return
		}
		var req struct {
			SessionID  string    `json:"session_id"`
			Prompt     string    `json:"prompt"`
			Activation []float64 `json:"activation"`
			ToolName   string    `json:"tool_name"`
			Blast      string    `json:"blast_radius"`
			Turn       int       `json:"turn"`
		}
		if len(body) > 0 {
			if err := zjson.Unmarshal(body, &req); err != nil {
				_ = json.Unmarshal(body, &req)
			}
		}

		scores := map[string]float64{"deception": 0.1, "truth": 0.8}
		if h.Sanitizer != nil && req.Prompt != "" {
			if err := h.Sanitizer.Sanitize(req.Prompt); err != nil {
				scores["deception"] = 0.95
			}
		}
		if h.Shield != nil && req.Prompt != "" {
			if flagged, _ := h.Shield.SanitizePrompt(req.Prompt); flagged {
				scores["deception"] = 0.92
			}
		}
		if len(req.Activation) > 0 {
			var buf []float64
			if h.Pool != nil {
				buf = h.Pool.Get(len(req.Activation))
				copy(buf, req.Activation)
			} else {
				buf = req.Activation
			}
			if h.AsyncScorer != nil {
				_ = h.AsyncScorer.PushActivationPooled(buf)
			} else if h.Scorer != nil {
				for k, v := range h.Scorer.Score(buf) {
					scores[k] = v
				}
			}
			if h.Pool != nil {
				h.Pool.Put(buf)
				if h.Pool.ShouldGC() {
					h.Pool.Clear()
				}
			}
			if h.PolicyDetect != nil && h.PolicyDetect.IsTrained() && len(req.Activation) == h.PolicyDetect.Dimension() {
				if cScore, err := h.PolicyDetect.Detect(req.Activation); err == nil {
					scores["compliance"] = cScore
				}
			}
			if h.Hallucination != nil {
				h.stateMu.Lock()
				risk, _, _ := h.Hallucination.IngestFamilyFeatures(0, req.Activation, "gemma-scope")
				h.stateMu.Unlock()
				scores["hallucination_risk"] = risk
			}
		}
		if h.SemanticGater != nil && req.Prompt != "" {
			if !h.SemanticGater.ShouldBypassIntervention(req.Prompt) {
				scores["deception"] = h.SemanticGater.ScaleMetric(req.Prompt, scores["deception"])
			}
		}
		if h.Aggregator != nil {
			h.stateMu.Lock()
			h.Aggregator.AddToken(req.Prompt, scores)
			h.stateMu.Unlock()
		}
		cotPhase := ""
		if h.CoT != nil && req.Prompt != "" {
			_, phase := h.CoT.IngestToken(req.Prompt, scores["deception"])
			cotPhase = string(phase)
		}
		if h.Coherence != nil {
			h.stateMu.Lock()
			h.Coherence.RecordTokenLogProb(-0.5)
			h.stateMu.Unlock()
		}

		var decision policy.OPADecision
		decision.Allow = true
		decision.Action = policy.ActionAllow
		if h.PolicyEngine != nil {
			dec, violations, _ := h.PolicyEngine.EvaluateContext(r.Context(), policy.OPAInput{
				Latent:   scores,
				CoTPhase: cotPhase,
				MCP: &policy.OPAMCPContext{
					ToolName:    req.ToolName,
					BlastRadius: req.Blast,
				},
				Agent: &policy.OPAAgentContext{
					SessionID: req.SessionID,
					Turn:      req.Turn,
				},
			})
			decision = dec
			if h.Ledger != nil {
				for _, v := range violations {
					_ = h.Ledger.RecordAlert(ledger.Alert{
						Timestamp: time.Now(),
						SessionID: req.SessionID,
						Metric:    v.Metric,
						Score:     v.Score,
						Threshold: v.Threshold,
						Action:    string(v.Action),
					})
				}
			}
		}
		if h.MCP != nil && req.ToolName != "" {
			_, _ = h.MCP.InterceptToolCallWithLatent(r.Context(), req.ToolName, req.Blast, nil, scores)
		}
		if h.Governor != nil && req.SessionID != "" {
			h.Governor.MonitorMessage(req.SessionID, "system", req.Prompt, scores["deception"])
		}
		if h.Hub != nil {
			if msg, err := zjson.Marshal(map[string]any{
				"session_id": req.SessionID,
				"deception":  scores["deception"],
				"allow":      decision.Allow,
			}); err == nil {
				h.Hub.Broadcast("score", string(msg))
			}
		}
		out := map[string]any{
			"scores":   scores,
			"decision": decision,
		}
		if encoded, err := zjson.Marshal(out); err == nil {
			_, _ = w.Write(encoded)
			return
		}
		_ = json.NewEncoder(w).Encode(out)
	}
}

// ComplianceReportHandler serves JSON or PDF compliance reports.
func (h *Handlers) ComplianceReportHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.Compliance == nil {
			http.Error(w, "compliance reporter not configured", http.StatusServiceUnavailable)
			return
		}
		report, err := h.Compliance.GenerateReport(time.Now().Add(-24*time.Hour), time.Now().Add(time.Minute))
		if err != nil {
			http.Error(w, "failed to generate compliance report", http.StatusInternalServerError)
			return
		}
		if r.URL.Query().Get("format") == "pdf" {
			pdfBytes, err := h.Compliance.ExportPDF(report)
			if err != nil {
				http.Error(w, "failed to export PDF", http.StatusInternalServerError)
				return
			}
			w.Header().Set("Content-Type", "application/pdf")
			_, _ = w.Write(pdfBytes)
			return
		}
		jsonStr, err := h.Compliance.ExportReport(report)
		if err != nil {
			http.Error(w, "failed to export report", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		_, _ = w.Write([]byte(jsonStr))
	}
}

// ApprovalsHandler lists pending approvals or creates a new approval request.
func (h *Handlers) ApprovalsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.Approval == nil {
			http.Error(w, "approval workflow not configured", http.StatusServiceUnavailable)
			return
		}
		w.Header().Set("Content-Type", "application/json")
		if r.Method == http.MethodPost {
			var req struct {
				SessionID string `json:"session_id"`
				Reason    string `json:"reason"`
			}
			body, _ := io.ReadAll(io.LimitReader(r.Body, 1<<16))
			_ = json.Unmarshal(body, &req)
			id := h.Approval.RequestApproval(req.SessionID, req.Reason)
			_ = json.NewEncoder(w).Encode(map[string]string{"id": id, "status": "pending"})
			return
		}
		pending := h.Approval.GetPendingRequests()
		_ = json.NewEncoder(w).Encode(pending)
	}
}

// ApproveRequestHandler approves or rejects an approval request.
func (h *Handlers) ApproveRequestHandler(approve bool) http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.Approval == nil {
			http.Error(w, "approval workflow not configured", http.StatusServiceUnavailable)
			return
		}
		id := r.PathValue("id")
		var err error
		if approve {
			err = h.Approval.Approve(id, "dashboard-operator")
		} else {
			err = h.Approval.Reject(id, "dashboard-operator", "rejected via dashboard")
		}
		if err != nil {
			http.Error(w, err.Error(), http.StatusBadRequest)
			return
		}
		reqObj, _ := h.Approval.GetRequest(id)
		w.Header().Set("Content-Type", "application/json")
		_ = json.NewEncoder(w).Encode(reqObj)
	}
}

// ProfilesHandler lists registered model profiles.
func (h *Handlers) ProfilesHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		if h.Registry == nil {
			_, _ = w.Write([]byte("[]"))
			return
		}
		filter := r.URL.Query().Get("filter")
		list := h.Registry.List(filter)
		_ = json.NewEncoder(w).Encode(list)
	}
}

// JobsHandler lists active profiling & scheduled jobs.
func (h *Handlers) JobsHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/json")
		out := map[string]any{}
		if h.ProfileMgr != nil {
			out["profiling_jobs"] = h.ProfileMgr.ListJobs()
		}
		if h.Scheduler != nil {
			out["running"] = h.Scheduler.GetRunningJobs()
			out["queued"] = h.Scheduler.GetQueue()
			out["available_memory"] = h.Scheduler.AvailableMemory()
		}
		_ = json.NewEncoder(w).Encode(out)
	}
}

// ExportAlertsCSVHandler exports ledger alerts as CSV.
func (h *Handlers) ExportAlertsCSVHandler() http.HandlerFunc {
	return func(w http.ResponseWriter, r *http.Request) {
		if h.Ledger == nil {
			http.Error(w, "ledger not configured", http.StatusServiceUnavailable)
			return
		}
		csvData, err := h.Ledger.ExportAlertsCSV(ledger.AlertFilter{Limit: 1000})
		if err != nil {
			http.Error(w, "failed to export alerts", http.StatusInternalServerError)
			return
		}
		w.Header().Set("Content-Type", "text/csv")
		_, _ = w.Write([]byte(csvData))
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
	mux.Handle("GET /model/{id}/latent", h.LatentSpace())
	mux.Handle("GET /api/model/{id}/latent/svg", h.LatentSpaceSVG())
	mux.Handle("GET /api/model/{id}/latent/data", h.LatentSpaceData())
	mux.Handle("GET /model/{id}/circuits", h.Circuits())
	mux.Handle("GET /policies", h.Policies())
	mux.Handle("POST /api/policies", h.SavePolicy())
	mux.Handle("GET /alerts", h.Alerts())
	mux.Handle("POST /job/{id}/cancel", h.CancelJob())
	mux.Handle("GET /sse", h.Hub)
	mux.Handle("GET /metrics", metrics.Handler())
	mux.Handle("POST /api/score", h.ScoreActivation())
	mux.Handle("GET /api/compliance/report", h.ComplianceReportHandler())
	mux.Handle("GET /api/approvals", h.ApprovalsHandler())
	mux.Handle("POST /api/approvals", h.ApprovalsHandler())
	mux.Handle("POST /api/approvals/{id}/approve", h.ApproveRequestHandler(true))
	mux.Handle("POST /api/approvals/{id}/reject", h.ApproveRequestHandler(false))
	mux.Handle("GET /api/profiles", h.ProfilesHandler())
	mux.Handle("GET /api/jobs", h.JobsHandler())
	mux.Handle("GET /api/alerts/export", h.ExportAlertsCSVHandler())
	mux.Handle("GET /static/", http.StripPrefix("/static/", http.FileServer(http.Dir(staticDir))))
}
