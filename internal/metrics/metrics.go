package metrics

import (
	"github.com/prometheus/client_golang/prometheus"
	"github.com/prometheus/client_golang/prometheus/promauto"
)

var (
	ActivationsProcessedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_activations_processed_total",
			Help: "Total number of activations processed",
		},
		[]string{"model_id", "layer"},
	)

	ScoringDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_scoring_duration_seconds",
			Help:    "Duration of scoring operations",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"model_id", "direction"},
	)

	DeceptionScoreCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_deception_score_current",
			Help: "Current deception risk score (0-1)",
		},
		[]string{"model_id", "session_id"},
	)

	ConflictScoreCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_conflict_score_current",
			Help: "Current conflict intensity score (0-1)",
		},
		[]string{"model_id", "session_id"},
	)

	TruthScoreCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_truth_score_current",
			Help: "Current truth confidence score (0-1)",
		},
		[]string{"model_id", "session_id"},
	)

	PolicyComplianceCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_policy_compliance_current",
			Help: "Current policy compliance score (0-1)",
		},
		[]string{"model_id", "session_id"},
	)

	SteeringInterventionsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_steering_interventions_total",
			Help: "Total number of steering interventions applied",
		},
		[]string{"model_id", "method", "layer"},
	)

	AlertsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_alerts_total",
			Help: "Total number of alerts triggered",
		},
		[]string{"model_id", "metric", "action"},
	)

	ProfilingBatchesCompletedTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_profiling_batches_completed_total",
			Help: "Total number of profiling batches completed",
		},
		[]string{"model_id"},
	)

	ProfilingJobDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_profiling_job_duration_seconds",
			Help:    "Duration of profiling jobs",
			Buckets: prometheus.ExponentialBuckets(1, 2, 15),
		},
		[]string{"model_id", "status"},
	)

	LongbowIngestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_longbow_ingest_duration_seconds",
			Help:    "Duration of Longbow ingestion operations",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"operation"},
	)

	LongbowSearchDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_longbow_search_duration_seconds",
			Help:    "Duration of Longbow search operations",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"k"},
	)

	ActiveSessions = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "burnnotice_active_sessions",
			Help: "Number of currently active monitoring sessions",
		},
	)

	ProfileCount = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "burnnotice_profile_count",
			Help: "Number of loaded model profiles",
		},
	)

	KillSwitchTriggersTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_kill_switch_triggers_total",
			Help: "Total number of kill switch activations",
		},
		[]string{"model_id", "reason"},
	)

	ONNXForwardDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_onnx_forward_duration_seconds",
			Help:    "Duration of ONNX forward pass operations",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
		},
		[]string{"model_id"},
	)

	WASMForwardDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_wasm_forward_duration_seconds",
			Help:    "Duration of WASM forward pass operations",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
		},
		[]string{"model_id"},
	)

	DatasetPairsLoaded = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_dataset_pairs_loaded",
			Help: "Number of contrastive pairs loaded from dataset",
		},
		[]string{"model_id", "dataset_path"},
	)

	DirectionExtractionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_direction_extraction_duration_seconds",
			Help:    "Duration of direction extraction operations",
			Buckets: prometheus.ExponentialBuckets(0.01, 2, 12),
		},
		[]string{"method", "model_id"},
	)

	PolicyViolationsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_policy_violations_total",
			Help: "Total number of policy violations detected",
		},
		[]string{"policy_name", "metric", "action"},
	)

	ProfileJobsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_profile_jobs_total",
			Help: "Total number of profiling jobs by status",
		},
		[]string{"status"},
	)

	ProfileJobsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "burnnotice_profile_jobs_active",
			Help: "Number of currently running profiling jobs",
		},
	)

	SteeringDispatchTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_steering_dispatch_total",
			Help: "Total number of steering commands dispatched by method",
		},
		[]string{"method"},
	)

	HTTPRequestDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_http_request_duration_seconds",
			Help:    "Duration of HTTP requests",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 12),
		},
		[]string{"method", "path", "status"},
	)

	SSEConnectionsActive = promauto.NewGauge(
		prometheus.GaugeOpts{
			Name: "burnnotice_sse_connections_active",
			Help: "Number of active SSE connections",
		},
	)

	SSEMessagesSentTotal = promauto.NewCounter(
		prometheus.CounterOpts{
			Name: "burnnotice_sse_messages_sent_total",
			Help: "Total number of SSE messages sent to clients",
		},
	)

	DashboardPageViewsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_dashboard_page_views_total",
			Help: "Total number of dashboard page views",
		},
		[]string{"page"},
	)
)
