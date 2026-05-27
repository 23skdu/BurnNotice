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

	SAEFeatureExtractionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_sae_feature_extraction_duration_seconds",
			Help:    "Duration of SAE feature extraction operations",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"model_id", "layer"},
	)

	SAEReconstructionError = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_sae_reconstruction_error",
			Help:    "SAE reconstruction L2 error",
			Buckets: prometheus.ExponentialBuckets(0.001, 2, 15),
		},
		[]string{"model_id"},
	)

	HallucinationEnergyCurrent = promauto.NewGaugeVec(
		prometheus.GaugeOpts{
			Name: "burnnotice_hallucination_energy_current",
			Help: "Current hallucination energy score",
		},
		[]string{"model_id", "layer"},
	)

	HallucinationAlertsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_hallucination_alerts_total",
			Help: "Total number of hallucination alerts triggered",
		},
		[]string{"model_id", "layer"},
	)

	WhiteningDetectionDuration = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_whitening_detection_duration_seconds",
			Help:    "Duration of whitening-based policy violation detection",
			Buckets: prometheus.ExponentialBuckets(0.0001, 2, 12),
		},
		[]string{"model_id"},
	)

	CrossModelTransferTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_cross_model_transfer_total",
			Help: "Total number of cross-model direction transfers",
		},
		[]string{"source_model", "target_model", "direction"},
	)

	MoEExpertActivationTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_moe_expert_activation_total",
			Help: "Total number of MoE expert activations",
		},
		[]string{"model_id", "expert_id", "layer"},
	)

	LoRAApplicationTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_lora_application_total",
			Help: "Total number of LoRA steering applications",
		},
		[]string{"model_id", "rank"},
	)

	ProfileExportTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_profile_export_total",
			Help: "Total number of profile exports by format",
		},
		[]string{"format"},
	)

	ProfileImportTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_profile_import_total",
			Help: "Total number of profile imports by format",
		},
		[]string{"format"},
	)

	AlignmentDriftAlertsTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_alignment_drift_alerts_total",
			Help: "Total number of alignment drift alerts",
		},
		[]string{"direction", "layer"},
	)

	CUDAKernelLatencyUS = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_cuda_kernel_latency_us",
			Help:    "Latency of C++/CUDA GPU kernel executions in microseconds",
			Buckets: prometheus.ExponentialBuckets(1, 2, 12),
		},
		[]string{"kernel_name"},
	)

	VLLMHookTriggersTotal = promauto.NewCounterVec(
		prometheus.CounterOpts{
			Name: "burnnotice_vllm_hook_triggers_total",
			Help: "Total number of times vLLM triggered the steering hook",
		},
		[]string{"model_id"},
	)

	LLMClassifierLatencyMS = promauto.NewHistogramVec(
		prometheus.HistogramOpts{
			Name:    "burnnotice_llm_classifier_latency_ms",
			Help:    "Latency of the async LLM pre-classifier in milliseconds",
			Buckets: prometheus.ExponentialBuckets(10, 2, 10),
		},
		[]string{"classifier_type"},
	)
)
