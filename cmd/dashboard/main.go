package main

import (
	"context"
	"log"
	"net/http"
	"os"
	"path/filepath"
	"strconv"
	"time"

	"burnnotice/internal/agent"
	"burnnotice/internal/checkpoint"
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
	"burnnotice/internal/web"
)

type dashboardApp struct {
	handler http.Handler
	closeFn func()
}

func buildDashboardApp() *dashboardApp {
	ctx, cancel := context.WithCancel(context.Background())

	otelCfg := metrics.OTelConfigFromEnv()
	if _, err := metrics.InitOTelExporter(otelCfg); err != nil {
		log.Printf("OTel exporter disabled: %v", err)
	}
	if rateStr := os.Getenv("BURNNOTICE_METRIC_SAMPLE_RATE"); rateStr != "" {
		if rate, err := strconv.ParseUint(rateStr, 10, 64); err == nil && rate > 0 {
			metrics.DefaultSampler.SetSampleRate(rate)
		}
	}

	dsn := os.Getenv("BURNNOTICE_LEDGER_DSN")
	if dsn == "" {
		dsn = ":memory:"
	}
	ledg, err := ledger.NewBackend("sqlite", dsn)
	if err == nil {
		if initErr := ledg.Init(); initErr != nil {
			log.Printf("ledger init warning: %v", initErr)
		}
	}

	hub := web.NewSSEHub()
	ks, _ := policy.NewKillSwitch(func(reason string) {
		hub.Broadcast("killswitch", reason)
	})

	policies := []policy.Policy{
		{
			Name:            "deception_prevention",
			Metric:          "deception",
			Threshold:       0.7,
			Operator:        "greater",
			Action:          policy.ActionSteer,
			EscalationCount: 3,
		},
	}
	if policyPath := os.Getenv("BURNNOTICE_POLICY_FILE"); policyPath != "" {
		if loaded, loadErr := policy.LoadPolicies(policyPath); loadErr == nil {
			policies = loaded
		}
	}
	polEngine := policy.NewEngine(policies)
	polEngine.RegisterKillSwitch(ks)

	if opa, opaErr := policy.NewOPAEvaluator(policy.OPAConfig{}); opaErr == nil {
		polEngine.SetOPAEvaluator(opa)
	}

	profilesDir := os.Getenv("BURNNOTICE_PROFILES_DIR")
	if profilesDir == "" {
		profilesDir = "./profiles"
	}
	reg := registry.NewRegistry(profilesDir)
	_ = reg.Scan()

	profMgr := profile.NewManager(2)
	profMgr.Start(func(job *profile.Job) error {
		return nil
	})

	sched := scheduler.NewScheduler(8192)

	scorer := scoring.NewScorer(nil)
	asyncScorer := scoring.NewAsyncScorer(scorer, 1000, func(scores map[string]float64) {
		polEngine.Evaluate(scores)
	})
	asyncScorer.Start(ctx, 1)

	var compRep *compliance.Reporter
	if ledg != nil {
		compRep = compliance.NewReporter(ledg)
	}
	approvalWF := compliance.NewApprovalWorkflow(5 * time.Minute)

	defaultSubspace := [][]float64{{1, 0, 0, 0}, {0, 1, 0, 0}}
	crescendo, _ := steer.NewCrescendoDetector(defaultSubspace, steer.DefaultCrescendoConfig())

	mw, _ := agent.NewMiddleware(agent.Config{
		ModelID:          "dashboard-default",
		ScoringThreshold: 0.7,
	})
	mcpMW := agent.NewMCPMiddleware(mw, "dashboard")
	if opa := polEngine.OPAEvaluator(); opa != nil {
		mcpMW.SetOPAEvaluator(opa)
	}

	var ckptMgr *checkpoint.Manager
	if ckptPath := os.Getenv("BURNNOTICE_CHECKPOINT_PATH"); ckptPath != "" {
		ckptMgr = checkpoint.NewManager(30*time.Second, filepath.Clean(ckptPath))
		ckptMgr.Start(map[string]any{"status": "running"})
	}

	handlers := web.NewHandlers(hub)
	handlers.PolicyEngine = polEngine
	handlers.ProfileMgr = profMgr
	handlers.Registry = reg
	handlers.Scheduler = sched
	handlers.Ledger = ledg
	handlers.Approval = approvalWF
	handlers.Compliance = compRep
	handlers.AsyncScorer = asyncScorer
	handlers.Scorer = scorer
	handlers.Sanitizer = scoring.NewSanitizer()
	handlers.Aggregator = scoring.NewAggregator()
	handlers.Hallucination = hallucination.NewHallucinationDetector(map[int]float64{0: 1.0}, 2.0, 8)
	handlers.Crescendo = crescendo
	handlers.Shield = steer.NewAdversarialShield(steer.DefaultAdversarialShieldConfig())
	handlers.CoT = steer.NewCoTTracker(true)
	handlers.Coherence = steer.NewCoherenceTracker(15.0, 20)
	handlers.Governor = agent.NewMultiAgentGovernor(0.7)
	handlers.MCP = mcpMW
	handlers.Pool = memory.NewPool(1024 * 1024)
	handlers.PolicyDetect = policydetect.NewWhiteningDetector(1e-8)
	handlers.SemanticGater = policy.NewSemanticGater(nil)
	handlers.KillSwitch = ks

	mux := http.NewServeMux()
	handlers.RegisterRoutes(mux, "static")

	handler := metrics.HTTPMiddleware(web.Middleware(mux))

	closeFn := func() {
		cancel()
		approvalWF.Stop()
		profMgr.Stop()
		if ckptMgr != nil {
			ckptMgr.Stop()
		}
		if opa := polEngine.OPAEvaluator(); opa != nil {
			_ = opa.Close(context.Background())
		}
		if ledg != nil {
			_ = ledg.Close()
		}
		_ = metrics.ShutdownOTel(context.Background())
	}

	return &dashboardApp{
		handler: handler,
		closeFn: closeFn,
	}
}

func main() {
	port := os.Getenv("PORT")
	if port == "" {
		port = "8080"
	}

	app := buildDashboardApp()
	defer app.closeFn()

	srv := &http.Server{
		Addr:         ":" + port,
		Handler:      app.handler,
		ReadTimeout:  15 * time.Second,
		WriteTimeout: 30 * time.Second,
		IdleTimeout:  60 * time.Second,
	}

	log.Printf("Dashboard listening on :%s", port) // #nosec G706 -- port from env var, safe for logging
	log.Fatal(srv.ListenAndServe())
}
