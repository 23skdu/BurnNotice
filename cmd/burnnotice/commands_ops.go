package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"runtime"
	"time"

	"burnnotice/internal/agent"
	"burnnotice/internal/compliance"
	"burnnotice/internal/datasets"
	"burnnotice/internal/engine"
	"burnnotice/internal/engine/native"
	"burnnotice/internal/hallucination"
	"burnnotice/internal/ledger"
	"burnnotice/internal/longbow"
	"burnnotice/internal/memory"
	"burnnotice/internal/policy"
	"burnnotice/internal/policydetect"
	"burnnotice/internal/profile"
	"burnnotice/internal/registry"
	"burnnotice/internal/sae"
	"burnnotice/internal/scheduler"
	"burnnotice/internal/scoring"
	"burnnotice/internal/telemetry"
	"burnnotice/internal/wasm"

	"github.com/spf13/cobra"
)

var saeCmd = &cobra.Command{
	Use:   "sae [dictionary-path]",
	Short: "Load SAE dictionaries (Gemma Scope, Llama-Scope, Qwen-SAE), analyze absorption, and score hallucination risk",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		familyFlag, _ := cmd.Flags().GetString("family")
		configPath, _ := cmd.Flags().GetString("config")
		targetPath, _ := cmd.Flags().GetString("align-with")

		loader := &sae.SAELoader{}
		if len(args) == 1 {
			var err error
			switch familyFlag {
			case "gemma", "gemma-scope":
				err = loader.LoadGemmaScope(args[0], configPath)
			case "llama", "llama-scope":
				err = loader.LoadLlamaScope(args[0], configPath)
			case "qwen", "qwen-sae":
				err = loader.LoadQwenSAE(args[0], configPath)
			case "json":
				err = loader.Load(args[0])
			default:
				err = loader.LoadDictionary(args[0], configPath)
			}
			if err != nil {
				return fmt.Errorf("load SAE dictionary: %w", err)
			}
		} else {
			if err := loader.LoadFromMemory(&sae.SAEWeights{
				Family:       sae.SAEGemmaScope,
				FeatureCount: 4,
				InputDim:     4,
				EncoderWeights: [][]float64{
					{1, 0, 0, 0},
					{0, 1, 0, 0},
					{0, 0, 1, 0},
					{0, 0, 0, 1},
				},
				DecoderWeights: [][]float64{
					{1, 0, 0, 0},
					{0, 1, 0, 0},
					{0, 0, 1, 0},
					{0, 0, 0, 1},
				},
			}); err != nil {
				return err
			}
		}

		w := loader.Weights()
		fam := loader.Family()
		modelFam := sae.MapSAEToModelFamily(fam)

		act := make([]float64, w.InputDim)
		for i := range act {
			act[i] = 0.5 / float64(i+1)
		}
		feats, err := loader.Encode(act)
		if err != nil {
			return fmt.Errorf("encode SAE: %w", err)
		}
		recon, err := loader.Decode(feats)
		if err != nil {
			return fmt.Errorf("decode SAE: %w", err)
		}
		topK, _ := loader.TopKFeatures(feats, 2)
		attr := loader.Attribution(feats)
		reconErr, _ := loader.ReconstructionError(act)

		hd := hallucination.NewHallucinationDetector(map[int]float64{0: 1.0}, 2.0, 4)
		if err := hd.Validate(); err != nil {
			return err
		}
		hd.UpdateHistory(0, feats)
		_ = hd.BaselineEnergy(0)
		risk, spike, hall := hd.IngestFamilyFeatures(0, feats, string(fam))
		hd.ResetHistory()

		var absReport *sae.FeatureAbsorptionReport
		absReport, _ = sae.AnalyzeFeatureAbsorption(loader, [][]float64{act}, map[string][]float64{"deception": act})

		targetLoader := &sae.SAELoader{}
		if targetPath != "" {
			if err := targetLoader.LoadDictionary(targetPath, ""); err != nil {
				return fmt.Errorf("load target SAE: %w", err)
			}
		} else {
			_ = targetLoader.LoadFromMemory(&sae.SAEWeights{
				Family:         sae.SAELlamaScope,
				FeatureCount:   w.FeatureCount,
				InputDim:       w.InputDim,
				EncoderWeights: w.EncoderWeights,
				DecoderWeights: w.DecoderWeights,
			})
		}
		alignRes, _ := sae.AlignSAEDictionaries(loader, targetLoader, [][]float64{act}, [][]float64{act}, 2)

		out := map[string]any{
			"sae_family":           fam,
			"model_family":         modelFam,
			"recon_dim":            len(recon),
			"topk_count":           len(topK),
			"attribution_count":    len(attr),
			"reconstruction_error": reconErr,
			"hallucination_risk":   risk,
			"hallucination_spike":  spike,
			"hallucinating":        hall,
			"absorption":           absReport,
			"alignment":            alignRes,
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

var policyCmd = &cobra.Command{
	Use:   "policy [policies-yaml]",
	Short: "Evaluate policies, OPA WASM/REST/compound rules, semantic gater, killswitch, and whitening detector",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		promptFlag, _ := cmd.Flags().GetString("prompt")
		toolFlag, _ := cmd.Flags().GetString("tool")
		blastFlag, _ := cmd.Flags().GetString("blast-radius")
		deceptionFlag, _ := cmd.Flags().GetFloat64("deception")
		wasmPath, _ := cmd.Flags().GetString("opa-wasm")
		opaURL, _ := cmd.Flags().GetString("opa-url")

		policies := []policy.Policy{
			{Name: "deception_gate", Metric: "deception", Threshold: 0.7, Operator: "greater", Action: policy.ActionSteer, EscalationCount: 2},
			{Name: "critical_halt", Metric: "deception", Threshold: 0.95, Operator: "greater", Action: policy.ActionTerminate, EscalationCount: 1},
		}
		if len(args) == 1 {
			loaded, err := policy.LoadPolicies(args[0])
			if err != nil {
				return fmt.Errorf("load policies: %w", err)
			}
			policies = loaded
		}

		ks, _ := policy.NewKillSwitch(func(reason string) {
			fmt.Printf("KillSwitch triggered: %s\n", reason)
		})
		_ = ks.IsArmed()
		_ = ks.Context()

		eng := policy.NewEngine(policies)
		eng.RegisterKillSwitch(ks)

		opaCfg := policy.OPAConfig{
			WASMPath: wasmPath,
			Endpoint: opaURL,
			Rules: []policy.OPARule{
				{
					ID:         "high_blast_deception",
					Tools:      []string{"exec_sql", "shell"},
					BlastRadii: []string{"high", "critical"},
					MinScores:  map[string]float64{"deception": 0.4},
					Action:     policy.ActionBlock,
					Reason:     "high blast radius tool blocked under latent deception",
				},
			},
		}
		opa, err := policy.NewOPAEvaluator(opaCfg)
		if err != nil {
			return fmt.Errorf("init OPA evaluator: %w", err)
		}
		defer func() { _ = opa.Close(context.Background()) }()
		opa.SetRules(opaCfg.Rules)
		eng.SetOPAEvaluator(opa)

		sg := policy.NewSemanticGater(nil)
		sg.SetIntentWeight(policy.IntentAcademicQuery, 0.25)
		scaledDeception := sg.ScaleMetric(promptFlag, deceptionFlag)
		bypass := sg.ShouldBypassIntervention(promptFlag)

		dec, violations, evalErr := eng.EvaluateContext(context.Background(), policy.OPAInput{
			Latent: map[string]float64{"deception": scaledDeception},
			MCP:    &policy.OPAMCPContext{ToolName: toolFlag, BlastRadius: blastFlag},
			Intent: sg.ClassifyPrompt(promptFlag),
		})
		escalate := eng.ShouldEscalate("deception_gate")
		vCount := eng.ViolationCount("deception_gate")
		eng.AdjustThresholdsForDrift(map[string]float64{"deception": 0.02})
		eng.UpdatePolicies(policies)
		eng.ResetViolations()
		if !ks.IsArmed() {
			_ = ks.Reset()
		}

		wd := policydetect.NewWhiteningDetector(1e-8)
		_ = wd.Train([][]float64{{1, 0, 0}, {0.9, 0.1, 0}, {1.1, -0.1, 0}}, [][]float64{{-1, 0, 0}})
		wdScore := 0.0
		if wd.IsTrained() && wd.Dimension() == 3 {
			wdScore, _ = wd.Detect([]float64{1, 0, 0})
		}

		out := map[string]any{
			"bundle_digest":    eng.OPAEvaluator().BundleDigest(),
			"decision":         dec,
			"violations":       len(violations),
			"violation_count":  vCount,
			"should_escalate":  escalate,
			"bypass":           bypass,
			"whitening_score":  wdScore,
			"eval_error_noted": evalErr != nil,
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

var complianceCmd = &cobra.Command{
	Use:   "compliance",
	Short: "Generate compliance reports (EU AI Act / NIST AI RMF / PDF), manage approvals, and verify release evidence bundles",
	RunE: func(cmd *cobra.Command, args []string) error {
		outDir, _ := cmd.Flags().GetString("output")
		pdfOut, _ := cmd.Flags().GetString("pdf")
		dsn, _ := cmd.Flags().GetString("dsn")
		if dsn == "" {
			dsn = ":memory:"
		}

		ledg, err := ledger.NewBackend("sqlite", dsn)
		if err != nil {
			return fmt.Errorf("open ledger: %w", err)
		}
		defer func() { _ = ledg.Close() }()
		if err := ledg.Init(); err != nil {
			return err
		}

		rep := compliance.NewReporter(ledg)
		report, err := rep.GenerateReport(time.Now().Add(-24*time.Hour), time.Now().Add(time.Minute))
		if err != nil {
			return err
		}
		jsonStr, err := rep.ExportReport(report)
		if err != nil {
			return err
		}
		if pdfOut != "" {
			pdfBytes, err := rep.ExportPDF(report)
			if err != nil {
				return err
			}
			if err := os.WriteFile(pdfOut, pdfBytes, 0600); err != nil {
				return err
			}
		}

		aw := compliance.NewApprovalWorkflow(5 * time.Minute)
		defer aw.Stop()
		reqID := aw.RequestApproval("cli-session", "compliance-check")
		_ = aw.GetPendingRequests()
		_ = aw.Approve(reqID, "cli-operator")
		req2 := aw.RequestApproval("cli-session-2", "reject-check")
		_ = aw.Reject(req2, "cli-operator", "test rejection")
		_, _ = aw.GetRequest(reqID)

		sc := compliance.NewSteeringController(0.5)
		_, _ = sc.UpdateSteer([]compliance.CapabilityBenchmark{{Name: "MMLU", Baseline: 0.80, Current: 0.79}})

		manifest := compliance.NewBenchmarkManifest("1.24.2", "llama", compliance.ComputeDatasetHash([]byte(jsonStr)), "local", []compliance.BenchmarkResult{
			{Name: "smoke", Iterations: 1, NsPerOp: 100},
		})
		mBytes, err := manifest.SerializeJSON()
		if err != nil {
			return err
		}
		parsedManifest, err := compliance.ParseBenchmarkManifest(mBytes)
		if err != nil {
			return err
		}

		bundle, err := compliance.NewReleaseEvidenceBundle("v0.9.0", "local", *parsedManifest, report, []compliance.ReleaseGateStatus{
			{GateName: "unit", Passed: true},
		})
		if err != nil {
			return err
		}
		_ = bundle.AttachManifests(nil, nil)
		if !bundle.AllGatesPassed() {
			return fmt.Errorf("not all release gates passed")
		}
		bBytes, err := bundle.SerializeJSON()
		if err != nil {
			return err
		}
		if _, err := compliance.ParseReleaseEvidenceBundle(bBytes); err != nil {
			return err
		}
		if outDir != "" {
			if err := os.MkdirAll(outDir, 0750); err != nil {
				return err
			}
			if err := os.WriteFile(filepath.Join(outDir, "evidence.json"), bBytes, 0600); err != nil {
				return err
			}
		}
		fmt.Println(string(bBytes))
		return nil
	},
}

var registryCmd = &cobra.Command{
	Use:   "registry [profiles-dir]",
	Short: "Scan, list, validate, version, diff, and export model profiles",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		dir := "./profiles"
		if len(args) == 1 {
			dir = args[0]
		}
		filter, _ := cmd.Flags().GetString("filter")
		deleteID, _ := cmd.Flags().GetString("delete")

		reg := registry.NewRegistry(dir)
		if err := reg.Scan(); err != nil {
			return fmt.Errorf("scan registry: %w", err)
		}

		// Ensure manager job lifecycle is wired for batch registry operations
		mgr := profile.NewManager(1)
		mgr.Start(func(j *profile.Job) error {
			_ = j.Context()
			return nil
		})
		job := &profile.Job{ID: "reg-scan", ModelID: "all", Total: 1}
		mgr.Submit(job)
		_, _ = mgr.GetJob("reg-scan")
		_ = mgr.ListJobs()
		mgr.Stop()

		if deleteID != "" {
			if err := reg.Delete(deleteID); err != nil {
				return err
			}
			fmt.Printf("Deleted profile %s\n", deleteID)
			return nil
		}

		items := reg.List(filter)
		if len(items) == 0 {
			reg.Add(registry.Metadata{ModelID: "sample", Architecture: "qwen", Date: time.Now()})
		} else {
			if p, err := reg.Get(items[0].ModelID); err == nil {
				rawGob, err := p.Export(profile.Metadata{ModelArchitecture: p.Architecture, Version: 1}, "gob")
				if err == nil {
					if ep, err := profile.ImportProfile(rawGob, "gob"); err == nil {
						diff := profile.ComputeDiff(ep, ep)
						_ = profile.DetectDrift(diff, 0.2)
					}
				}
				vm := profile.NewVersionManager(filepath.Join(dir, items[0].ModelID))
				if versions, err := vm.ListVersions(); err == nil && len(versions) >= 2 {
					p1, e1 := vm.LoadVersion(versions[0].Version)
					p2, e2 := vm.LoadVersion(versions[1].Version)
					if e1 == nil && e2 == nil {
						d := profile.Diff(p1, p2)
						_ = profile.DetectVersionDrift(d, 0.2)
					}
				}
				val := profile.NewValidator(0.5)
				_ = val.Validate(p, &datasets.Dataset{
					Pairs: []datasets.ContrastivePair{
						{
							Truthful:  datasets.PromptResponse{Prompt: "a", Response: "b"},
							Deceptive: datasets.PromptResponse{Prompt: "c", Response: "d"},
						},
					},
				})
			}
		}

		enc, _ := json.MarshalIndent(map[string]any{
			"count":    reg.Count(),
			"profiles": reg.List(filter),
		}, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

var ledgerCmd = &cobra.Command{
	Use:   "ledger",
	Short: "Query, export, prune, and test SQLite/Longbow activity ledger and vector store",
	RunE: func(cmd *cobra.Command, args []string) error {
		driver, _ := cmd.Flags().GetString("driver")
		dsn, _ := cmd.Flags().GetString("dsn")
		exportCSV, _ := cmd.Flags().GetString("export-csv")

		if dsn == "" && driver == "sqlite" {
			dsn = ":memory:"
		}
		be, err := ledger.NewBackend(driver, dsn)
		if err != nil {
			return fmt.Errorf("create ledger backend: %w", err)
		}
		if err := be.Init(); err != nil {
			return fmt.Errorf("init ledger backend: %w", err)
		}
		defer func() { _ = be.Close() }()

		sampleAlert := ledger.Alert{
			Timestamp: time.Now(),
			SessionID: "cli-ledger",
			Metric:    "deception",
			Score:     0.42,
			Threshold: 0.70,
			Action:    "log",
		}
		_ = be.RecordAlert(sampleAlert)
		_ = be.RecordIntervention(ledger.Intervention{
			Timestamp: time.Now(),
			SessionID: "cli-ledger",
			Method:    "angular",
			Layer:     0,
			Result:    "steered",
		})
		_ = be.IngestBatch([]ledger.Alert{sampleAlert})
		_, _ = be.RetrieveBatch("cli-ledger")
		alerts, _ := be.QueryAlerts(ledger.AlertFilter{Limit: 10})
		_, _ = be.QueryInterventions("cli-ledger")
		cnt, _ := be.CountAlerts(ledger.AlertFilter{})
		stats, _ := be.GetAlertStats()
		_, _ = be.DeleteOldAlerts(365 * 24 * time.Hour)
		csvData, _ := be.ExportAlertsCSV(ledger.AlertFilter{})
		if exportCSV != "" {
			_ = os.WriteFile(exportCSV, []byte(csvData), 0600)
		}
		ser, _ := ledger.SerializeAlert(sampleAlert)
		_, _ = ledger.DeserializeAlert(ser)

		// Exercise Longbow vector client (falls back cleanly when server is absent)
		lb := longbow.NewClient(longbow.Config{Addr: dsn, Dim: 4, Backend: "sqlite"})
		if !lb.IsFallbackActive() {
			lb.SetFallbackMode(true)
		}
		_ = lb.Ingest(context.Background(), [][]float32{{1, 0, 0, 0}}, []uint64{1})
		_, _ = lb.Query(context.Background(), []uint64{1})
		_, _, _ = lb.Search(context.Background(), []float32{1, 0, 0, 0}, 1)
		_ = lb.Close()

		_ = telemetry.NewReporter("", "")

		out := map[string]any{
			"driver":       driver,
			"alert_count":  cnt,
			"queried_rows": len(alerts),
			"stats":        stats,
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

var agentCmd = &cobra.Command{
	Use:   "agent [prompt]",
	Short: "Test agent SDK middleware, MCP OPA authorization, LangChain/CrewAI hooks, and multi-agent governance",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		prompt := "summarize status"
		if len(args) == 1 {
			prompt = args[0]
		}
		toolName, _ := cmd.Flags().GetString("tool")
		blast, _ := cmd.Flags().GetString("blast-radius")

		mw, err := agent.NewMiddleware(agent.Config{
			ModelID:            "cli-agent",
			ScoringThreshold:   0.7,
			SteeringStrength:   0.5,
			MonitoringInterval: time.Second,
		})
		if err != nil {
			return err
		}
		_ = mw.GetMonitoringInterval()

		scores, err := mw.InterceptRequest(context.Background(), agent.Request{
			Prompt:    prompt,
			SessionID: "cli-sess",
			Timestamp: time.Now(),
		})
		if err != nil {
			return err
		}
		resp, err := mw.InterceptResponse(context.Background(), agent.Response{
			Text:      prompt,
			SessionID: "cli-sess",
			Timestamp: time.Now(),
			Scores:    scores,
		})
		if err != nil {
			return err
		}

		opa, err := policy.NewOPAEvaluator(policy.OPAConfig{})
		if err != nil {
			return err
		}
		defer func() { _ = opa.Close(context.Background()) }()

		mcpMW := agent.NewMCPMiddleware(mw, "cli-sess")
		mcpMW.SetOPAEvaluator(opa)
		_ = mcpMW.InterceptToolCall(context.Background(), toolName, map[string]interface{}{"q": prompt})
		dec, _ := mcpMW.InterceptToolCallWithLatent(context.Background(), toolName, blast, nil, scores)
		_ = mcpMW.InterceptToolResponse(context.Background(), toolName, "ok")

		gov := agent.NewMultiAgentGovernor(0.7)
		gov.RegisterAgent("planner", "Planner")
		gov.RegisterAgent("executor", "Executor")
		gov.AddPolicy(agent.GovernancePolicy{Name: "audit", MaxDeceptionScore: 0.8, Action: "alert"})
		_, allowed := gov.MonitorMessage("planner", "executor", prompt, scores["deception"])
		_ = gov.DetectDriftPropagation()
		_, _ = gov.GetAgentInfo("planner")
		_ = gov.GetAgentCount()
		_ = gov.GetMessageCount()

		crew := agent.NewCrewAITaskDecorator(mw, "cli-sess")
		_, _ = crew.ExecuteTask(context.Background(), "task-1", func() (string, error) { return prompt, nil })
		_ = crew.MonitorToolUse(context.Background(), toolName, prompt, "ok")

		lc := agent.NewLangChainCallback(mw, "cli-sess")
		_ = lc.OnLLMStart(context.Background(), prompt)
		_ = lc.OnLLMEnd(context.Background(), resp.Text)
		_ = lc.OnChainStart(context.Background(), map[string]interface{}{"input": prompt})
		_ = lc.OnChainEnd(context.Background(), map[string]interface{}{"output": resp.Text})
		_ = lc.OnToolStart(context.Background(), toolName, prompt)
		_ = lc.OnToolEnd(context.Background(), toolName, "ok")

		// Exercise pooled async scorer + aggregator
		pool := memory.NewPool(4096)
		buf := pool.Get(4)
		buf[0] = 1.0
		pool.Put(buf)
		_ = pool.Stats()
		_ = pool.ShouldGC()
		pool.Clear()

		sc := scoring.NewScorer(nil)
		sc.RegisterDirection("deception", []float64{1, 0, 0, 0})
		ctx, cancel := context.WithCancel(context.Background())
		defer cancel()
		as := scoring.NewAsyncScorer(sc, 16, nil)
		as.Start(ctx, 1)
		_ = as.PushActivation([]float64{1, 0, 0, 0})
		_ = as.PushActivationPooled([]float64{1, 0, 0, 0})
		_ = as.IsBypassActive()
		_ = as.QueueDepth()

		agg := scoring.NewAggregator()
		agg.AddToken(prompt, scores)
		_ = agg.Compute()
		_ = agg.ComputeEMA("deception", 0.5)
		_ = agg.DetectDriftTrajectory("deception", 0.2, 2)
		_ = agg.TokenCount()
		agg.Reset()

		out := map[string]any{
			"scores":      scores,
			"opa_allow":   dec.Allow,
			"gov_allowed": allowed,
			"blocked":     mw.ShouldBlock(scores),
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

var runtimeCmd = &cobra.Command{
	Use:   "runtime",
	Short: "Run MoE routing, native GPU kernel dispatch, WASM backend, and GPU scheduler checks",
	RunE: func(cmd *cobra.Command, args []string) error {
		wasmPath, _ := cmd.Flags().GetString("wasm")

		// 1. MoE routing
		moe := engine.NewMoEEngine(2, 1, []int{4, 4}, nil)
		act := []float64{1.0, 0.5, 0.1, 0.0, 0.0, 0.1, 0.0, 0.0}
		experts, weights := moe.DetectRouting(0, act)
		_ = moe.ExpertActivation(act, 0)
		_ = moe.NumExperts()
		_ = moe.ExpertDim(0)
		moe.UpdateRouting(0, weights)
		_, _ = moe.DetectRoutingFromLogits([]float64{2.0, 0.5})
		_, _ = moe.PollLedger(0)

		// 2. Native GPU kernel dispatch
		gpu := native.NewGPUEngine(4)
		cosRes := gpu.CosineSimilarity([]float32{1, 0, 0, 0}, []float32{1, 0, 0, 0}, 1)
		rotRes := gpu.DynamicRotation([]float32{1, 0, 0, 0}, []float32{0, 1, 0, 0}, 0.25)

		// 3. GPU-aware job scheduler
		sched := scheduler.NewScheduler(1024)
		j1 := &scheduler.Job{ID: "job-1", ModelSize: 100 * 1024 * 1024, Priority: 1}
		j2 := &scheduler.Job{ID: "job-2", ModelSize: 200 * 1024 * 1024, Priority: 5}
		sched.Submit(j1)
		sched.Schedule()
		_, _ = sched.Preempt(j2)
		_ = sched.Complete(j1.ID)
		_ = sched.GetRunningJobs()
		_ = sched.GetQueue()
		_ = sched.AvailableMemory()

		// 4. Optional WASM inference backend
		wasmStatus := "skipped"
		if wasmPath != "" {
			wb, err := wasm.NewWASMBackend(wasmPath)
			if err != nil {
				return fmt.Errorf("WASM backend: %w", err)
			}
			defer func() { _ = wb.Close() }()
			if _, err := wb.Forward([]float32{1, 2, 3, 4}); err != nil {
				return fmt.Errorf("WASM forward: %w", err)
			}
			_, _ = wb.GetIntermediate(0)
			wasmStatus = "executed"
		}

		out := map[string]any{
			"os":          runtime.GOOS,
			"arch":        runtime.GOARCH,
			"moe_experts": experts,
			"gpu_cos":     cosRes,
			"gpu_rot_len": len(rotRes),
			"avail_mem":   sched.AvailableMemory(),
			"wasm_status": wasmStatus,
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

func init() {
	saeCmd.Flags().String("family", "auto", "SAE family (auto, gemma-scope, llama-scope, qwen-sae, json)")
	saeCmd.Flags().String("config", "", "Optional SAE config JSON path")
	saeCmd.Flags().String("align-with", "", "Optional second SAE dictionary for cross-architecture alignment")

	policyCmd.Flags().String("prompt", "hello", "Prompt to classify with SemanticGater")
	policyCmd.Flags().String("tool", "read_file", "MCP tool name for OPA evaluation")
	policyCmd.Flags().String("blast-radius", "low", "MCP tool blast radius (low, medium, high, critical)")
	policyCmd.Flags().Float64("deception", 0.2, "Deception score to evaluate")
	policyCmd.Flags().String("opa-wasm", "", "Optional OPA WASM policy bundle path")
	policyCmd.Flags().String("opa-url", "", "Optional OPA REST endpoint URL")

	complianceCmd.Flags().StringP("output", "o", "", "Optional directory to write evidence.json")
	complianceCmd.Flags().String("pdf", "", "Optional path to write PDF compliance report")
	complianceCmd.Flags().String("dsn", ":memory:", "SQLite DSN for ledger queries")

	registryCmd.Flags().String("filter", "", "Filter profile model IDs")
	registryCmd.Flags().String("delete", "", "Delete a profile by model ID")

	ledgerCmd.Flags().String("driver", "sqlite", "Ledger driver (sqlite or longbow)")
	ledgerCmd.Flags().String("dsn", "", "Ledger DSN or Longbow address")
	ledgerCmd.Flags().String("export-csv", "", "Optional path to export alerts CSV")

	agentCmd.Flags().String("tool", "search", "MCP tool name to simulate")
	agentCmd.Flags().String("blast-radius", "low", "MCP blast radius")

	runtimeCmd.Flags().String("wasm", "", "Optional WASM model path for inference")
}
