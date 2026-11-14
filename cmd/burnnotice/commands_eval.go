package main

import (
	"fmt"
	"os"

	"burnnotice/internal/analysis"
	"burnnotice/internal/extract"
	burnmath "burnnotice/internal/math"
	"burnnotice/internal/profile"
	"burnnotice/internal/scoring"
	"burnnotice/internal/zjson"

	"github.com/spf13/cobra"
)

type evalInput struct {
	Positive     [][]float64                    `json:"positive"`
	Negative     [][]float64                    `json:"negative"`
	BaseLogits   [][]float64                    `json:"base_logits"`
	SteerLogits  [][]float64                    `json:"steer_logits"`
	HarmResults  []analysis.JailbreakEvalResult `json:"harm_results"`
	ParetoPoints []analysis.ParetoPoint         `json:"pareto_points"`
	PreScores    map[int]float64                `json:"pre_scores"`
	PostScores   map[int]float64                `json:"post_scores"`
	TargetDim    int                            `json:"target_dim"`
}

var evalCmd = &cobra.Command{
	Use:   "eval [input-json]",
	Short: "Run geometric, KL, Heretic/HarmBench, Ouroboros, obfuscation, and zjson diagnostics",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		in := evalInput{
			Positive:    [][]float64{{1.0, 0.2, 0.1, 0.0}, {0.9, 0.3, 0.2, 0.1}, {1.1, 0.1, 0.0, 0.2}},
			Negative:    [][]float64{{-1.0, -0.2, 0.1, 0.0}, {-0.9, -0.3, 0.2, -0.1}, {-1.1, -0.1, 0.0, -0.2}},
			BaseLogits:  [][]float64{{2.0, 1.0, 0.5}, {1.5, 0.5, 0.2}},
			SteerLogits: [][]float64{{1.9, 1.1, 0.5}, {1.4, 0.6, 0.2}},
			HarmResults: []analysis.JailbreakEvalResult{
				{PromptID: "hb-1", Category: analysis.HarmCategoryCybercrime, BaselineSuccess: true, SteeredSuccess: false, EarlyTokenShift: 0.04},
				{PromptID: "hb-2", Category: analysis.HarmCategoryCBRN, BaselineSuccess: false, SteeredSuccess: false, EarlyTokenShift: 0.02},
			},
			ParetoPoints: []analysis.ParetoPoint{
				{Strength: 0.5, KLDivergence: 0.03, RefusalRate: 0.90},
				{Strength: 1.0, KLDivergence: 0.08, RefusalRate: 0.96},
			},
			PreScores:  map[int]float64{0: 0.85, 1: 0.70, 2: 0.60},
			PostScores: map[int]float64{0: 0.25, 1: 0.75, 2: 0.20},
			TargetDim:  8,
		}

		if len(args) == 1 {
			// #nosec G304 -- CLI accepts user-provided evaluation file path
			raw, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read eval input: %w", err)
			}
			if err := zjson.UnsafeDecode(raw, &in); err != nil {
				return fmt.Errorf("parse eval JSON: %w", err)
			}
			inspectZJSON(raw)
		}

		if len(in.Positive) == 0 || len(in.Negative) == 0 {
			return fmt.Errorf("positive and negative activation matrices are required")
		}
		dim := len(in.Positive[0])

		// 1. Math & Whitening pipeline
		winsorized := burnmath.Winsorize(in.Positive[0], 0.05, 0.95)
		if !burnmath.IsStable(winsorized) {
			return fmt.Errorf("winsorized activation contains non-finite values")
		}
		_ = burnmath.EstimateCovariance(in.Positive, -1.0)
		cov := burnmath.Covariance(in.Positive)
		avgVar := burnmath.AverageVariance(cov)
		xWhite, wMat, _, ok := burnmath.ZCAWhitening(in.Positive, -1.0, 0.0)
		if ok {
			_ = burnmath.ApplyWhitening(in.Negative, wMat)
		}
		alphaSched := burnmath.NewAlphaScheduler(4)
		projVec, _ := burnmath.RidgeRegularizedProjection(in.Positive[0], [][]float64{in.Negative[0]}, alphaSched.Alpha(1))

		// 2. Direction extraction, transfer & scoring
		dir := extract.ExtractDIM(in.Positive, in.Negative)
		dotVal := extract.Dot(dir, in.Positive[0])
		sigVal := extract.Sigmoid(dotVal)
		targetDim := in.TargetDim
		if targetDim <= 0 {
			targetDim = dim
		}
		xfer := extract.NewDirectionTransfer(dim, targetDim)
		xferDir, err := xfer.TransferDirection(dir)
		if err != nil {
			return fmt.Errorf("transfer direction: %w", err)
		}
		ckaExt, _ := extract.CKASimilarity(in.Positive, in.Negative)

		sc := scoring.NewScorer(nil)
		sc.RegisterDirection("deception", dir)
		_ = sc.Score(in.Positive[0])
		subEnergy := sc.ScoreSubspace(in.Positive[0], [][]float64{dir})

		// 3. KL, Heretic/HarmBench, Externalities, Ouroboros, Separation, Alignment
		tokenKLs, meanKL, err := analysis.ComputeBatchKLDivergence(in.BaseLogits, in.SteerLogits)
		if err != nil {
			return fmt.Errorf("batch KL: %w", err)
		}
		shift := analysis.ComputeEarlyTokenShift(in.BaseLogits, in.SteerLogits)
		drift := analysis.DetectEarlyTokenDrift(in.Positive, in.Negative)

		assessor := analysis.NewJailbreakRiskAssessor(1.2)
		extReport, err := assessor.Assess(in.HarmResults)
		if err != nil {
			return fmt.Errorf("externalities assess: %w", err)
		}
		refusedMap := make(map[string]bool, len(in.HarmResults))
		catMap := make(map[string]string, len(in.HarmResults))
		for _, hr := range in.HarmResults {
			refusedMap[hr.PromptID] = !hr.SteeredSuccess
			catMap[hr.PromptID] = hr.Category
		}
		hbCats := analysis.EvaluateHarmBenchCategories(refusedMap, catMap)
		pareto, err := analysis.ComputeParetoFrontier(in.ParetoPoints)
		if err != nil {
			return fmt.Errorf("pareto frontier: %w", err)
		}
		heretic := analysis.ComputeHereticScore(0.95, meanKL, 0.70, 0.30)

		ouro := analysis.NewOuroborosDetector()
		for l, v := range in.PreScores {
			ouro.RecordPre(l, v)
		}
		for l, v := range in.PostScores {
			ouro.RecordPost(l, v)
		}
		compLayers := ouro.CompensatingLayers()
		netRed := ouro.NetReduction()
		reqRef := ouro.RequiresRefinement(0.5)
		refPasses := ouro.RefinementPasses(0.5, 3)
		ouro.Reset()

		centroid, _ := analysis.ComputeCentroid(in.Positive)
		sep, _ := analysis.ComputeLatentSeparation(in.Negative, in.Positive)
		ret, deg, _ := analysis.MeasureSeparationRetention(in.Negative, in.Positive, in.Negative, in.Positive)

		// 4. Obfuscation defenses (MLP probe, FGSM perturber, Mahalanobis OOD)
		probe := profile.NewMLPProbe(dim, 8)
		probePred := probe.Predict(in.Positive[0])
		perturber := &profile.AdversarialPerturber{Epsilon: 0.05}
		perturbed := perturber.Perturb(in.Positive[0], probe, 0.1)
		ood, err := profile.NewOODAnomalyDetector(in.Positive, 3.0)
		if err != nil {
			return fmt.Errorf("OOD detector: %w", err)
		}
		isOOD := ood.IsAnomaly(perturbed)

		// 5. Serialize summary via zjson
		summary := map[string]any{
			"avg_variance":       avgVar,
			"whitened_rows":      len(xWhite),
			"ridge_proj_dim":     len(projVec),
			"sigmoid_score":      sigVal,
			"subspace_energy":    subEnergy,
			"transfer_dim":       len(xferDir),
			"cka_extract":        ckaExt,
			"mean_kl":            meanKL,
			"token_kl_count":     len(tokenKLs),
			"early_token_shift":  shift,
			"early_token_drift":  drift,
			"externalities_risk": extReport.ElevatedRisk,
			"harmbench_cats":     len(hbCats),
			"pareto_count":       len(pareto),
			"heretic_score":      heretic,
			"compensating":       len(compLayers),
			"net_reduction":      netRed,
			"requires_refine":    reqRef,
			"refine_passes":      refPasses,
			"centroid_dim":       len(centroid),
			"separation":         sep,
			"retention":          ret,
			"degraded":           deg,
			"mlp_probe_pred":     probePred,
			"perturbed_ood":      isOOD,
		}
		outBytes, err := zjson.EncodeToWriter(summary)
		if err != nil {
			return fmt.Errorf("encode eval summary: %w", err)
		}
		fmt.Println(zjson.Mem2Str(outBytes))
		return nil
	},
}

func inspectZJSON(raw []byte) {
	cfg := zjson.DefaultConfig.Valid()
	bp := zjson.NewBufferPool(256, cfg.MaxBufferSize)
	bufPtr := bp.Get()
	defer bp.Put(bufPtr)

	node := zjson.Parse(raw)
	if node != nil {
		_ = node.Type()
		_ = node.IsNull()
		_ = node.Len()
		_ = node.Raw()
		_ = node.Keys()
		_ = node.Exists("positive")
		if posNode := node.Get("positive"); posNode != nil {
			_ = posNode.At(0)
			posNode.Each(func(_ string, _ *zjson.Node) bool { return false })
		}
		_, _ = node.GetString("missing")
		_, _ = node.GetInt("target_dim")
		_, _ = node.GetFloat("missing")
		_, _ = node.GetBool("missing")
		if sNode := node.At(0); sNode != nil {
			_, _ = sNode.String()
			_, _ = sNode.Int()
			_, _ = sNode.Float()
			_, _ = sNode.Bool()
		}
	}

	tok := zjson.NewTokenizer(raw)
	if st, err := tok.Next(); err == nil {
		b := tok.GetBytes(st)
		_ = tok.GetString(st, true)
		_ = zjson.UnsafeString(b)
	}
	var dst [64]byte
	_ = zjson.Unescape(dst[:], []byte(`hello`))

	_, _ = zjson.UnmarshalAny(raw)
	_, _, _ = zjson.DecodeStream(raw, 0)

	var strBytes []byte
	_ = zjson.GetBytes([]byte(`"burnnotice"`), &strBytes)

	digitBuf := zjson.GetDigitBuf()
	zjson.PutDigitBuf(digitBuf)
	bb := zjson.GetBytesBuffer()
	zjson.PutBytesBuffer(bb)

	rawCopy := zjson.Str2Mem("eval")
	_ = zjson.GrowSlice(rawCopy, 16)
	_ = zjson.SliceHeader(&rawCopy)
	s := "eval"
	_ = zjson.StringHeader(&s)

	var mBuf [512]byte
	_, _ = zjson.MarshalToBuf(map[string]int{"ok": 1}, mBuf[:])
	_, _ = zjson.AppendJSON(nil, map[string]int{"ok": 1})
}
