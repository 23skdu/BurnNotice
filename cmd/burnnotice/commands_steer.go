package main

import (
	"context"
	"encoding/json"
	"fmt"
	"os"

	"burnnotice/internal/steer"

	"github.com/spf13/cobra"
)

type steerInput struct {
	Activation []float64   `json:"activation"`
	Direction  []float64   `json:"direction"`
	Secondary  []float64   `json:"secondary"`
	Basis      [][]float64 `json:"basis"`
	Weights    [][]float64 `json:"weights"`
	Bias       []float64   `json:"bias"`
	Prompt     string      `json:"prompt"`
	Tokens     []string    `json:"tokens"`
	Method     string      `json:"method"`
	Strength   float64     `json:"strength"`
}

var steerCmd = &cobra.Command{
	Use:   "steer [input-json]",
	Short: "Run activation steering, CAST, LoRA, GCAD, Crescendo, CoT, and Adversarial Shield pipelines",
	Args:  cobra.MaximumNArgs(1),
	RunE: func(cmd *cobra.Command, args []string) error {
		methodFlag, _ := cmd.Flags().GetString("method")
		strengthFlag, _ := cmd.Flags().GetFloat64("strength")

		in := steerInput{
			Activation: []float64{1.0, 2.0, -0.5, 0.5},
			Direction:  []float64{0.5, -0.5, 0.5, -0.5},
			Secondary:  []float64{0.0, 1.0, 0.0, 0.0},
			Basis: [][]float64{
				{1.0, 0.0, 0.0, 0.0},
				{0.0, 1.0, 0.0, 0.0},
			},
			Weights: [][]float64{
				{1.0, 0.1, 0.0, 0.0},
				{0.1, 1.0, 0.0, 0.0},
				{0.0, 0.0, 1.0, 0.1},
				{0.0, 0.0, 0.1, 1.0},
			},
			Bias:     []float64{0.2, -0.3, 0.1, 0.4},
			Prompt:   "Evaluate steering pipeline safety.",
			Tokens:   []string{"<think>", "checking", "</think>", "verified"},
			Method:   methodFlag,
			Strength: strengthFlag,
		}

		if len(args) == 1 {
			// #nosec G304 -- CLI accepts user-provided steering input JSON
			raw, err := os.ReadFile(args[0])
			if err != nil {
				return fmt.Errorf("read steer input: %w", err)
			}
			if err := json.Unmarshal(raw, &in); err != nil {
				return fmt.Errorf("parse steer input: %w", err)
			}
		}

		if in.Strength == 0 {
			in.Strength = 0.5
		}
		if len(in.Activation) == 0 || len(in.Direction) == 0 {
			return fmt.Errorf("activation and direction vectors are required")
		}

		// 1. Interpolate direction & resolve component strengths
		interpDir, err := steer.InterpolateDirectionFromCommand(in.Direction, in.Basis, 0.5)
		if err != nil {
			return fmt.Errorf("interpolate direction: %w", err)
		}
		if len(in.Basis) > 0 {
			_, _ = steer.FloatDirectionInterpolation(in.Basis, 0.5, 1.0)
		}

		attnScale := steer.ResolveComponentStrengths(steer.Command{Strength: in.Strength}, steer.ComponentAttentionOut)

		// 2. Dispatcher across methods (Addition, Angular, Ablation, Selective)
		methods := map[int]steer.Method{
			0: steer.MethodAddition,
			1: steer.MethodAngular,
			2: steer.MethodAblation,
			3: steer.MethodSelective,
		}
		disp := steer.NewDispatcher([]int{0, 1, 2, 3}, methods)
		steered := disp.Apply(0, in.Activation, interpDir, attnScale)
		_ = disp.Apply(1, in.Activation, interpDir, attnScale)
		_ = disp.Apply(2, in.Activation, interpDir, attnScale)
		_ = disp.Apply(3, in.Activation, interpDir, attnScale)

		// 3. SelectiveSteerer & ApplyCommand
		sel := steer.NewSelectiveSteerer(map[int][]float64{0: in.Secondary}, map[int][]float64{0: in.Direction}, []int{0})
		_ = sel.IsDiscriminative(0, in.Activation)
		_ = sel.ComputeSteering(0, in.Activation, in.Strength)
		_ = steer.ApplyCommand(in.Activation, steer.Command{
			Method:          steer.MethodBiprojection,
			Vector:          in.Direction,
			SecondaryVector: in.Secondary,
			Strength:        in.Strength,
		})

		// 4. CAST conditional steerer
		cast := steer.NewConditionalSteerer()
		condNode := steer.ConditionNode{
			Category:      "deception",
			Vector:        in.Direction,
			Threshold:     -1.0,
			Operator:      steer.OpSingle,
			SmoothScaling: true,
		}
		_, _, _ = cast.Evaluate(in.Activation, condNode)
		castOut, castApplied, err := cast.ApplyConditional(in.Activation, steer.Command{
			Method:   steer.MethodNormPreserving,
			Vector:   in.Direction,
			Strength: in.Strength,
		}, condNode)
		if err != nil {
			return fmt.Errorf("CAST conditional steering: %w", err)
		}

		// 5. Bias steering
		var biasOut []float64
		if steer.IsBiasSupported(steer.ComponentAttentionOut) && len(in.Bias) == len(in.Direction) {
			biasOut, _ = steer.ComputeBiasSteering(in.Bias, in.Direction, in.Strength, 1e-4)
		}

		// 6. LoRA reversible steering
		loraApplied := false
		if len(in.Weights) == len(in.Direction) {
			lora := steer.NewLoRASteerer(len(in.Direction), 2)
			if err := lora.ComputeLoRAWeights(in.Direction, in.Strength); err == nil {
				if wPrime, err := lora.ApplyLoRA(in.Weights); err == nil {
					loraApplied = lora.IsApplied()
					_, _ = lora.ReverseLoRA(wPrime)
					_ = lora.Rank()
				}
			}
		}

		// 7. Quantization direction mapper
		qm := steer.NewQuantizationMapper()
		sCalib := qm.CalibrateScale([][]float64{in.Activation}, [][]float64{in.Activation})
		_, _ = qm.TransferDirection(in.Direction, sCalib, in.Direction)

		// 8. KV-cache accumulation + GCAD + Crescendo detector
		gcadCfg := steer.DefaultGCADConfig()
		gcadOut, _ := steer.GCADApply(steered, in.Activation, 1, gcadCfg)
		_ = steer.GCADApplyInPlace(append([]float64(nil), steered...), in.Activation, 1, gcadCfg)

		crescendoCfg := steer.DefaultCrescendoConfig()
		cd, err := steer.NewCrescendoDetector([][]float64{in.Direction}, crescendoCfg)
		if err != nil {
			return fmt.Errorf("crescendo detector: %w", err)
		}
		turnRes, err := cd.StepTurn(context.Background(), 0, in.Activation, in.Direction, [][]float64{in.Activation})
		if err != nil {
			return fmt.Errorf("crescendo step: %w", err)
		}
		_, _, _ = cd.ReAnchor(context.Background(), [][]float64{in.Activation})
		_ = cd.ReAnchorCount()
		_ = cd.EscalationCount()
		tr := cd.Tracker()
		_ = tr.History()
		_ = tr.AlarmCount()
		_ = tr.ReAnchorCount()
		cd.Reset()

		// 9. CoT & Coherence trackers & Adversarial Shield
		cot := steer.NewCoTTracker(true)
		for _, tok := range in.Tokens {
			_, _ = cot.IngestToken(tok, 0.25)
		}
		cotPhase := cot.CurrentPhase()
		cotSummary := cot.Summary()
		cot.Reset()

		coh := steer.NewCoherenceTracker(15.0, 20)
		coh.RecordTokenLogProb(-0.4)
		coh.RecordTokenLogProb(-0.6)
		perp := coh.ComputePerplexity()
		cohMult := coh.AdaptiveSteeringMultiplier()
		coh.Reset()

		shield := steer.NewAdversarialShield(steer.DefaultAdversarialShieldConfig())
		flagged, _ := shield.SanitizePrompt(in.Prompt)
		noisyAct, _ := shield.InjectLatentNoise(in.Activation)
		_, _ = shield.MaskSteeringGradient(in.Direction, in.Direction)
		shieldStats := shield.Stats()

		out := map[string]any{
			"steered_dim":      len(steered),
			"cast_applied":     castApplied,
			"cast_dim":         len(castOut),
			"bias_dim":         len(biasOut),
			"lora_applied":     loraApplied,
			"gcad_dim":         len(gcadOut),
			"crescendo_score":  turnRes.SubspaceScore,
			"cot_phase":        cotPhase,
			"cot_answer_toks":  cotSummary.AnswerTokens,
			"perplexity":       perp,
			"coherence_mult":   cohMult,
			"prompt_flagged":   flagged,
			"noisy_dim":        len(noisyAct),
			"shield_sanitized": shieldStats.BlockedSuffixes,
		}
		enc, _ := json.MarshalIndent(out, "", "  ")
		fmt.Println(string(enc))
		return nil
	},
}

func init() {
	steerCmd.Flags().String("method", "angular", "Primary steering method")
	steerCmd.Flags().Float64("strength", 0.5, "Steering strength")
}
