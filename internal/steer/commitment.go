package steer

import (
	"fmt"
	"math"
)

// CommitmentWindowConfig defines parameters for prefill behavioral commitment analysis.
type CommitmentWindowConfig struct {
	ModelFamily     string  `json:"model_family"`
	DefaultWindow   int     `json:"default_window"`    // window span in tokens around commitment peak (e.g. 16)
	PrefillDominant bool    `json:"prefill_dominant"`  // if true, steering deactivates post-prefill
	PostCommitDecay float64 `json:"post_commit_decay"` // decay factor for decoding tokens [0.0, 1.0]
}

// DefaultCommitmentConfig returns architecture-calibrated defaults for a model family.
func DefaultCommitmentConfig(family string) CommitmentWindowConfig {
	switch family {
	case "gemma":
		// Gemma 2 concentrates commitment in tight prefill bands due to pre/post-ff layernorms
		return CommitmentWindowConfig{
			ModelFamily:     family,
			DefaultWindow:   12,
			PrefillDominant: true,
			PostCommitDecay: 0.10,
		}
	case "qwen":
		// Qwen maintains broader prefill window with bias term accumulation
		return CommitmentWindowConfig{
			ModelFamily:     family,
			DefaultWindow:   20,
			PrefillDominant: false,
			PostCommitDecay: 0.70,
		}
	case "llama":
		// Llama distributes decisions across both prefill tail and early decoding tokens
		return CommitmentWindowConfig{
			ModelFamily:     family,
			DefaultWindow:   16,
			PrefillDominant: false,
			PostCommitDecay: 0.60,
		}
	default:
		return CommitmentWindowConfig{
			ModelFamily:     family,
			DefaultWindow:   16,
			PrefillDominant: false,
			PostCommitDecay: 0.50,
		}
	}
}

// CommitmentSnapshot records projection measurements at an individual token position.
type CommitmentSnapshot struct {
	Position        int     `json:"position"`
	ProjectionScore float64 `json:"projection_score"`
	Derivative      float64 `json:"derivative"`
	IsPeak          bool    `json:"is_peak"`
}

// CommitmentAnalysisResult details where behavioral commitment solidified in latent space.
type CommitmentAnalysisResult struct {
	PeakPosition       int                  `json:"peak_position"`
	MaxDerivative      float64              `json:"max_derivative"`
	WindowStart        int                  `json:"window_start"`
	WindowEnd          int                  `json:"window_end"`
	PrefillEnergyRatio float64              `json:"prefill_energy_ratio"` // fraction of decision energy inside prefill
	Snapshots          []CommitmentSnapshot `json:"snapshots"`
}

// BehavioralCommitmentDetector monitors projection evolution across token sequences to identify commitment points.
type BehavioralCommitmentDetector struct {
	cfg CommitmentWindowConfig
}

// NewBehavioralCommitmentDetector creates a detector calibrated with the given configuration.
func NewBehavioralCommitmentDetector(cfg CommitmentWindowConfig) *BehavioralCommitmentDetector {
	if cfg.DefaultWindow <= 0 {
		cfg.DefaultWindow = 16
	}
	return &BehavioralCommitmentDetector{cfg: cfg}
}

// AnalyzeCommitmentWindow sweeps through token-level projection scores and detects the commitment peak.
func (d *BehavioralCommitmentDetector) AnalyzeCommitmentWindow(scores []float64, prefillLength int) (CommitmentAnalysisResult, error) {
	if len(scores) < 2 {
		return CommitmentAnalysisResult{}, fmt.Errorf("insufficient scores for commitment analysis (need at least 2)")
	}

	n := len(scores)
	snapshots := make([]CommitmentSnapshot, n)
	maxDeriv := -1.0
	peakPos := 0
	totalDerivSum := 0.0
	prefillDerivSum := 0.0

	for i := 0; i < n; i++ {
		deriv := 0.0
		if i > 0 {
			deriv = math.Abs(scores[i] - scores[i-1])
		}
		snapshots[i] = CommitmentSnapshot{
			Position:        i,
			ProjectionScore: scores[i],
			Derivative:      deriv,
		}
		totalDerivSum += deriv
		if i < prefillLength {
			prefillDerivSum += deriv
		}

		if deriv > maxDeriv {
			maxDeriv = deriv
			peakPos = i
		}
	}

	if peakPos < n {
		snapshots[peakPos].IsPeak = true
	}

	halfWin := d.cfg.DefaultWindow / 2
	winStart := peakPos - halfWin
	if winStart < 0 {
		winStart = 0
	}
	winEnd := peakPos + halfWin
	if winEnd >= n {
		winEnd = n - 1
	}

	ratio := 1.0
	if totalDerivSum > 1e-9 {
		ratio = prefillDerivSum / totalDerivSum
	}

	return CommitmentAnalysisResult{
		PeakPosition:       peakPos,
		MaxDerivative:      maxDeriv,
		WindowStart:        winStart,
		WindowEnd:          winEnd,
		PrefillEnergyRatio: ratio,
		Snapshots:          snapshots,
	}, nil
}

// ComputeSteeringMultiplier calculates the dynamic scaling factor alpha_eff at a given token position.
// If the architecture is prefill-dominant and the current position is past prefill, steering is attenuated.
func (d *BehavioralCommitmentDetector) ComputeSteeringMultiplier(pos int, prefillLength int, res CommitmentAnalysisResult, baseStrength float64) float64 {
	if baseStrength <= 0 {
		return 0.0
	}

	// If position is inside the critical commitment window, apply 100% full strength
	if pos >= res.WindowStart && pos <= res.WindowEnd {
		return baseStrength
	}

	// Post-prefill decoding behavior
	if pos >= prefillLength {
		if d.cfg.PrefillDominant {
			// Architecture where decoding steering is causally inert (e.g. Gemma 2 / 35B MoE prefill commit)
			return baseStrength * d.cfg.PostCommitDecay
		}
		// Standard decoder (Llama / Qwen) retains active steering
		return baseStrength * math.Max(0.5, d.cfg.PostCommitDecay)
	}

	// Outside commitment window but still within prefill
	return baseStrength * 0.75
}
