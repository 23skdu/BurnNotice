package steer

import (
	"context"
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/floats"
)

// CrescendoConfig configures Multi-Turn Crescendo Context Re-Anchoring.
type CrescendoConfig struct {
	ReAnchorCadence         int        `json:"re_anchor_cadence"`         // Full-prefix re-score cadence K (default: 5 turns)
	WindowSize              int        `json:"window_size"`               // Sliding turn window for crescendo ladder detection (default: 15 turns)
	ThetaMaxDeg             float64    `json:"theta_max_deg"`             // Angular drift budget in degrees (default: 15.0°)
	EscalationThreshold     float64    `json:"escalation_threshold"`      // Composite escalation score threshold (default: 0.55)
	MonotonicSlopeThreshold float64    `json:"monotonic_slope_threshold"` // Minimum positive per-turn slope to flag crescendo ladder (default: 0.015)
	MinLadderTurns          int        `json:"min_ladder_turns"`          // Minimum turns in window before slope-based ladder flag triggers (default: 5)
	GCAD                    GCADConfig `json:"gcad"`                      // Exponential KV-cache decay configuration
}

// DefaultCrescendoConfig returns production defaults for detecting 15+ turn crescendo escalations
// without saturating the 15° angular drift budget.
func DefaultCrescendoConfig() CrescendoConfig {
	return CrescendoConfig{
		ReAnchorCadence:         5,
		WindowSize:              15,
		ThetaMaxDeg:             ThetaMaxDegrees,
		EscalationThreshold:     0.55,
		MonotonicSlopeThreshold: 0.015,
		MinLadderTurns:          5,
		GCAD:                    DefaultGCADConfig(),
	}
}

// CrescendoTurnResult captures the per-turn diagnostics from CrescendoDetector.
type CrescendoTurnResult struct {
	TurnIndex             int     `json:"turn_index"`
	SubspaceScore         float64 `json:"subspace_score"`
	DecayedScore          float64 `json:"decayed_score"`
	FullPrefixScore       float64 `json:"full_prefix_score"`
	IncrementalDriftDeg   float64 `json:"incremental_drift_deg"`
	MonotonicSlope        float64 `json:"monotonic_slope"`
	PositiveStepRatio     float64 `json:"positive_step_ratio"`
	EscalationScore       float64 `json:"escalation_score"`
	ReAnchored            bool    `json:"re_anchored"`
	IsEscalationStep      bool    `json:"is_escalation_step"`
	IsCrescendoEscalation bool    `json:"is_crescendo_escalation"`
	DriftAlert            bool    `json:"drift_alert"`
}

type crescendoWindowEntry struct {
	turnIndex     int
	activation    []float64
	steerVec      []float64
	subspaceScore float64
	decayedScore  float64
}

// CrescendoDetector combines exponential GCAD KV-cache decay with periodic full-prefix
// subspace re-scoring to detect slow 15+ turn crescendo escalations without saturating
// the 15° angular drift budget.
type CrescendoDetector struct {
	mu              sync.RWMutex
	cfg             CrescendoConfig
	basis           [][]float64 // Orthonormalized safety subspace basis
	dim             int
	tracker         *KVCacheAccumulationTracker
	window          []crescendoWindowEntry
	turnsSinceAnc   int
	lastPrefixScore float64
	reAnchorTotal   int
	escalationTotal int
}

// NewCrescendoDetector creates a CrescendoDetector initialized with an orthonormalized safety subspace basis.
func NewCrescendoDetector(subspace [][]float64, cfg CrescendoConfig) (*CrescendoDetector, error) {
	if len(subspace) == 0 || len(subspace[0]) == 0 {
		return nil, fmt.Errorf("subspace basis cannot be empty")
	}
	dim := len(subspace[0])
	for i, v := range subspace {
		if len(v) != dim {
			return nil, fmt.Errorf("subspace vector %d has dimension %d, want %d", i, len(v), dim)
		}
	}

	if cfg.ReAnchorCadence <= 0 {
		cfg.ReAnchorCadence = 5
	}
	if cfg.WindowSize <= 0 {
		cfg.WindowSize = 15
	}
	if cfg.ThetaMaxDeg <= 0 {
		cfg.ThetaMaxDeg = ThetaMaxDegrees
	}
	if cfg.EscalationThreshold <= 0 {
		cfg.EscalationThreshold = 0.55
	}
	if cfg.MonotonicSlopeThreshold <= 0 {
		cfg.MonotonicSlopeThreshold = 0.015
	}
	if cfg.MinLadderTurns <= 1 {
		cfg.MinLadderTurns = 5
	}
	if cfg.GCAD.MaxDeltaNorm <= 0 {
		cfg.GCAD.MaxDeltaNorm = 1.5
	}
	if cfg.GCAD.GateDecay <= 0 || cfg.GCAD.GateDecay >= 1.0 {
		cfg.GCAD.GateDecay = 0.95
	}

	basis := orthonormalizeSubspace(subspace)
	if len(basis) == 0 {
		return nil, fmt.Errorf("subspace basis has zero rank")
	}

	return &CrescendoDetector{
		cfg:     cfg,
		basis:   basis,
		dim:     dim,
		tracker: NewKVCacheAccumulationTracker(cfg.ThetaMaxDeg),
		window:  make([]crescendoWindowEntry, 0, cfg.WindowSize),
	}, nil
}

// orthonormalizeSubspace performs Modified Gram-Schmidt orthonormalization.
func orthonormalizeSubspace(vecs [][]float64) [][]float64 {
	var ortho [][]float64
	for _, v := range vecs {
		u := make([]float64, len(v))
		copy(u, v)
		for _, b := range ortho {
			dot := floats.Dot(u, b)
			floats.AddScaled(u, -dot, b)
		}
		norm := floats.Norm(u, 2)
		if norm > 1e-9 {
			floats.Scale(1.0/norm, u)
			ortho = append(ortho, u)
		}
	}
	return ortho
}

// SubspaceEnergy computes the normalized projection energy ||P_U x||_2 / ||x||_2 in [0, 1].
func (cd *CrescendoDetector) SubspaceEnergy(activation []float64) (float64, error) {
	if len(activation) != cd.dim {
		return 0, fmt.Errorf("activation dimension %d != subspace dimension %d", len(activation), cd.dim)
	}
	norm := floats.Norm(activation, 2)
	if norm < 1e-10 {
		return 0.0, nil
	}

	var projSq float64
	for _, b := range cd.basis {
		dot := floats.Dot(activation, b)
		projSq += dot * dot
	}
	ratio := math.Sqrt(projSq) / norm
	if ratio > 1.0 {
		ratio = 1.0
	}
	return ratio, nil
}

// ReAnchor re-runs subspace projection on the full conversation prefix without exponential
// GCAD decay attenuation and refreshes the KV-cache angular drift reference frame.
func (cd *CrescendoDetector) ReAnchor(ctx context.Context, prefix [][]float64) (float64, bool, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return 0, false, err
		}
	}
	if len(prefix) == 0 {
		return 0, false, fmt.Errorf("cannot re-anchor on empty prefix")
	}

	var weightedSum, weightTotal, maxScore float64
	var lastValid []float64

	for i, act := range prefix {
		if len(act) != cd.dim {
			return 0, false, fmt.Errorf("prefix turn %d dimension %d != %d", i, len(act), cd.dim)
		}
		score, err := cd.SubspaceEnergy(act)
		if err != nil {
			return 0, false, err
		}
		// Tail-weighted full-prefix scoring emphasizes recent turns while retaining full context
		w := 1.0 + float64(i)/float64(len(prefix))
		weightedSum += w * score
		weightTotal += w
		if score > maxScore {
			maxScore = score
		}
		if floats.Norm(act, 2) > 1e-9 {
			lastValid = act
		}
	}

	meanScore := 0.0
	if weightTotal > 0 {
		meanScore = weightedSum / weightTotal
	}
	// Blend tail-weighted mean and peak prefix energy
	prefixScore := 0.65*meanScore + 0.35*maxScore

	cd.mu.Lock()
	defer cd.mu.Unlock()

	if len(lastValid) > 0 {
		_ = cd.tracker.ReAnchorBaseline(lastValid)
	} else if len(cd.window) > 0 {
		lastSteer := cd.window[len(cd.window)-1].steerVec
		if len(lastSteer) > 0 {
			_ = cd.tracker.ReAnchorBaseline(lastSteer)
		}
	}

	cd.turnsSinceAnc = 0
	cd.lastPrefixScore = prefixScore
	cd.reAnchorTotal++

	escalated := prefixScore >= cd.cfg.EscalationThreshold
	return prefixScore, escalated, nil
}

// StepTurn processes a dialogue turn, applies exponential GCAD KV-cache decay to the sliding window,
// triggers periodic or drift-proactive full-prefix subspace re-anchoring so angular drift stays
// within the 15° budget, and evaluates whether a multi-turn crescendo escalation is underway.
func (cd *CrescendoDetector) StepTurn(ctx context.Context, turnIndex int, turnAct, steerVec []float64, prefix [][]float64) (CrescendoTurnResult, error) {
	if ctx != nil {
		if err := ctx.Err(); err != nil {
			return CrescendoTurnResult{}, err
		}
	}
	if len(turnAct) != cd.dim {
		return CrescendoTurnResult{}, fmt.Errorf("turn activation dimension %d != %d", len(turnAct), cd.dim)
	}
	if len(steerVec) == 0 {
		steerVec = turnAct
	}
	if len(steerVec) != cd.dim {
		return CrescendoTurnResult{}, fmt.Errorf("steering vector dimension %d != %d", len(steerVec), cd.dim)
	}

	rawScore, err := cd.SubspaceEnergy(turnAct)
	if err != nil {
		return CrescendoTurnResult{}, err
	}

	cd.mu.Lock()

	// Check prospective drift before recording; if close to saturating ThetaMaxDeg, re-anchor proactively
	prospectiveDrift := cd.computeProspectiveDriftLocked(steerVec)
	needsProactiveReAnchor := len(cd.window) > 0 && prospectiveDrift >= cd.cfg.ThetaMaxDeg*0.80
	if needsProactiveReAnchor {
		_ = cd.tracker.ReAnchorBaseline(steerVec)
		cd.turnsSinceAnc = 0
		cd.reAnchorTotal++
	}

	driftDeg, driftAlert, err := cd.tracker.RecordTurn(turnIndex, steerVec)
	if err != nil {
		cd.mu.Unlock()
		return CrescendoTurnResult{}, err
	}

	// Update sliding window with exponential GCAD decay on prior turns
	decay := cd.cfg.GCAD.GateDecay
	var decayedWindowSum, decayWeightSum float64
	for i := range cd.window {
		cd.window[i].decayedScore *= decay
		age := len(cd.window) - i
		w := math.Pow(decay, float64(age))
		decayedWindowSum += cd.window[i].subspaceScore * w
		decayWeightSum += w
	}

	actCopy := make([]float64, len(turnAct))
	copy(actCopy, turnAct)
	steerCopy := make([]float64, len(steerVec))
	copy(steerCopy, steerVec)

	isStepUp := false
	if len(cd.window) > 0 {
		prev := cd.window[len(cd.window)-1].subspaceScore
		isStepUp = rawScore > prev+1e-4
	}

	entry := crescendoWindowEntry{
		turnIndex:     turnIndex,
		activation:    actCopy,
		steerVec:      steerCopy,
		subspaceScore: rawScore,
		decayedScore:  rawScore,
	}
	if len(cd.window) >= cd.cfg.WindowSize {
		cd.window = cd.window[1:]
	}
	cd.window = append(cd.window, entry)
	cd.turnsSinceAnc++

	decayedScore := (decayedWindowSum + rawScore) / (decayWeightSum + 1.0)
	slope, posRatio := computeWindowLadderMetrics(cd.window)
	shouldReAnchor := needsProactiveReAnchor || (cd.turnsSinceAnc >= cd.cfg.ReAnchorCadence)

	var effectivePrefix [][]float64
	if shouldReAnchor {
		if len(prefix) > 0 {
			effectivePrefix = prefix
		} else {
			effectivePrefix = make([][]float64, len(cd.window))
			for i, w := range cd.window {
				effectivePrefix[i] = w.activation
			}
		}
	}
	windowLen := len(cd.window)
	cd.mu.Unlock()

	reAnchored := needsProactiveReAnchor
	fullPrefixScore := rawScore
	if shouldReAnchor && len(effectivePrefix) > 0 {
		pScore, _, rErr := cd.ReAnchor(ctx, effectivePrefix)
		if rErr != nil {
			return CrescendoTurnResult{}, rErr
		}
		// Ensure tracker baseline is anchored to current steerVec so incremental drift resets cleanly
		_ = cd.tracker.ReAnchorBaseline(steerCopy)
		driftDeg = 0.0
		driftAlert = false
		fullPrefixScore = pScore
		reAnchored = true
	} else {
		cd.mu.RLock()
		if cd.lastPrefixScore > 0 {
			fullPrefixScore = cd.lastPrefixScore
		}
		cd.mu.RUnlock()
	}

	// Combine full-prefix score, current raw score, and monotonic ladder bonus
	ladderBonus := 0.0
	if windowLen >= cd.cfg.MinLadderTurns && slope >= cd.cfg.MonotonicSlopeThreshold && posRatio >= 0.55 {
		ladderBonus = math.Min(0.35, slope*float64(windowLen)*0.6+0.10*posRatio)
	}
	escalationScore := math.Max(rawScore, 0.55*fullPrefixScore+0.45*rawScore) + ladderBonus
	if escalationScore > 1.0 {
		escalationScore = 1.0
	}

	isCrescendo := escalationScore >= cd.cfg.EscalationThreshold &&
		((windowLen >= cd.cfg.MinLadderTurns && slope >= cd.cfg.MonotonicSlopeThreshold && posRatio >= 0.55) ||
			fullPrefixScore >= cd.cfg.EscalationThreshold)

	if isCrescendo {
		cd.mu.Lock()
		cd.escalationTotal++
		cd.mu.Unlock()
	}

	return CrescendoTurnResult{
		TurnIndex:             turnIndex,
		SubspaceScore:         rawScore,
		DecayedScore:          decayedScore,
		FullPrefixScore:       fullPrefixScore,
		IncrementalDriftDeg:   driftDeg,
		MonotonicSlope:        slope,
		PositiveStepRatio:     posRatio,
		EscalationScore:       escalationScore,
		ReAnchored:            reAnchored,
		IsEscalationStep:      isStepUp,
		IsCrescendoEscalation: isCrescendo,
		DriftAlert:            driftAlert,
	}, nil
}

func (cd *CrescendoDetector) computeProspectiveDriftLocked(steerVec []float64) float64 {
	cd.tracker.mu.RLock()
	defer cd.tracker.mu.RUnlock()
	if len(cd.tracker.initialTurn) == 0 || len(cd.tracker.initialTurn) != len(steerVec) {
		return 0.0
	}
	initNorm := floats.Norm(cd.tracker.initialTurn, 2)
	norm := floats.Norm(steerVec, 2)
	if initNorm < 1e-9 || norm < 1e-9 {
		return 0.0
	}
	cosTheta := floats.Dot(cd.tracker.initialTurn, steerVec) / (initNorm * norm)
	if cosTheta > 1.0 {
		cosTheta = 1.0
	} else if cosTheta < -1.0 {
		cosTheta = -1.0
	}
	return math.Acos(cosTheta) * (180.0 / math.Pi)
}

func computeWindowLadderMetrics(window []crescendoWindowEntry) (slope float64, posRatio float64) {
	n := len(window)
	if n < 2 {
		return 0.0, 0.0
	}
	posSteps := 0
	var sumX, sumY, sumXY, sumX2 float64
	for i, w := range window {
		x := float64(i)
		y := w.subspaceScore
		sumX += x
		sumY += y
		sumXY += x * y
		sumX2 += x * x
		if i > 0 && w.subspaceScore > window[i-1].subspaceScore+1e-4 {
			posSteps++
		}
	}
	nf := float64(n)
	denom := nf*sumX2 - sumX*sumX
	if math.Abs(denom) > 1e-12 {
		slope = (nf*sumXY - sumX*sumY) / denom
	}
	posRatio = float64(posSteps) / float64(n-1)
	return slope, posRatio
}

// ReAnchorCount returns the total number of full-prefix re-anchor operations performed.
func (cd *CrescendoDetector) ReAnchorCount() int {
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	return cd.reAnchorTotal
}

// EscalationCount returns the total number of turns flagged as crescendo escalations.
func (cd *CrescendoDetector) EscalationCount() int {
	cd.mu.RLock()
	defer cd.mu.RUnlock()
	return cd.escalationTotal
}

// Tracker returns the underlying KVCacheAccumulationTracker.
func (cd *CrescendoDetector) Tracker() *KVCacheAccumulationTracker {
	return cd.tracker
}

// Reset clears the detector's sliding window and tracker state.
func (cd *CrescendoDetector) Reset() {
	cd.mu.Lock()
	defer cd.mu.Unlock()
	cd.window = cd.window[:0]
	cd.turnsSinceAnc = 0
	cd.lastPrefixScore = 0
	cd.reAnchorTotal = 0
	cd.escalationTotal = 0
	cd.tracker.Reset()
}
