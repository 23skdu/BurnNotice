package steer

import (
	"fmt"
	"math"
	"sync"

	"gonum.org/v1/gonum/floats"
)

// ThetaMaxDegrees defines the maximum permissible cumulative angular drift in degrees (15.0°).
const ThetaMaxDegrees = 15.0

// TurnSteeringRecord captures the steering intervention applied during a single dialogue turn.
type TurnSteeringRecord struct {
	TurnIndex    int       `json:"turn_index"`
	Vector       []float64 `json:"vector"`
	Norm         float64   `json:"norm"`
	AngularDrift float64   `json:"angular_drift_deg"` // angular deviation from initial baseline turn (degrees)
	ExceedsLimit bool      `json:"exceeds_limit"`
}

// KVCacheAccumulationTracker tracks steering vector evolution across multi-turn sessions
// to prevent cumulative KV-cache corruption and attention distortion.
type KVCacheAccumulationTracker struct {
	mu          sync.RWMutex
	initialTurn []float64
	history     []TurnSteeringRecord
	thetaMaxDeg float64
	alarmCount  int
}

// NewKVCacheAccumulationTracker creates a new tracker initialized with default ThetaMax (15°).
func NewKVCacheAccumulationTracker(thetaMax float64) *KVCacheAccumulationTracker {
	if thetaMax <= 0 {
		thetaMax = ThetaMaxDegrees
	}
	return &KVCacheAccumulationTracker{
		thetaMaxDeg: thetaMax,
		history:     make([]TurnSteeringRecord, 0),
	}
}

// RecordTurn logs a turn steering vector and computes cumulative angular drift from the session baseline.
// Returns the angular drift in degrees, and whether the safety threshold was violated.
func (t *KVCacheAccumulationTracker) RecordTurn(turnIndex int, steerVec []float64) (driftDeg float64, alert bool, err error) {
	if len(steerVec) == 0 {
		return 0, false, fmt.Errorf("empty steering vector")
	}

	norm := floats.Norm(steerVec, 2)
	if norm < 1e-9 {
		return 0, false, fmt.Errorf("steering vector norm is zero")
	}

	t.mu.Lock()
	defer t.mu.Unlock()

	// Initialize baseline direction on first turn
	if len(t.initialTurn) == 0 {
		t.initialTurn = make([]float64, len(steerVec))
		copy(t.initialTurn, steerVec)
		rec := TurnSteeringRecord{
			TurnIndex:    turnIndex,
			Vector:       steerVec,
			Norm:         norm,
			AngularDrift: 0.0,
			ExceedsLimit: false,
		}
		t.history = append(t.history, rec)
		return 0.0, false, nil
	}

	if len(steerVec) != len(t.initialTurn) {
		return 0, false, fmt.Errorf("vector dimension %d mismatches initial baseline %d", len(steerVec), len(t.initialTurn))
	}

	// Compute angle between initialTurn and steerVec
	initNorm := floats.Norm(t.initialTurn, 2)
	dot := floats.Dot(t.initialTurn, steerVec)
	cosTheta := dot / (initNorm * norm)

	if cosTheta > 1.0 {
		cosTheta = 1.0
	} else if cosTheta < -1.0 {
		cosTheta = -1.0
	}

	rad := math.Acos(cosTheta)
	driftDeg = rad * (180.0 / math.Pi)
	alert = driftDeg > t.thetaMaxDeg

	if alert {
		t.alarmCount++
	}

	rec := TurnSteeringRecord{
		TurnIndex:    turnIndex,
		Vector:       steerVec,
		Norm:         norm,
		AngularDrift: driftDeg,
		ExceedsLimit: alert,
	}
	t.history = append(t.history, rec)

	return driftDeg, alert, nil
}

// History returns a copy of the turn records.
func (t *KVCacheAccumulationTracker) History() []TurnSteeringRecord {
	t.mu.RLock()
	defer t.mu.RUnlock()
	copied := make([]TurnSteeringRecord, len(t.history))
	copy(copied, t.history)
	return copied
}

// AlarmCount returns how many turns exceeded ThetaMax.
func (t *KVCacheAccumulationTracker) AlarmCount() int {
	t.mu.RLock()
	defer t.mu.RUnlock()
	return t.alarmCount
}

// Reset clears the tracker state.
func (t *KVCacheAccumulationTracker) Reset() {
	t.mu.Lock()
	defer t.mu.Unlock()
	t.initialTurn = nil
	t.history = nil
	t.alarmCount = 0
}

// GCADConfig specifies parameters for Gated Cropped Attention-Delta defense.
type GCADConfig struct {
	MaxDeltaNorm float64 // Maximum permissible norm for attention key/value deltas (tau)
	GateDecay    float64 // Exponential attenuation factor for historical KV tokens (e.g. 0.95)
}

// DefaultGCADConfig returns recommended production parameters for attention protection.
func DefaultGCADConfig() GCADConfig {
	return GCADConfig{
		MaxDeltaNorm: 1.5,
		GateDecay:    0.95,
	}
}

// GCADApply crops and attenuates attention-delta vectors between steered and unsteered states.
// For key/value activations, delta = steered - unsteered.
// If ||delta|| > MaxDeltaNorm, delta is cropped to MaxDeltaNorm.
// Additionally applies historical distance decay (decay^(turnDelta)) to prevent deep KV contamination.
func GCADApply(steered, unsteered []float64, turnDelta int, cfg GCADConfig) ([]float64, bool) {
	if len(steered) == 0 || len(steered) != len(unsteered) {
		return steered, false
	}

	dim := len(steered)
	delta := make([]float64, dim)
	for i := 0; i < dim; i++ {
		delta[i] = steered[i] - unsteered[i]
	}

	deltaNorm := floats.Norm(delta, 2)
	wasCropped := false

	maxNorm := cfg.MaxDeltaNorm
	if maxNorm <= 0 {
		maxNorm = 1.5
	}

	// Crop if delta exceeds bound
	if deltaNorm > maxNorm && deltaNorm > 1e-9 {
		scale := maxNorm / deltaNorm
		floats.Scale(scale, delta)
		wasCropped = true
	}

	// Apply age decay
	if turnDelta > 0 && cfg.GateDecay > 0 && cfg.GateDecay < 1.0 {
		decay := math.Pow(cfg.GateDecay, float64(turnDelta))
		floats.Scale(decay, delta)
		wasCropped = true
	}

	// Reconstruct bounded steered activation: unsteered + bounded delta
	result := make([]float64, dim)
	for i := 0; i < dim; i++ {
		result[i] = unsteered[i] + delta[i]
	}

	return result, wasCropped
}
