package hallucination

import (
	"fmt"
	"math"
	"strings"
)

// HallucinationDetector computes energy metrics for hallucination detection.
type HallucinationDetector struct {
	layerThresholds map[int]float64
	spikeMultiplier float64
	windowSize      int
	history         map[int][]float64
}

// NewHallucinationDetector creates a detector with layer-specific thresholds.
func NewHallucinationDetector(layerThresholds map[int]float64, spikeMultiplier float64, windowSize int) *HallucinationDetector {
	return &HallucinationDetector{
		layerThresholds: layerThresholds,
		spikeMultiplier: spikeMultiplier,
		windowSize:      windowSize,
		history:         make(map[int][]float64),
	}
}

// ComputeEnergy computes the geometric energy metric: sum(feature_activations^2).
func (d *HallucinationDetector) ComputeEnergy(saeFeatures []float64) float64 {
	var energy float64
	for _, f := range saeFeatures {
		energy += f * f
	}
	return energy
}

// DetectHallucination checks if energy exceeds the layer-specific threshold.
func (d *HallucinationDetector) DetectHallucination(layerIdx int, saeFeatures []float64) bool {
	energy := d.ComputeEnergy(saeFeatures)
	threshold, ok := d.layerThresholds[layerIdx]
	if !ok {
		threshold = 1.0 // default threshold
	}
	return energy > threshold
}

// DetectSpike checks for sudden energy spikes compared to recent history.
func (d *HallucinationDetector) DetectSpike(layerIdx int, saeFeatures []float64) bool {
	if d.windowSize <= 0 {
		return false
	}
	energy := d.ComputeEnergy(saeFeatures)

	history := d.history[layerIdx]
	if len(history) < d.windowSize {
		d.history[layerIdx] = append(d.history[layerIdx], energy)
		return false
	}

	var sum float64
	for _, e := range history {
		sum += e
	}
	avg := sum / float64(len(history))

	d.history[layerIdx] = append(d.history[layerIdx][1:], energy)

	return energy > avg*d.spikeMultiplier
}

// UpdateHistory adds an energy value to the rolling window.
func (d *HallucinationDetector) UpdateHistory(layerIdx int, saeFeatures []float64) {
	if d.windowSize <= 0 {
		return
	}
	energy := d.ComputeEnergy(saeFeatures)
	history := d.history[layerIdx]
	if len(history) >= d.windowSize {
		history = history[1:]
	}
	d.history[layerIdx] = append(history, energy)
}

// BaselineEnergy returns the current average energy for a layer.
func (d *HallucinationDetector) BaselineEnergy(layerIdx int) float64 {
	history := d.history[layerIdx]
	if len(history) == 0 {
		return 0
	}
	var sum float64
	for _, e := range history {
		sum += e
	}
	return sum / float64(len(history))
}

// RiskScore computes a normalized hallucination risk score [0, 1].
func (d *HallucinationDetector) RiskScore(layerIdx int, saeFeatures []float64) float64 {
	energy := d.ComputeEnergy(saeFeatures)
	threshold, ok := d.layerThresholds[layerIdx]
	if !ok {
		threshold = 1.0
	}
	if threshold <= 0 {
		return 1.0
	}
	score := energy / threshold
	if score > 1.0 {
		return 1.0
	}
	return score
}

// ResetHistory clears the rolling energy history.
func (d *HallucinationDetector) ResetHistory() {
	d.history = make(map[int][]float64)
}

// Validate checks that the detector configuration is valid.
func (d *HallucinationDetector) Validate() error {
	if d.spikeMultiplier <= 1.0 {
		return fmt.Errorf("spikeMultiplier must be > 1.0")
	}
	if d.windowSize < 1 {
		return fmt.Errorf("windowSize must be >= 1")
	}
	return nil
}

// FamilyEnergyScale returns the canonical SAE feature-energy normalization factor
// for Gemma Scope, Llama-Scope, and Qwen-SAE dictionaries.
func FamilyEnergyScale(family string) float64 {
	switch strings.ToLower(strings.TrimSpace(family)) {
	case "gemma-scope", "gemma":
		return 1.0
	case "llama-scope", "llama":
		return 0.90
	case "qwen-sae", "qwen":
		return 0.95
	default:
		return 1.0
	}
}

// IngestFamilyFeatures normalizes SAE feature activations for a given SAE family
// (Gemma Scope, Llama-Scope, or Qwen-SAE), updates the rolling window via DetectSpike,
// and returns the risk score, spike flag, and hallucination flag.
func (d *HallucinationDetector) IngestFamilyFeatures(layerIdx int, saeFeatures []float64, family string) (risk float64, spike bool, hallucinating bool) {
	scale := math.Sqrt(FamilyEnergyScale(family))
	scaled := make([]float64, len(saeFeatures))
	for i, f := range saeFeatures {
		scaled[i] = f * scale
	}
	risk = d.RiskScore(layerIdx, scaled)
	hallucinating = d.DetectHallucination(layerIdx, scaled)
	spike = d.DetectSpike(layerIdx, scaled)
	return risk, spike, hallucinating
}
