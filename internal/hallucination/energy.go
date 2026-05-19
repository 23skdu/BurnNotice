package hallucination

import (
	"fmt"
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
