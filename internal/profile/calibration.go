package profile

import (
	"fmt"
	"burnnotice/internal/datasets"
	"burnnotice/internal/scoring"
)

// Calibrator optimizes metric thresholds using precision-recall curves to maximize F1-score.
type Calibrator struct{}

// NewCalibrator creates a new calibrator.
func NewCalibrator() *Calibrator {
	return &Calibrator{}
}

// Calibrate computes optimal model-specific thresholds over a validation dataset.
func (c *Calibrator) Calibrate(prof *ModelProfile, dataset *datasets.Dataset) (map[string]float64, error) {
	if len(dataset.Pairs) == 0 {
		return nil, fmt.Errorf("empty calibration dataset")
	}

	scorer := scoring.NewScorer()
	for dirName, layers := range prof.Directions {
		for layerIdx, dir := range layers {
			dirKey := fmt.Sprintf("%s:%d", dirName, layerIdx)
			scorer.RegisterDirection(dirKey, dir)
		}
	}

	posScores := make(map[string][]float64)
	negScores := make(map[string][]float64)

	for _, pair := range dataset.Pairs {
		posActivation := activationsFromText(pair.Truthful.Prompt + " " + pair.Truthful.Response)
		negActivation := activationsFromText(pair.Deceptive.Prompt + " " + pair.Deceptive.Response)

		posResults := scorer.Score(posActivation)
		negResults := scorer.Score(negActivation)

		for k, s := range posResults {
			posScores[k] = append(posScores[k], s)
		}
		for k, s := range negResults {
			negScores[k] = append(negScores[k], s)
		}
	}

	optimalThresholds := make(map[string]float64)

	for dirKey := range posScores {
		pos := posScores[dirKey]
		neg := negScores[dirKey]

		if len(pos) == 0 || len(neg) == 0 {
			optimalThresholds[dirKey] = 0.5
			continue
		}

		var bestThreshold float64 = 0.5
		var bestF1 float64 = -1.0

		// Auto-detect whether higher scores represent deceptive (negative pair element) or truthful
		var sumPos, sumNeg float64
		for _, s := range pos {
			sumPos += s
		}
		for _, s := range neg {
			sumNeg += s
		}
		meanPos := sumPos / float64(len(pos))
		meanNeg := sumNeg / float64(len(neg))

		isGreaterRule := meanNeg > meanPos

		// Sweep thresholds from 0.0 to 1.0 with step 0.01
		for t := 0.0; t <= 1.0; t += 0.01 {
			var tp, fp, fn int

			for _, s := range neg {
				if isGreaterRule {
					if s > t {
						tp++
					} else {
						fn++
					}
				} else {
					if s < t {
						tp++
					} else {
						fn++
					}
				}
			}

			for _, s := range pos {
				if isGreaterRule {
					if s > t {
						fp++
					}
				} else {
					if s < t {
						fp++
					}
				}
			}

			var precision, recall, f1 float64
			if tp+fp > 0 {
				precision = float64(tp) / float64(tp+fp)
			}
			if tp+fn > 0 {
				recall = float64(tp) / float64(tp+fn)
			}
			if precision+recall > 0 {
				f1 = 2 * precision * recall / (precision + recall)
			}

			if f1 > bestF1 {
				bestF1 = f1
				bestThreshold = t
			}
		}

		optimalThresholds[dirKey] = bestThreshold
	}

	return optimalThresholds, nil
}
