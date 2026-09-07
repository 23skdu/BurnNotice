package profile

import (
	"encoding/gob"
	"fmt"
	"math"
	"os"
	"sync"
	"time"

	"burnnotice/internal/datasets"
	"burnnotice/internal/scoring"
)

// CalibratedThreshold holds an optimal threshold with its ROC metrics.
type CalibratedThreshold struct {
	DirectionName string  `json:"direction_name"`
	LayerIndex    int     `json:"layer_index"`
	Threshold     float64 `json:"threshold"`
	F1Score       float64 `json:"f1_score"`
	Precision     float64 `json:"precision"`
	Recall        float64 `json:"recall"`
	AUROC         float64 `json:"auroc"`
	Direction     string  `json:"-"` // "greater" or "less"
}

// CalibrationResult holds all calibrated thresholds for a model profile.
type CalibrationResult struct {
	ModelID    string                          `json:"model_id"`
	Thresholds map[string]*CalibratedThreshold `json:"thresholds"`
	Timestamp  int64                           `json:"timestamp"`
	DatasetSize int                            `json:"dataset_size"`
}

// Save persists calibration results to a gob file alongside the profile.
func (cr *CalibrationResult) Save(dir string) error {
	path := dir + "/calibration.gob"
	// #nosec G304 -- profile dir is controlled by application config
	f, err := os.Create(path)
	if err != nil {
		return fmt.Errorf("create calibration file: %w", err)
	}
	defer f.Close()
	return gob.NewEncoder(f).Encode(cr)
}

// LoadCalibration loads calibration results from a gob file.
func LoadCalibration(dir string) (*CalibrationResult, error) {
	path := dir + "/calibration.gob"
	// #nosec G304 -- profile dir is controlled by application config
	f, err := os.Open(path)
	if err != nil {
		return nil, fmt.Errorf("open calibration file: %w", err)
	}
	defer f.Close()
	var cr CalibrationResult
	if err := gob.NewDecoder(f).Decode(&cr); err != nil {
		return nil, fmt.Errorf("decode calibration: %w", err)
	}
	return &cr, nil
}

// DriftDetector monitors rolling average projection drift and triggers recalibration.
type DriftDetector struct {
	mu             sync.Mutex
	window         []float64
	windowSize     int
	mean           float64
	variance       float64
	lastRecalibrated time.Time
	sigmaThreshold float64
}

// NewDriftDetector creates a drift detector with the given window size and sigma threshold.
func NewDriftDetector(windowSize int, sigmaThreshold float64) *DriftDetector {
	if windowSize <= 0 {
		windowSize = 100
	}
	if sigmaThreshold <= 0 {
		sigmaThreshold = 1.5
	}
	return &DriftDetector{
		windowSize:     windowSize,
		sigmaThreshold: sigmaThreshold,
	}
}

// Record adds a projection drift measurement and returns true if recalibration is needed.
func (d *DriftDetector) Record(drift float64) bool {
	d.mu.Lock()
	defer d.mu.Unlock()

	d.window = append(d.window, drift)
	if len(d.window) > d.windowSize {
		d.window = d.window[1:]
	}

	if len(d.window) < 2 {
		return false
	}

	var sum float64
	for _, v := range d.window {
		sum += v
	}
	d.mean = sum / float64(len(d.window))

	var sqDiff float64
	for _, v := range d.window {
		diff := v - d.mean
		sqDiff += diff * diff
	}
	d.variance = sqDiff / float64(len(d.window))
	stdDev := math.Sqrt(d.variance)

	if stdDev == 0 {
		return false
	}

	return math.Abs(drift-d.mean)/stdDev > d.sigmaThreshold
}

// LastRecalibrated returns when calibration was last performed.
func (d *DriftDetector) LastRecalibrated() time.Time {
	d.mu.Lock()
	defer d.mu.Unlock()
	return d.lastRecalibrated
}

// MarkRecalibrated records that calibration was performed.
func (d *DriftDetector) MarkRecalibrated() {
	d.mu.Lock()
	defer d.mu.Unlock()
	d.lastRecalibrated = time.Now()
}

// Calibrator optimizes metric thresholds using precision-recall curves to maximize F1-score.
type Calibrator struct{}

// NewCalibrator creates a new calibrator.
func NewCalibrator() *Calibrator {
	return &Calibrator{}
}

// Calibrate computes optimal model-specific thresholds over a validation dataset.
func (c *Calibrator) Calibrate(prof *ModelProfile, dataset *datasets.Dataset) (*CalibrationResult, error) {
	thresholds, err := c.calibrateInternal(prof, dataset)
	if err != nil {
		return nil, err
	}

	result := &CalibrationResult{
		ModelID:     prof.ModelID,
		Thresholds:  thresholds,
		Timestamp:   time.Now().Unix(),
		DatasetSize: len(dataset.Pairs),
	}
	return result, nil
}

func (c *Calibrator) calibrateInternal(prof *ModelProfile, dataset *datasets.Dataset) (map[string]*CalibratedThreshold, error) {
	if len(dataset.Pairs) == 0 {
		return nil, fmt.Errorf("empty calibration dataset")
	}

	scorer := scoring.NewScorer(nil)
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

	optimalThresholds := make(map[string]*CalibratedThreshold)

	for dirKey := range posScores {
		pos := posScores[dirKey]
		neg := negScores[dirKey]

		if len(pos) == 0 || len(neg) == 0 {
			optimalThresholds[dirKey] = &CalibratedThreshold{
				Threshold: 0.5,
			}
			continue
		}

		var bestThreshold float64 = 0.5
		var bestF1 float64 = -1.0
		var bestPrecision, bestRecall float64

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
				bestPrecision = precision
				bestRecall = recall
			}
		}

		dirLabel := "less"
		if isGreaterRule {
			dirLabel = "greater"
		}

		optimalThresholds[dirKey] = &CalibratedThreshold{
			Threshold: bestThreshold,
			F1Score:   bestF1,
			Precision: bestPrecision,
			Recall:    bestRecall,
			Direction: dirLabel,
		}
	}

	return optimalThresholds, nil
}

// AdjustThresholdForCone adjusts the baseline detection threshold based on the concept cone dimensionality.
// Higher-dimensional concept cones naturally capture more variance, requiring scaled thresholds to maintain constant false alarm rates.
func AdjustThresholdForCone(baseThreshold float64, coneDimension int) float64 {
	if coneDimension <= 1 {
		return baseThreshold
	}
	// Scale threshold with square root of cone dimension
	scale := math.Sqrt(float64(coneDimension))
	adjusted := baseThreshold * scale
	// Clamp to maximum 0.95 to preserve sensitivity
	if adjusted > 0.95 {
		adjusted = 0.95
	}
	return adjusted
}

// RecommendedINLPIterations returns the suggested number of Iterative Nullspace Projections for a given cone dimensionality.
func RecommendedINLPIterations(coneDimension int) int {
	if coneDimension <= 1 {
		return 5
	}
	iterations := 2 * coneDimension
	if iterations < 5 {
		iterations = 5
	} else if iterations > 40 {
		iterations = 40
	}
	return iterations
}
