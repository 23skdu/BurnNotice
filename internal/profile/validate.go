package profile

import (
	"fmt"
	"math"
	"sort"

	"burnnotice/internal/datasets"
	"burnnotice/internal/scoring"
)

// ValidationResult holds validation metrics.
type ValidationResult struct {
	DirectionName string
	AUROC         float64
	FPR           float64
	Precision     float64
	Recall        float64
	Passed        bool
}

// Validator validates profile directions against held-out data.
type Validator struct {
	minAUROC float64
}

// NewValidator creates a validator.
func NewValidator(minAUROC float64) *Validator {
	if minAUROC <= 0 {
		minAUROC = 0.9
	}
	return &Validator{minAUROC: minAUROC}
}

// Validate validates a profile's directions against a dataset.
func (v *Validator) Validate(profile *ModelProfile, dataset *datasets.Dataset) []ValidationResult {
	var results []ValidationResult

	scorer := scoring.NewScorer(nil)

	for dirName, layers := range profile.Directions {
		for layerIdx, dir := range layers {
			dirKey := fmt.Sprintf("%s:%d", dirName, layerIdx)
			scorer.RegisterDirection(dirKey, dir)
		}
	}

	var scores []float64
	var labels []bool

	for _, pair := range dataset.Pairs {
		posActivation := activationsFromText(pair.Truthful.Prompt + " " + pair.Truthful.Response)
		negActivation := activationsFromText(pair.Deceptive.Prompt + " " + pair.Deceptive.Response)

		posScores := scorer.Score(posActivation)
		negScores := scorer.Score(negActivation)

		for dirName := range profile.Directions {
			dirKey := fmt.Sprintf("%s:0", dirName)
			if posScore, ok := posScores[dirKey]; ok {
				scores = append(scores, posScore)
				labels = append(labels, true)
			}
			if negScore, ok := negScores[dirKey]; ok {
				scores = append(scores, negScore)
				labels = append(labels, false)
			}
		}
	}

	for dirName := range profile.Directions {
		result := v.computeMetrics(dirName, scores, labels)
		result.Passed = result.AUROC >= v.minAUROC
		results = append(results, result)
	}

	return results
}

func (v *Validator) computeMetrics(dirName string, scores []float64, labels []bool) ValidationResult {
	if len(scores) < 2 {
		return ValidationResult{DirectionName: dirName}
	}

	type scoredLabel struct {
		score float64
		label bool
	}

	pairs := make([]scoredLabel, len(scores))
	for i := range scores {
		pairs[i] = scoredLabel{scores[i], labels[i]}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].score > pairs[j].score
	})

	var tp, fp, tn, fn int
	var auc float64
	var prevScore float64
	var prevTP, prevFP int

	for i, p := range pairs {
		if p.score != prevScore && i > 0 {
			auc += float64(prevTP+tp) * float64(fp-prevFP) / 2.0
			prevTP = tp
			prevFP = fp
		}

		if p.label {
			tp++
		} else {
			fp++
		}
		prevScore = p.score
	}

	auc += float64(prevTP+tp) * float64(fp-prevFP) / 2.0

	totalPos := 0
	totalNeg := 0
	for _, l := range labels {
		if l {
			totalPos++
		} else {
			totalNeg++
		}
	}

	if totalPos > 0 && totalNeg > 0 {
		auc /= float64(totalPos * totalNeg)
	}

	tp = 0
	fp = 0
	fn = totalPos
	tn = totalNeg

	bestFPR := 1.0
	bestPrecision := 0.0
	bestRecall := 0.0

	for _, p := range pairs {
		if p.label {
			tp++
			fn--
		} else {
			fp++
			tn--
		}

		fpr := 0.0
		if totalNeg > 0 {
			fpr = float64(fp) / float64(totalNeg)
		}
		precision := 0.0
		if tp+fp > 0 {
			precision = float64(tp) / float64(tp+fp)
		}
		recall := 0.0
		if totalPos > 0 {
			recall = float64(tp) / float64(totalPos)
		}

		if fpr < bestFPR {
			bestFPR = fpr
			bestPrecision = precision
			bestRecall = recall
		}
	}

	return ValidationResult{
		DirectionName: dirName,
		AUROC:         auc,
		FPR:           bestFPR,
		Precision:     bestPrecision,
		Recall:        bestRecall,
	}
}

func activationsFromText(text string) []float64 {
	activation := make([]float64, 4096)
	for i := 0; i < len(text) && i < len(activation); i++ {
		activation[i] = float64(text[i]) / 255.0
	}
	for i := len(text); i < len(activation); i++ {
		activation[i] = math.Sin(float64(i)) * 0.1
	}
	return activation
}
