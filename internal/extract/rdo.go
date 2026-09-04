package extract

import (
	"math"

	burnmath "burnnotice/internal/math"
	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

// RDOConfig defines hyperparameters for Refusal Direction Optimization.
type RDOConfig struct {
	LearningRate float64
	Epochs       int
	L2Reg        float64
	Tolerance    float64
}

// DefaultRDOConfig returns standard recommended hyperparameters for RDO.
func DefaultRDOConfig() RDOConfig {
	return RDOConfig{
		LearningRate: 0.05,
		Epochs:       100,
		L2Reg:        0.01,
		Tolerance:    1e-5,
	}
}

// RefusalDirectionOptimizer refines behavioral directions using gradient descent
// on a linear probe loss (binary cross-entropy with L2 regularization).
type RefusalDirectionOptimizer struct {
	Config RDOConfig
}

// NewRefusalDirectionOptimizer constructs an optimizer with the given config.
func NewRefusalDirectionOptimizer(cfg RDOConfig) *RefusalDirectionOptimizer {
	if cfg.LearningRate <= 0 {
		cfg.LearningRate = 0.05
	}
	if cfg.Epochs <= 0 {
		cfg.Epochs = 100
	}
	if cfg.L2Reg < 0 {
		cfg.L2Reg = 0.01
	}
	if cfg.Tolerance <= 0 {
		cfg.Tolerance = 1e-5
	}
	return &RefusalDirectionOptimizer{Config: cfg}
}

// sigmoid computes a numerically stable logistic sigmoid function.
func sigmoid(z float64) float64 {
	if z >= 0 {
		return 1.0 / (1.0 + math.Exp(-z))
	}
	ez := math.Exp(z)
	return ez / (1.0 + ez)
}

// ComputeAUROC calculates the Area Under the Receiver Operating Characteristic
// curve using the Mann-Whitney U statistic.
func ComputeAUROC(posScores, negScores []float64) float64 {
	nPos := len(posScores)
	nNeg := len(negScores)
	if nPos == 0 || nNeg == 0 {
		return 0.5
	}

	var sumRank float64
	for _, p := range posScores {
		for _, n := range negScores {
			if p > n {
				sumRank += 1.0
			} else if p == n {
				sumRank += 0.5
			}
		}
	}
	return sumRank / float64(nPos*nNeg)
}

// Optimize refines a direction initialized from initDir (or PCA/DIM if empty)
// using gradient descent on linear probe classification loss.
// Returns the L2-normalized direction, the final AUROC, and ok=true on success.
func (rdo *RefusalDirectionOptimizer) Optimize(positive, negative [][]float64, initDir []float64) ([]float64, float64, bool) {
	if len(positive) == 0 || len(negative) == 0 {
		return nil, 0, false
	}
	d := len(positive[0])
	if d == 0 || len(negative[0]) != d {
		return nil, 0, false
	}

	for _, row := range positive {
		if len(row) != d {
			return nil, 0, false
		}
	}
	for _, row := range negative {
		if len(row) != d {
			return nil, 0, false
		}
	}

	// Apply robust winsorization to suppress activation outliers
	burnmath.WinsorizeMatrixInPlace(copyMatrix(positive), 0.005, 0.995)
	burnmath.WinsorizeMatrixInPlace(copyMatrix(negative), 0.005, 0.995)

	nPos := len(positive)
	nNeg := len(negative)
	totalSamples := nPos + nNeg

	// Pack samples and labels into matrices for vector math
	// X is (totalSamples x d), Y is (totalSamples x 1)
	xData := make([]float64, totalSamples*d)
	yData := make([]float64, totalSamples)

	for i, row := range positive {
		copy(xData[i*d:(i+1)*d], row)
		yData[i] = 1.0
	}
	for i, row := range negative {
		idx := nPos + i
		copy(xData[idx*d:(idx+1)*d], row)
		yData[idx] = 0.0
	}

	X := mat.NewDense(totalSamples, d, xData)

	// Initialize weight vector w
	w := make([]float64, d)
	if len(initDir) == d {
		copy(w, initDir)
	} else {
		// Fallback to PCA SVD initialization
		pcaDirs, _ := ExtractPCA(positive, negative, 1)
		if len(pcaDirs) > 0 && len(pcaDirs[0]) == d {
			copy(w, pcaDirs[0])
		} else {
			// Fallback to difference-in-means
			dimDir := ExtractDIM(positive, negative)
			if len(dimDir) == d {
				copy(w, dimDir)
			} else {
				for i := range w {
					w[i] = 1.0 / math.Sqrt(float64(d))
				}
			}
		}
	}

	// Ensure initial w is normalized
	wNorm := floats.Norm(w, 2)
	if wNorm > 1e-10 {
		floats.Scale(1.0/wNorm, w)
	}

	bias := 0.0
	prevLoss := math.MaxFloat64

	gradW := make([]float64, d)
	diff := make([]float64, totalSamples)

	// Gradient descent loop
	for epoch := 0; epoch < rdo.Config.Epochs; epoch++ {
		// Compute predictions: z = X * w + bias
		var loss float64
		for i := 0; i < totalSamples; i++ {
			row := X.RawRowView(i)
			z := floats.Dot(row, w) + bias
			p := sigmoid(z)
			y := yData[i]

			diff[i] = p - y

			// Binary cross-entropy loss with stability
			if y == 1.0 {
				loss += -math.Log(math.Max(p, 1e-15))
			} else {
				loss += -math.Log(math.Max(1.0-p, 1e-15))
			}
		}

		loss = loss/float64(totalSamples) + 0.5*rdo.Config.L2Reg*floats.Dot(w, w)

		// Check convergence
		if math.Abs(prevLoss-loss) < rdo.Config.Tolerance {
			break
		}
		prevLoss = loss

		// Compute gradients
		for j := range gradW {
			gradW[j] = rdo.Config.L2Reg * w[j]
		}
		var gradBias float64

		for i := 0; i < totalSamples; i++ {
			row := X.RawRowView(i)
			errVal := diff[i]
			floats.AddScaled(gradW, errVal/float64(totalSamples), row)
			gradBias += errVal
		}
		gradBias /= float64(totalSamples)

		// Update w and bias
		floats.AddScaled(w, -rdo.Config.LearningRate, gradW)
		bias -= rdo.Config.LearningRate * gradBias
	}

	// L2-normalize the resulting direction
	norm := floats.Norm(w, 2)
	if norm < 1e-10 {
		return nil, 0, false
	}
	floats.Scale(1.0/norm, w)

	// Compute final AUROC using projection scores
	posScores := make([]float64, nPos)
	for i, row := range positive {
		posScores[i] = floats.Dot(row, w)
	}
	negScores := make([]float64, nNeg)
	for i, row := range negative {
		negScores[i] = floats.Dot(row, w)
	}

	auroc := ComputeAUROC(posScores, negScores)
	// If AUROC < 0.5, direction is inverted; flip it so high score corresponds to positive class
	if auroc < 0.5 {
		floats.Scale(-1.0, w)
		for i := range posScores {
			posScores[i] = -posScores[i]
		}
		for i := range negScores {
			negScores[i] = -negScores[i]
		}
		auroc = ComputeAUROC(posScores, negScores)
	}

	return w, auroc, true
}

// ExtractRDO is a convenience function that initializes and runs RefusalDirectionOptimizer.
func ExtractRDO(positive, negative [][]float64, cfg *RDOConfig) ([]float64, float64, bool) {
	c := DefaultRDOConfig()
	if cfg != nil {
		c = *cfg
	}
	optimizer := NewRefusalDirectionOptimizer(c)
	return optimizer.Optimize(positive, negative, nil)
}
