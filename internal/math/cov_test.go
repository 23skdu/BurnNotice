package math

import (
	"math"
	"testing"
)

func TestEstimateCovariance(t *testing.T) {
	// 4 samples, 2 features
	X := [][]float64{
		{1.0, 2.0},
		{2.0, 4.0},
		{3.0, 6.0},
		{4.0, 8.0},
	}

	// Test pure empirical (alpha = 0.0)
	empCov := EstimateCovariance(X, 0.0)
	if empCov == nil {
		t.Fatal("Expected covariance matrix, got nil")
	}

	// Variance of [1, 2, 3, 4] is 5/3 ~ 1.666
	// Variance of [2, 4, 6, 8] is 20/3 ~ 6.666
	// Covariance is 10/3 ~ 3.333
	expectedVar1 := 5.0 / 3.0
	expectedVar2 := 20.0 / 3.0
	expectedCov := 10.0 / 3.0

	if math.Abs(empCov.At(0, 0)-expectedVar1) > 1e-5 {
		t.Errorf("Expected Var1 %.4f, got %.4f", expectedVar1, empCov.At(0, 0))
	}
	if math.Abs(empCov.At(1, 1)-expectedVar2) > 1e-5 {
		t.Errorf("Expected Var2 %.4f, got %.4f", expectedVar2, empCov.At(1, 1))
	}
	if math.Abs(empCov.At(0, 1)-expectedCov) > 1e-5 {
		t.Errorf("Expected Cov %.4f, got %.4f", expectedCov, empCov.At(0, 1))
	}

	// Test pure shrinkage (alpha = 1.0)
	// Trace = 25/3, avgVar = 25/6 ~ 4.1666
	shrunkCov := EstimateCovariance(X, 1.0)
	expectedAvgVar := (expectedVar1 + expectedVar2) / 2.0
	if math.Abs(shrunkCov.At(0, 0)-expectedAvgVar) > 1e-5 {
		t.Errorf("Expected Shrunk Var1 %.4f, got %.4f", expectedAvgVar, shrunkCov.At(0, 0))
	}
	if math.Abs(shrunkCov.At(1, 1)-expectedAvgVar) > 1e-5 {
		t.Errorf("Expected Shrunk Var2 %.4f, got %.4f", expectedAvgVar, shrunkCov.At(1, 1))
	}
	if math.Abs(shrunkCov.At(0, 1)) > 1e-5 {
		t.Errorf("Expected Shrunk Cov 0.0, got %.4f", shrunkCov.At(0, 1))
	}

	// Test partial shrinkage (alpha = 0.5)
	partialCov := EstimateCovariance(X, 0.5)
	expectedPartialVar1 := 0.5*expectedVar1 + 0.5*expectedAvgVar
	if math.Abs(partialCov.At(0, 0)-expectedPartialVar1) > 1e-5 {
		t.Errorf("Expected Partial Var1 %.4f, got %.4f", expectedPartialVar1, partialCov.At(0, 0))
	}

	// Test optimal Ledoit-Wolf shrinkage (alpha < 0)
	lwCov := EstimateCovariance(X, -1.0)
	if lwCov == nil {
		t.Fatal("Expected Ledoit-Wolf covariance matrix, got nil")
	}

	// Test empty/nil inputs
	if EstimateCovariance(nil, 0.5) != nil {
		t.Error("expected nil for nil matrix")
	}
	if EstimateCovariance([][]float64{}, 0.5) != nil {
		t.Error("expected nil for empty matrix")
	}
	if EstimateCovariance([][]float64{{}}, 0.5) != nil {
		t.Error("expected nil for matrix with empty row")
	}

	// Test zero variance input (leads to diffNormSq == 0)
	zeroVarX := [][]float64{
		{1.0, 2.0},
		{1.0, 2.0},
		{1.0, 2.0},
	}
	zeroCov := EstimateCovariance(zeroVarX, -1.0)
	if zeroCov == nil {
		t.Fatal("Expected zero covariance matrix, got nil")
	}
}

func TestCovariance_EdgeCases(t *testing.T) {
	// Covariance and ShrinkageCovariance edge cases
	if Covariance(nil) != nil {
		t.Error("expected nil for nil input in Covariance")
	}
	if Covariance([][]float64{}) != nil {
		t.Error("expected nil for empty input in Covariance")
	}
	if Covariance([][]float64{{1, 2}}) != nil {
		t.Error("expected nil for single row in Covariance")
	}
	if Covariance([][]float64{{1, 0}, {1}}) != nil {
		t.Error("expected nil for ragged input in Covariance")
	}
	if Covariance([][]float64{{}, {}}) != nil {
		t.Error("expected nil for empty cols in Covariance")
	}

	if ShrinkageCovariance(nil, 0.5) != nil {
		t.Error("expected nil for nil input in ShrinkageCovariance")
	}
	if ShrinkageCovariance([][]float64{}, 0.5) != nil {
		t.Error("expected nil for empty input in ShrinkageCovariance")
	}
	if ShrinkageCovariance([][]float64{{1, 2}}, 0.5) != nil {
		t.Error("expected nil for single row in ShrinkageCovariance")
	}
	if ShrinkageCovariance([][]float64{{1, 0}, {1}}, 0.5) != nil {
		t.Error("expected nil for ragged input in ShrinkageCovariance")
	}
	if ShrinkageCovariance([][]float64{{}, {}}, 0.5) != nil {
		t.Error("expected nil for empty cols in ShrinkageCovariance")
	}

	// columnMean edge cases
	if columnMean(nil) != nil {
		t.Error("expected nil for nil input in columnMean")
	}

	// AverageVariance edge cases
	if AverageVariance(nil) != 0 {
		t.Error("expected 0 for nil input in AverageVariance")
	}
}
