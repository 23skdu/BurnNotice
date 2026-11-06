package math

import (
	"math"
	"testing"
)

func TestWinsorize(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10}
	result := Winsorize(input, 0.1, 0.9)
	if len(result) != 10 {
		t.Fatalf("len = %d, want 10", len(result))
	}
	lo, hi := result[0], result[9]
	if lo != 2 {
		t.Errorf("min after winsorize = %f, want 2", lo)
	}
	if hi != 9 {
		t.Errorf("max after winsorize = %f, want 9", hi)
	}
	// Original must be unmodified
	if input[0] != 1 {
		t.Errorf("original modified: input[0] = %f", input[0])
	}
}

func TestWinsorize_Empty(t *testing.T) {
	if r := Winsorize(nil, 0.005, 0.995); r != nil {
		t.Error("expected nil for nil input")
	}
	if r := Winsorize([]float64{}, 0.005, 0.995); r != nil {
		t.Error("expected nil for empty input")
	}
}

func TestWinsorize_AllSame(t *testing.T) {
	input := []float64{5, 5, 5, 5, 5}
	result := Winsorize(input, 0.1, 0.9)
	for i, v := range result {
		if v != 5 {
			t.Errorf("result[%d] = %f, want 5", i, v)
		}
	}
}

func TestWinsorizeInPlace(t *testing.T) {
	input := []float64{1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 100}
	WinsorizeInPlace(input, 0.1, 0.8)
	// n=11, loIdx=int(11*0.1)=1 → sorted[1]=2, hiIdx=int(11*0.8)-1=7 → sorted[7]=8
	if input[0] != 2 {
		t.Errorf("min after winsorize = %f, want 2", input[0])
	}
	if input[10] != 8 {
		t.Errorf("max after winsorize = %f, want 8", input[10])
	}
}

func TestWinsorizeInPlace_Empty(t *testing.T) {
	WinsorizeInPlace(nil, 0.005, 0.995)
	WinsorizeInPlace([]float64{}, 0.005, 0.995)
}

func TestIsStable(t *testing.T) {
	if !IsStable([]float64{1, 2, 3}) {
		t.Error("expected stable")
	}
	if IsStable([]float64{1, math.NaN(), 3}) {
		t.Error("expected unstable for NaN")
	}
	if IsStable([]float64{1, math.Inf(1), 3}) {
		t.Error("expected unstable for +Inf")
	}
	if IsStable([]float64{1, math.Inf(-1), 3}) {
		t.Error("expected unstable for -Inf")
	}
}

func TestClampToFloat64(t *testing.T) {
	input := []float64{1, math.NaN(), math.Inf(1), math.Inf(-1), -2}
	result := ClampToFloat64(input)
	if result[0] != 1 {
		t.Errorf("result[0] = %f, want 1", result[0])
	}
	if result[1] != 0 {
		t.Errorf("NaN clamped to %f, want 0", result[1])
	}
	if result[2] != maxFloat64 {
		t.Errorf("+Inf clamped to %f, want maxFloat64", result[2])
	}
	if result[3] != -maxFloat64 {
		t.Errorf("-Inf clamped to %f, want -maxFloat64", result[3])
	}
	if result[4] != -2 {
		t.Errorf("result[4] = %f, want -2", result[4])
	}
}

func TestWinsorizeMatrixInPlace(t *testing.T) {
	// 3 samples, 2 features
	data := [][]float64{
		{1.0, 10.0},
		{2.0, 20.0},
		{3.0, 30.0},
	}
	// Winsorize each column with lower=0.3, upper=0.7
	// Column 0: [1, 2, 3] -> n=3. loIdx=int(3*0.3)=0 (val 1.0), hiIdx=int(3*0.7)-1=1 (val 2.0)
	// Column 1: [10, 20, 30] -> loIdx=0 (val 10.0), hiIdx=1 (val 20.0)
	WinsorizeMatrixInPlace(data, 0.3, 0.7)

	if data[0][0] != 1.0 || data[1][0] != 2.0 || data[2][0] != 2.0 {
		t.Errorf("unexpected column 0 winsorized: %v", data)
	}
	if data[0][1] != 10.0 || data[1][1] != 20.0 || data[2][1] != 20.0 {
		t.Errorf("unexpected column 1 winsorized: %v", data)
	}

	// Empty matrix
	WinsorizeMatrixInPlace(nil, 0.1, 0.9)
	WinsorizeMatrixInPlace([][]float64{}, 0.1, 0.9)
}

func TestWinsorize_ExtremeQuantiles(t *testing.T) {
	input := []float64{1, 2, 3}
	// lower < 0, upper > 1
	r := Winsorize(input, -0.5, 1.5)
	if r[0] != 1 || r[2] != 3 {
		t.Errorf("expected no winsorization for extreme quantiles, got %v", r)
	}

	// lower > 1, upper < 0
	r2 := Winsorize(input, 1.5, -0.5)
	if len(r2) != 3 {
		t.Fatal("expected length 3")
	}
}

func TestWinsorizeInPlace_ExtremeClamping(t *testing.T) {
	// lower < 0, upper > 1
	input1 := []float64{1, 2, 3}
	WinsorizeInPlace(input1, -0.5, 1.5)
	if input1[0] != 1 || input1[2] != 3 {
		t.Errorf("expected unmodified for lower<0 upper>1, got %v", input1)
	}

	// lower > 1, upper < 0
	input2 := []float64{1, 2, 3}
	WinsorizeInPlace(input2, 1.5, -0.5)
	if input2[0] != 3 || input2[1] != 3 || input2[2] != 3 {
		t.Errorf("expected all clamped to loVal, got %v", input2)
	}
}
