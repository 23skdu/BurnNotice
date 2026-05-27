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
