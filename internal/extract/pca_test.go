package extract

import (
	"math"
	"testing"
)

func TestExtractPCA(t *testing.T) {
	positive := [][]float64{
		{1, 0, 0},
		{1.1, 0.1, 0},
		{0.9, -0.1, 0},
	}
	negative := [][]float64{
		{0, 1, 0},
		{0.1, 1.1, 0},
		{-0.1, 0.9, 0},
	}

	directions, singularValues := ExtractPCA(positive, negative, 1)

	if directions == nil {
		t.Fatal("ExtractPCA returned nil directions")
	}
	if len(directions) != 1 {
		t.Fatalf("len(directions) = %d, want 1", len(directions))
	}
	if len(directions[0]) != 3 {
		t.Fatalf("len(directions[0]) = %d, want 3", len(directions[0]))
	}

	norm := Norm(directions[0])
	if math.Abs(norm-1.0) > 1e-6 {
		t.Errorf("direction norm = %f, want 1.0", norm)
	}

	if singularValues == nil {
		t.Fatal("ExtractPCA returned nil singular values")
	}
	if len(singularValues) != 1 {
		t.Fatalf("len(singularValues) = %d, want 1", len(singularValues))
	}
	if singularValues[0] <= 0 {
		t.Errorf("singular value = %f, want > 0", singularValues[0])
	}
}

func TestExtractPCA_Empty(t *testing.T) {
	d, s := ExtractPCA(nil, [][]float64{{1}}, 1)
	if d != nil || s != nil {
		t.Errorf("ExtractPCA(nil, ...) = %v, %v, want nil, nil", d, s)
	}
}

func TestExtractPCA_K_LargerThanDims(t *testing.T) {
	positive := [][]float64{{1, 2}, {3, 4}}
	negative := [][]float64{{0, 0}, {0, 0}}

	directions, _ := ExtractPCA(positive, negative, 10)

	if directions == nil {
		t.Fatal("ExtractPCA returned nil")
	}
	if len(directions) != 2 {
		t.Errorf("len(directions) = %d, want 2 (clamped to cols)", len(directions))
	}
}

func TestExtractPCA_MultipleComponents(t *testing.T) {
	positive := [][]float64{
		{1, 0, 0, 0},
		{0, 1, 0, 0},
		{1, 1, 0, 0},
	}
	negative := [][]float64{
		{0, 0, 1, 0},
		{0, 0, 0, 1},
		{0, 0, 1, 1},
	}

	directions, singularValues := ExtractPCA(positive, negative, 2)

	if len(directions) != 2 {
		t.Fatalf("len(directions) = %d, want 2", len(directions))
	}
	if len(singularValues) != 2 {
		t.Fatalf("len(singularValues) = %d, want 2", len(singularValues))
	}

	for i, d := range directions {
		n := Norm(d)
		if math.Abs(n-1.0) > 1e-6 {
			t.Errorf("direction[%d] norm = %f, want 1.0", i, n)
		}
	}

	if singularValues[0] < singularValues[1] {
		t.Errorf("singular values not sorted descending: %v", singularValues)
	}
}
