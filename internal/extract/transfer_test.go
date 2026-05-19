package extract

import (
	"math"
	"testing"
)

func TestNewDirectionTransfer(t *testing.T) {
	dt := NewDirectionTransfer(4096, 8192)

	if dt.sourceDim != 4096 {
		t.Errorf("sourceDim = %d, want 4096", dt.sourceDim)
	}
	if dt.targetDim != 8192 {
		t.Errorf("targetDim = %d, want 8192", dt.targetDim)
	}
}

func TestDirectionTransfer_TransferDirection_SameDim(t *testing.T) {
	tr := NewDirectionTransfer(4, 4)

	source := []float64{1, 2, 3, 4}
	result, err := tr.TransferDirection(source)
	if err != nil {
		t.Fatalf("TransferDirection error: %v", err)
	}

	for i := range source {
		if result[i] != source[i] {
			t.Errorf("result[%d] = %f, want %f", i, result[i], source[i])
		}
	}
}

func TestDirectionTransfer_TransferDirection_Pad(t *testing.T) {
	tr := NewDirectionTransfer(2, 4)

	source := []float64{1, 2}
	result, err := tr.TransferDirection(source)
	if err != nil {
		t.Fatalf("TransferDirection error: %v", err)
	}

	if len(result) != 4 {
		t.Fatalf("len(result) = %d, want 4", len(result))
	}
	if result[0] != 1 || result[1] != 2 {
		t.Errorf("result[:2] = %v, want [1 2]", result[:2])
	}
	if result[2] != 0 || result[3] != 0 {
		t.Errorf("padded values = %v, want [0 0]", result[2:])
	}
}

func TestDirectionTransfer_TransferDirection_Truncate(t *testing.T) {
	tr := NewDirectionTransfer(4, 2)

	source := []float64{1, 2, 3, 4}
	result, err := tr.TransferDirection(source)
	if err != nil {
		t.Fatalf("TransferDirection error: %v", err)
	}

	if len(result) != 2 {
		t.Fatalf("len(result) = %d, want 2", len(result))
	}
	if result[0] != 1 || result[1] != 2 {
		t.Errorf("result = %v, want [1 2]", result)
	}
}

func TestDirectionTransfer_TransferDirection_WrongDim(t *testing.T) {
	tr := NewDirectionTransfer(4, 4)

	_, err := tr.TransferDirection([]float64{1, 2})
	if err == nil {
		t.Fatal("expected error for wrong source dimension")
	}
}

func TestCKASimilarity_Identical(t *testing.T) {
	X := [][]float64{
		{1, 0},
		{0, 1},
		{1, 1},
	}

	similarity, err := CKASimilarity(X, X)
	if err != nil {
		t.Fatalf("CKASimilarity error: %v", err)
	}

	if math.Abs(similarity-1.0) > 1e-6 {
		t.Errorf("CKA(X, X) = %f, want 1.0", similarity)
	}
}

func TestCKASimilarity_Different(t *testing.T) {
	X := [][]float64{
		{1, 0},
		{0, 1},
	}
	Y := [][]float64{
		{0, 1},
		{1, 0},
	}

	similarity, err := CKASimilarity(X, Y)
	if err != nil {
		t.Fatalf("CKASimilarity error: %v", err)
	}

	if similarity < 0 || similarity > 1 {
		t.Errorf("CKA(X, Y) = %f, want [0, 1]", similarity)
	}
}

func TestCKASimilarity_RowMismatch(t *testing.T) {
	X := [][]float64{{1, 0}}
	Y := [][]float64{{1, 0}, {0, 1}}

	_, err := CKASimilarity(X, Y)
	if err == nil {
		t.Fatal("expected error for row count mismatch")
	}
}

func TestCKASimilarity_Empty(t *testing.T) {
	_, err := CKASimilarity([][]float64{}, [][]float64{})
	if err == nil {
		t.Fatal("expected error for empty matrices")
	}
}
