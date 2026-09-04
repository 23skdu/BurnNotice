package engine

import (
	"math"
	"testing"
)

type mockLedger struct {
	activations map[int][]float64
	err         error
}

func (m *mockLedger) PollActivation(layerIdx int) ([]float64, error) {
	if m.err != nil {
		return nil, m.err
	}
	return m.activations[layerIdx], nil
}

func TestMoEEngine_PollLedger(t *testing.T) {
	// 1. Without ledger
	engNoLedger := NewMoEEngine(4, 2, []int{2, 2, 2, 2}, nil)
	_, err := engNoLedger.PollLedger(0)
	if err == nil {
		t.Error("expected error when polling without ledger")
	}

	// 2. With ledger
	mock := &mockLedger{
		activations: map[int][]float64{
			1: {1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0},
		},
	}
	eng := NewMoEEngine(4, 2, []int{2, 2, 2, 2}, mock)
	data, err := eng.PollLedger(1)
	if err != nil {
		t.Fatalf("unexpected error polling ledger: %v", err)
	}
	if len(data) != 8 {
		t.Fatalf("expected 8 floats, got %d", len(data))
	}
}

func TestMoEEngine_DetectRouting_WithLedger(t *testing.T) {
	mock := &mockLedger{
		activations: map[int][]float64{
			2: {10.0, 10.0, 1.0, 1.0, 0.0, 0.0, 0.0, 0.0},
		},
	}
	eng := NewMoEEngine(4, 2, []int{2, 2, 2, 2}, mock)

	// Call DetectRouting with empty slice; should poll ledger at layer 2
	indices, weights := eng.DetectRouting(2, nil)
	if len(indices) != 2 || len(weights) != 2 {
		t.Fatalf("expected 2 experts, got %d indices, %d weights", len(indices), len(weights))
	}
	if indices[0] != 0 {
		t.Errorf("expected expert 0 to have highest weight, got %d", indices[0])
	}
	if weights[0] <= weights[1] {
		t.Errorf("expected weight[0] > weight[1], got %f <= %f", weights[0], weights[1])
	}
}

func TestMoEEngine_DetectRoutingFromLogits(t *testing.T) {
	eng := NewMoEEngine(4, 2, []int{2, 2, 2, 2}, nil)

	// Test dimension mismatch
	resIdx, resW := eng.DetectRoutingFromLogits([]float64{1.0, 2.0})
	if resIdx != nil || resW != nil {
		t.Errorf("expected nil for mismatched logits dimension, got %v, %v", resIdx, resW)
	}

	// 4 experts, top-2: logits [1.0, 3.0, 0.0, 2.0]
	// Top 2 should be expert 1 (logit 3.0) and expert 3 (logit 2.0)
	indices, weights := eng.DetectRoutingFromLogits([]float64{1.0, 3.0, 0.0, 2.0})
	if len(indices) != 2 || len(weights) != 2 {
		t.Fatalf("expected 2 indices and weights, got %d, %d", len(indices), len(weights))
	}
	if indices[0] != 1 || indices[1] != 3 {
		t.Errorf("expected indices [1, 3], got %v", indices)
	}

	// Weights should sum to 1.0
	sum := weights[0] + weights[1]
	if math.Abs(sum-1.0) > 1e-6 {
		t.Errorf("expected weights sum to 1.0, got %f", sum)
	}
	// Exp(3-3) / (Exp(3-3) + Exp(2-3)) = 1 / (1 + 1/e) ~= 1 / 1.367879 ~= 0.731058
	expectedW0 := 1.0 / (1.0 + math.Exp(-1.0))
	if math.Abs(weights[0]-expectedW0) > 1e-4 {
		t.Errorf("expected weight[0] ~ %f, got %f", expectedW0, weights[0])
	}
}

func TestMoEEngine_AccessorsAndPanic(t *testing.T) {
	defer func() {
		if r := recover(); r == nil {
			t.Error("expected panic for mismatched expertDims")
		}
	}()

	eng := NewMoEEngine(3, 2, []int{4, 4, 4}, nil)
	if eng.NumExperts() != 3 {
		t.Errorf("NumExperts = %d, want 3", eng.NumExperts())
	}
	if eng.ExpertDim(1) != 4 {
		t.Errorf("ExpertDim(1) = %d, want 4", eng.ExpertDim(1))
	}
	if eng.ExpertDim(-1) != 0 || eng.ExpertDim(10) != 0 {
		t.Error("expected 0 for out-of-bounds ExpertDim")
	}

	act := []float64{1, 2, 3, 4, 5, 6, 7, 8}
	eAct := eng.ExpertActivation(act, 0)
	if len(eAct) != 4 || eAct[0] != 1 || eAct[3] != 4 {
		t.Errorf("unexpected ExpertActivation: %v", eAct)
	}
	if eng.ExpertActivation(act, -1) != nil {
		t.Error("expected nil for invalid expertIdx")
	}

	eng.UpdateRouting(0, []float64{0.5, 0.3, 0.2})
	eng.UpdateRouting(0, []float64{0.5}) // mismatch length, ignored

	// Trigger panic
	_ = NewMoEEngine(3, 2, []int{4, 4}, nil)
}
