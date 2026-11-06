package engine

import (
	"errors"
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
		t.Errorf("unexpected ExpertActivation(0): %v", eAct)
	}
	eAct1 := eng.ExpertActivation(act, 1)
	if len(eAct1) != 4 || eAct1[0] != 5 || eAct1[3] != 8 {
		t.Errorf("unexpected ExpertActivation(1): %v", eAct1)
	}
	eAct2 := eng.ExpertActivation(act, 2)
	if len(eAct2) != 0 {
		t.Errorf("expected empty slice when offset >= len(act), got %v", eAct2)
	}
	if eng.ExpertActivation(act, -1) != nil || eng.ExpertActivation(act, 3) != nil {
		t.Error("expected nil for invalid expertIdx")
	}

	eng.UpdateRouting(0, []float64{0.5, 0.3, 0.2})
	eng.UpdateRouting(0, []float64{0.5}) // mismatch length, ignored

	// Trigger panic
	_ = NewMoEEngine(3, 2, []int{4, 4}, nil)
}

func TestMoEEngine_ExpertActivationOffsets_Accuracy(t *testing.T) {
	// Variable expert dimensions: [2, 3, 1, 4], total = 10
	eng := NewMoEEngine(4, 2, []int{2, 3, 1, 4}, nil)
	act := []float64{10, 11, 20, 21, 22, 30, 40, 41, 42, 43}

	want := [][]float64{
		{10, 11},
		{20, 21, 22},
		{30},
		{40, 41, 42, 43},
	}
	for idx, expected := range want {
		got := eng.ExpertActivation(act, idx)
		if len(got) != len(expected) {
			t.Fatalf("expert %d len = %d, want %d", idx, len(got), len(expected))
		}
		for j := range expected {
			if got[j] != expected[j] {
				t.Errorf("expert %d[%d] = %v, want %v", idx, j, got[j], expected[j])
			}
		}
	}

	// Partial/truncated activation for last expert
	shortAct := act[:8]
	got3 := eng.ExpertActivation(shortAct, 3)
	if len(got3) != 2 || got3[0] != 40 || got3[1] != 41 {
		t.Errorf("truncated expert 3 = %v, want [40 41]", got3)
	}
}

func TestMoEEngine_RoutingAccuracyInvariants(t *testing.T) {
	// Uniform fallback when totalSumSq < 1e-10
	eng := NewMoEEngine(4, 6, []int{2, 2, 2, 2}, nil) // topK > numExperts clamped to 4
	indices, weights := eng.DetectRouting(0, []float64{0, 0, 0, 0, 0, 0, 0, 0})
	if len(indices) != 4 || len(weights) != 4 {
		t.Fatalf("expected 4 clamped experts, got %d, %d", len(indices), len(weights))
	}
	var uniformSum float64
	for i := 0; i < 4; i++ {
		if indices[i] != i {
			t.Errorf("uniform fallback index[%d] = %d, want %d", i, indices[i], i)
		}
		if math.Abs(weights[i]-0.25) > 1e-9 {
			t.Errorf("uniform fallback weight[%d] = %f, want 0.25", i, weights[i])
		}
		uniformSum += weights[i]
	}
	if math.Abs(uniformSum-1.0) > 1e-9 {
		t.Errorf("uniform fallback weights sum = %f, want 1.0", uniformSum)
	}

	// Non-zero activations with topK > numExperts and truncated activation slice
	shortNonZero := []float64{3, 4, 0} // expert 0 energy = 25, expert 1 partial = 0
	idxShort, wShort := eng.DetectRouting(0, shortNonZero)
	if len(idxShort) != 4 || idxShort[0] != 0 || math.Abs(wShort[0]-1.0) > 1e-9 {
		t.Errorf("short non-zero routing failed: idx=%v w=%v", idxShort, wShort)
	}

	// Non-zero activations: expert 2 has energy 100, expert 1 has energy 25, others 0
	act := []float64{0, 0, 3, 4, 6, 8, 0, 0}
	engTop2 := NewMoEEngine(4, 2, []int{2, 2, 2, 2}, nil)
	idx2, w2 := engTop2.DetectRouting(0, act)
	if idx2[0] != 2 || idx2[1] != 1 {
		t.Errorf("expected top-2 experts [2, 1], got %v", idx2)
	}
	if math.Abs(w2[0]-100.0/125.0) > 1e-9 || math.Abs(w2[1]-25.0/125.0) > 1e-9 {
		t.Errorf("unexpected weights %v, want [0.8, 0.2]", w2)
	}
	if math.Abs((w2[0]+w2[1])-1.0) > 1e-9 {
		t.Errorf("top-2 weights sum = %f, want 1.0", w2[0]+w2[1])
	}

	// DetectRoutingFromLogits with topK > numExperts
	idxClamped, wClamped := eng.DetectRoutingFromLogits([]float64{1.0, 4.0, 2.0, 3.0})
	if len(idxClamped) != 4 || idxClamped[0] != 1 || idxClamped[1] != 3 || idxClamped[2] != 2 || idxClamped[3] != 0 {
		t.Errorf("expected ordering [1, 3, 2, 0], got %v", idxClamped)
	}
	var logitSum float64
	for _, w := range wClamped {
		logitSum += w
	}
	if math.Abs(logitSum-1.0) > 1e-9 {
		t.Errorf("softmax weights sum = %f, want 1.0", logitSum)
	}

	// Logits with NaN fallback to uniform 1/topK
	idxNaN, wNaN := engTop2.DetectRoutingFromLogits([]float64{math.NaN(), math.NaN(), math.NaN(), math.NaN()})
	if len(idxNaN) != 2 || math.Abs(wNaN[0]-0.5) > 1e-9 || math.Abs(wNaN[1]-0.5) > 1e-9 {
		t.Errorf("NaN logits fallback failed: idx=%v w=%v", idxNaN, wNaN)
	}

	// PollLedger error fallback in DetectRouting
	engErrLedger := NewMoEEngine(2, 1, []int{2, 2}, &mockLedger{err: errors.New("ledger error")})
	_, errErr := engErrLedger.PollLedger(0)
	if errErr == nil {
		t.Error("expected error from mockLedger with err")
	}
	idxFallback, wFallback := engErrLedger.DetectRouting(0, nil)
	if len(idxFallback) != 1 || math.Abs(wFallback[0]-0.5) > 1e-9 {
		t.Errorf("expected uniform fallback on ledger error, got idx=%v w=%v", idxFallback, wFallback)
	}
}
