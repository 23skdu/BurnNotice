package steer

import (
	"math"
	"sync"
	"testing"

	"gonum.org/v1/gonum/floats"
)

func TestKVCacheAccumulationTracker_DriftDetection(t *testing.T) {
	tracker := NewKVCacheAccumulationTracker(15.0)

	// Turn 0: baseline vector along x-axis
	v0 := []float64{1.0, 0.0, 0.0}
	drift, alert, err := tracker.RecordTurn(0, v0)
	if err != nil {
		t.Fatalf("unexpected error on turn 0: %v", err)
	}
	if drift != 0.0 || alert {
		t.Errorf("turn 0 should have 0 drift and no alert, got drift=%f, alert=%v", drift, alert)
	}

	// Turn 1: vector rotated by 10 degrees (< 15 degrees)
	// cos(10 deg) ≈ 0.9848, sin(10 deg) ≈ 0.1736
	rad10 := 10.0 * math.Pi / 180.0
	v1 := []float64{math.Cos(rad10), math.Sin(rad10), 0.0}
	drift, alert, err = tracker.RecordTurn(1, v1)
	if err != nil {
		t.Fatalf("unexpected error on turn 1: %v", err)
	}
	if math.Abs(drift-10.0) > 0.1 || alert {
		t.Errorf("turn 1 should have ~10 deg drift and no alert, got drift=%f, alert=%v", drift, alert)
	}

	// Turn 2: vector rotated by 20 degrees (> 15 degrees)
	rad20 := 20.0 * math.Pi / 180.0
	v2 := []float64{math.Cos(rad20), math.Sin(rad20), 0.0}
	drift, alert, err = tracker.RecordTurn(2, v2)
	if err != nil {
		t.Fatalf("unexpected error on turn 2: %v", err)
	}
	if math.Abs(drift-20.0) > 0.1 || !alert {
		t.Errorf("turn 2 should have ~20 deg drift and ALERT=true, got drift=%f, alert=%v", drift, alert)
	}

	if tracker.AlarmCount() != 1 {
		t.Errorf("expected alarm count 1, got %d", tracker.AlarmCount())
	}

	history := tracker.History()
	if len(history) != 3 {
		t.Errorf("expected 3 history records, got %d", len(history))
	}
}

func TestKVCacheAccumulationTracker_Concurrent(t *testing.T) {
	tracker := NewKVCacheAccumulationTracker(15.0)
	v0 := []float64{1.0, 0.0}
	if _, _, err := tracker.RecordTurn(0, v0); err != nil {
		t.Fatalf("unexpected error on turn 0: %v", err)
	}

	var wg sync.WaitGroup
	for i := 1; i <= 20; i++ {
		wg.Add(1)
		go func(idx int) {
			defer wg.Done()
			ang := float64(idx) * math.Pi / 180.0
			vec := []float64{math.Cos(ang), math.Sin(ang)}
			_, _, _ = tracker.RecordTurn(idx, vec)
		}(i)
	}
	wg.Wait()

	if len(tracker.History()) != 21 {
		t.Errorf("expected 21 records, got %d", len(tracker.History()))
	}
}

func TestGCADApply(t *testing.T) {
	unsteered := []float64{10.0, 10.0}
	cfg := GCADConfig{
		MaxDeltaNorm: 2.0,
		GateDecay:    0.9,
	}

	// Small delta (within limit): delta = (1.0, 1.0), norm = sqrt(2) ≈ 1.414 < 2.0
	steeredSmall := []float64{11.0, 11.0}
	res, wasCropped := GCADApply(steeredSmall, unsteered, 0, cfg)
	if wasCropped {
		t.Errorf("small delta should not be cropped")
	}
	if math.Abs(res[0]-11.0) > 1e-6 || math.Abs(res[1]-11.0) > 1e-6 {
		t.Errorf("expected unmodified steered, got %v", res)
	}

	// Large delta (exceeds limit): delta = (3.0, 4.0), norm = 5.0 > 2.0
	steeredLarge := []float64{13.0, 14.0}
	resLarge, wasCropped := GCADApply(steeredLarge, unsteered, 0, cfg)
	if !wasCropped {
		t.Errorf("large delta should be cropped")
	}

	// Delta after cropping should have norm exactly 2.0
	croppedDelta := []float64{resLarge[0] - unsteered[0], resLarge[1] - unsteered[1]}
	norm := floats.Norm(croppedDelta, 2)
	if math.Abs(norm-2.0) > 1e-6 {
		t.Errorf("expected cropped delta norm 2.0, got %f", norm)
	}

	// Historical age decay: turnDelta = 3
	resDecayed, wasCropped := GCADApply(steeredSmall, unsteered, 3, cfg)
	if !wasCropped {
		t.Errorf("age decay should mark cropped/attenuated")
	}
	// Delta should be scaled by 0.9^3 = 0.729
	decayedDelta := resDecayed[0] - unsteered[0]
	expectedDelta := 1.0 * 0.9 * 0.9 * 0.9
	if math.Abs(decayedDelta-expectedDelta) > 1e-6 {
		t.Errorf("expected decayed delta %f, got %f", expectedDelta, decayedDelta)
	}
}
