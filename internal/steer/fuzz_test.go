package steer

import (
	"context"
	"math"
	"testing"
)

func FuzzActAdd(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 0.5, 1.5)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(-1.0, -2.0, -3.0, -0.5, -1.5)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, alpha float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := ActAdd(activation, direction, alpha)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		for i := range result {
			expected := activation[i] + alpha*direction[i]
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
			_ = expected
		}
	})
}

func FuzzAngular(f *testing.F) {
	f.Add(1.0, 2.0, 3.0, 0.5, 0.1)
	f.Add(0.0, 0.0, 0.0, 1.0, 0.5)
	f.Add(1.0, 0.0, 0.0, 1.0, 0.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, alpha float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := Angular(activation, direction, alpha)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		inputNorm := vecNorm(activation)
		outputNorm := vecNorm(result)

		if inputNorm > 1e-10 && outputNorm > 1e-10 {
			ratio := outputNorm / inputNorm
			if math.Abs(ratio-1.0) > 1e-5 {
				t.Errorf("norm ratio = %f, want ~1.0", ratio)
			}
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}

func FuzzAblate(f *testing.F) {
	f.Add(3.0, 4.0, 5.0, 1.0, 0.0)
	f.Add(0.0, 0.0, 0.0, 0.0, 0.0)
	f.Add(1.0, 1.0, 1.0, 1.0, 1.0)

	f.Fuzz(func(t *testing.T, a1, a2, a3, d1, d2 float64) {
		activation := []float64{a1, a2, a3}
		direction := []float64{d1, d2, 1.0}

		result := Ablate(activation, direction)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		dot := vecDot(result, direction)
		if math.Abs(dot) > 1e-6 {
			t.Errorf("result not orthogonal to direction: dot = %f", dot)
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}

func FuzzDispatcher(f *testing.F) {
	f.Add(0, 1.0, 2.0, 3.0, 0.5, 1.0)
	f.Add(1, 1.0, 0.0, 0.0, 1.0, 0.5)
	f.Add(2, 3.0, 4.0, 5.0, 1.0, 0.0)

	f.Fuzz(func(t *testing.T, layer int, a1, a2, a3, d1, strength float64) {
		layers := []int{0, 1, 2, 3}
		methods := map[int]Method{0: MethodAddition, 1: MethodAngular, 2: MethodAblation}
		d := NewDispatcher(layers, methods)

		activation := []float64{a1, a2, a3}
		direction := []float64{d1, 1.0, 0.0}

		result := d.Apply(layer, activation, direction, strength)

		if len(result) != len(activation) {
			t.Fatalf("len(result) = %d, want %d", len(result), len(activation))
		}

		for i := range result {
			if math.IsInf(result[i], 0) || math.IsNaN(result[i]) {
				t.Errorf("result[%d] = %f (invalid)", i, result[i])
			}
		}
	})
}

func FuzzCrescendoReAnchoring(f *testing.F) {
	f.Add(0.5, 0.5, 1.0, 5)
	f.Add(0.1, 0.9, 0.2, 3)

	f.Fuzz(func(t *testing.T, a0, a1, a2 float64, cadence int) {
		for _, v := range []float64{a0, a1, a2} {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e5 {
				return
			}
		}
		if cadence < 1 || cadence > 10 {
			return
		}
		act := []float64{a0, a1, a2}
		if vecNorm(act) < 1e-6 {
			return
		}
		cfg := DefaultCrescendoConfig()
		cfg.ReAnchorCadence = cadence
		det, err := NewCrescendoDetector([][]float64{{1, 0, 0}, {0, 1, 0}}, cfg)
		if err != nil {
			t.Fatalf("NewCrescendoDetector failed: %v", err)
		}
		res, err := det.StepTurn(context.Background(), 0, act, act, [][]float64{act})
		if err != nil {
			t.Fatalf("StepTurn failed: %v", err)
		}
		if res.SubspaceScore < -1e-9 || res.SubspaceScore > 1.0+1e-9 {
			t.Errorf("SubspaceScore out of [0,1]: %f", res.SubspaceScore)
		}
		if res.IncrementalDriftDeg > cfg.ThetaMaxDeg+1e-6 {
			t.Errorf("IncrementalDriftDeg %f > ThetaMaxDeg %f", res.IncrementalDriftDeg, cfg.ThetaMaxDeg)
		}
	})
}

func FuzzDoRAExtendedRefusal(f *testing.F) {
	f.Add(3.0, 4.0, 1.0, 2.0, 0.8)
	f.Add(1.5, -2.5, 0.5, 1.5, 0.5)

	f.Fuzz(func(t *testing.T, w00, w01, w10, w11, strength float64) {
		for _, v := range []float64{w00, w01, w10, w11, strength} {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e4 {
				return
			}
		}
		if strength <= 0.05 || strength > 1.0 {
			return
		}
		cfg := DefaultDoRAConfig()
		cfg.Strength = strength
		cfg.Rank = 1
		ft := NewExtendedRefusalFineTuner(cfg)
		w := [][]float64{{w00, w01}, {w10, w11}}
		adapter, hardened, err := ft.HardenLayer(0, w, [][]float64{{1.0, 0.0}}, []float64{0.0, 1.0})
		if err != nil {
			t.Fatalf("HardenLayer failed: %v", err)
		}
		if len(hardened) != 2 || adapter.BaselineKL > cfg.MaxKLDivergence+1e-6 {
			t.Errorf("unexpected DoRA output: len=%d, BaselineKL=%f", len(hardened), adapter.BaselineKL)
		}
	})
}
