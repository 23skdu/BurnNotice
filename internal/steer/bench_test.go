package steer

import (
	"math/rand"
	"testing"
)

func BenchmarkActAdd(b *testing.B) {
	b.ReportAllocs()
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		ActAdd(activation, direction, 0.5)
	}
}

func BenchmarkAngular(b *testing.B) {
	b.ReportAllocs()
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		Angular(activation, direction, 0.5)
	}
}

func BenchmarkAblate(b *testing.B) {
	b.ReportAllocs()
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		Ablate(activation, direction)
	}
}

func BenchmarkNormPreservingAblation(b *testing.B) {
	b.ReportAllocs()
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		NormPreservingAblation(activation, direction, 1.0, 0.01)
	}
}

func BenchmarkBiprojectedAblation(b *testing.B) {
	b.ReportAllocs()
	activation := randomVecF64(4096)
	harmDir := randomVecF64(4096)
	benignDir := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		BiprojectedAblation(activation, harmDir, benignDir, 1.0, 0.01)
	}
}

func BenchmarkGCADApply(b *testing.B) {
	b.ReportAllocs()
	steered := randomVecF64(4096)
	unsteered := randomVecF64(4096)
	cfg := DefaultGCADConfig()

	b.ResetTimer()
	for b.Loop() {
		GCADApply(steered, unsteered, 2, cfg)
	}
}

func BenchmarkGCADApplyInPlace(b *testing.B) {
	b.ReportAllocs()
	steered := randomVecF64(4096)
	unsteered := randomVecF64(4096)
	cfg := DefaultGCADConfig()

	b.ResetTimer()
	for b.Loop() {
		GCADApplyInPlace(steered, unsteered, 2, cfg)
	}
}

func BenchmarkKVCacheAccumulationTracker_RecordTurn(b *testing.B) {
	b.ReportAllocs()
	tracker := NewKVCacheAccumulationTracker(15.0)
	v0 := randomVecF64(4096)
	if _, _, err := tracker.RecordTurn(0, v0); err != nil {
		b.Fatal(err)
	}
	v1 := randomVecF64(4096)

	idx := 1
	b.ResetTimer()
	for b.Loop() {
		_, _, _ = tracker.RecordTurn(idx, v1)
		idx++
	}
}

func BenchmarkDispatcher_Apply(b *testing.B) {
	b.ReportAllocs()
	d := NewDispatcher([]int{0, 1, 2, 3}, map[int]Method{0: MethodAddition, 1: MethodAngular, 2: MethodAblation})

	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	i := 0
	b.ResetTimer()
	for b.Loop() {
		d.Apply(i%4, activation, direction, 0.5)
		i++
	}
}

func BenchmarkSelectiveSteerer_ComputeSteering(b *testing.B) {
	b.ReportAllocs()
	truthDir := map[int][]float64{0: randomVecF64(4096)}
	deceptionDir := map[int][]float64{0: randomVecF64(4096)}
	s := NewSelectiveSteerer(truthDir, deceptionDir, []int{0})

	activation := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		s.ComputeSteering(0, activation, 0.5)
	}
}

func BenchmarkLoRASteerer_ApplyLoRA(b *testing.B) {
	b.ReportAllocs()
	l := NewLoRASteerer(4096, 16)
	if err := l.ComputeLoRAWeights(randomVecF64(4096), 0.5); err != nil {
		b.Fatal(err)
	}

	weights := make([][]float64, 4096)
	for i := range weights {
		weights[i] = randomVecF64(4096)
	}

	b.ResetTimer()
	for b.Loop() {
		if _, err := l.ApplyLoRA(weights); err != nil {
			b.Fatal(err)
		}
		if _, err := l.ReverseLoRA(weights); err != nil {
			b.Fatal(err)
		}
	}
}

func randomVecF64(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64()*2 - 1
	}
	return v
}
