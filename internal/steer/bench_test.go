package steer

import (
	"math/rand"
	"testing"
)

func BenchmarkActAdd(b *testing.B) {
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		ActAdd(activation, direction, 0.5)
	}
}

func BenchmarkAngular(b *testing.B) {
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		Angular(activation, direction, 0.5)
	}
}

func BenchmarkAblate(b *testing.B) {
	activation := randomVecF64(4096)
	direction := randomVecF64(4096)

	b.ResetTimer()
	for b.Loop() {
		Ablate(activation, direction)
	}
}

func BenchmarkDispatcher_Apply(b *testing.B) {
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
	l := NewLoRASteerer(4096, 16)
	l.ComputeLoRAWeights(randomVecF64(4096), 0.5)

	weights := make([][]float64, 4096)
	for i := range weights {
		weights[i] = randomVecF64(4096)
	}

	b.ResetTimer()
	for b.Loop() {
		l.ApplyLoRA(weights)
		l.ReverseLoRA(weights)
	}
}

func randomVecF64(n int) []float64 {
	v := make([]float64, n)
	for i := range v {
		v[i] = rand.Float64()*2 - 1
	}
	return v
}
