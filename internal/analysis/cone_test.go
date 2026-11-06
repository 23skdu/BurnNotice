package analysis

import (
	"fmt"
	"math"
	"sync"
	"testing"
)

func TestConceptConeAnalyzer_Basic(t *testing.T) {
	// 1. Create mock activations for 3 harm categories in 4-dimensional space
	cat1 := HarmCategory{
		Name: "violence",
		Activations: [][]float64{
			{1.0, 0.0, 0.0, 0.0},
			{1.1, 0.1, 0.0, 0.0},
		},
	}
	cat2 := HarmCategory{
		Name: "deception",
		Activations: [][]float64{
			{0.0, 1.0, 0.0, 0.0},
			{0.1, 1.1, 0.0, 0.0},
		},
	}
	cat3 := HarmCategory{
		Name: "CSAM",
		Activations: [][]float64{
			{0.0, 0.0, 1.0, 0.0},
			{0.0, 0.1, 1.1, 0.0},
		},
	}

	analyzer := NewConceptConeAnalyzer([]HarmCategory{cat1, cat2, cat3})

	result := analyzer.Analyze()
	if result == nil {
		t.Fatal("Analyze returned nil")
	}

	// Verify singular values exist
	if len(result.SingularValues) == 0 {
		t.Error("expected non-empty singular values")
	}

	// Intrinsic dimensionality should be estimated
	id := analyzer.IntrinsicDimensionality()
	if id <= 0 {
		t.Errorf("IntrinsicDimensionality = %d, want > 0", id)
	}

	// Pairwise similarities
	sims := analyzer.PairwiseSimilarities()
	if len(sims) != 3 {
		t.Errorf("len(sims) = %d, want 3", len(sims))
	}
	for i := 0; i < 3; i++ {
		if math.Abs(sims[i][i]-1.0) > 1e-5 {
			t.Errorf("self-similarity sims[%d][%d] = %f, want 1.0", i, i, sims[i][i])
		}
	}
}

func TestConceptConeAnalyzer_EdgeCases(t *testing.T) {
	// Empty categories
	a1 := NewConceptConeAnalyzer(nil)
	if a1.Analyze() != nil {
		t.Error("expected nil Analyze for nil categories")
	}
	if a1.PairwiseSimilarities() != nil {
		t.Error("expected nil PairwiseSimilarities for nil categories")
	}
	if a1.IntrinsicDimensionality() != 0 {
		t.Error("expected 0 IntrinsicDimensionality for nil categories")
	}

	// Single category
	cat := HarmCategory{
		Name: "only",
		Activations: [][]float64{
			{1.0, 0.0},
		},
	}
	a2 := NewConceptConeAnalyzer([]HarmCategory{cat})
	if a2.Analyze() != nil {
		t.Error("expected nil Analyze for single category")
	}
	if a2.PairwiseSimilarities() != nil {
		t.Error("expected nil PairwiseSimilarities for single category")
	}

	// Empty activations in categories
	catEmpty1 := HarmCategory{Name: "empty1", Activations: [][]float64{}}
	catEmpty2 := HarmCategory{Name: "empty2", Activations: [][]float64{}}
	a3 := NewConceptConeAnalyzer([]HarmCategory{catEmpty1, catEmpty2})
	res := a3.Analyze()
	// Should handle empty activations gracefully
	_ = res
}

func makeSyntheticHarmCategories(numCats, numSamples, dim int) []HarmCategory {
	cats := make([]HarmCategory, numCats)
	for c := 0; c < numCats; c++ {
		acts := make([][]float64, numSamples)
		for s := 0; s < numSamples; s++ {
			vec := make([]float64, dim)
			for d := 0; d < dim; d++ {
				if d == c%dim {
					vec[d] = 1.0 + 0.05*float64(s)
				} else {
					vec[d] = 0.01 * float64((c+s+d)%7)
				}
			}
			acts[s] = vec
		}
		cats[c] = HarmCategory{
			Name:        fmt.Sprintf("cat_%d", c),
			Activations: acts,
		}
	}
	return cats
}

func TestConceptConeAnalyzer_ConcurrentRaceSafety(t *testing.T) {
	cats := makeSyntheticHarmCategories(6, 4, 16)
	analyzer := NewConceptConeAnalyzer(cats)

	var wg sync.WaitGroup
	for g := 0; g < 8; g++ {
		wg.Add(1)
		go func() {
			defer wg.Done()
			res := analyzer.Analyze()
			if res == nil {
				t.Errorf("Analyze returned nil")
				return
			}
			sims := analyzer.PairwiseSimilarities()
			if len(sims) != 6 {
				t.Errorf("PairwiseSimilarities len = %d, want 6", len(sims))
			}
			id := analyzer.IntrinsicDimensionality()
			if id <= 0 {
				t.Errorf("IntrinsicDimensionality = %d, want > 0", id)
			}
		}()
	}
	wg.Wait()
}

func BenchmarkConceptConeAnalyze(b *testing.B) {
	cats := makeSyntheticHarmCategories(16, 16, 64)
	analyzer := NewConceptConeAnalyzer(cats)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if res := analyzer.Analyze(); res == nil {
			b.Fatal("nil result")
		}
	}
}

func BenchmarkConceptConePairwiseSimilarities(b *testing.B) {
	cats := makeSyntheticHarmCategories(24, 8, 128)
	analyzer := NewConceptConeAnalyzer(cats)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if sims := analyzer.PairwiseSimilarities(); len(sims) != 24 {
			b.Fatal("unexpected sims length")
		}
	}
}
