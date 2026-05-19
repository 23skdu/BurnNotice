package profiler

import (
	"testing"
)

func TestLayerGrouping(t *testing.T) {
	p := NewProgressiveProfiler(32, 8, 0.01)

	if len(p.layerGroups) != 4 {
		t.Fatalf("len(layerGroups) = %d, want 4", len(p.layerGroups))
	}

	if len(p.layerGroups[0]) != 8 {
		t.Errorf("first group size = %d, want 8", len(p.layerGroups[0]))
	}
}

func TestLayerGroupingNonDivisible(t *testing.T) {
	p := NewProgressiveProfiler(30, 8, 0.01)

	if len(p.layerGroups) != 4 {
		t.Fatalf("len(layerGroups) = %d, want 4", len(p.layerGroups))
	}

	if len(p.layerGroups[3]) != 6 {
		t.Errorf("last group size = %d, want 6", len(p.layerGroups[3]))
	}
}

func TestValidateLayerGroups(t *testing.T) {
	p := NewProgressiveProfiler(32, 8, 0.01)

	if err := p.ValidateLayerGroups(32); err != nil {
		t.Fatalf("ValidateLayerGroups error: %v", err)
	}
}

func TestMergeDirections(t *testing.T) {
	p := NewProgressiveProfiler(32, 8, 0.01)

	dirs1 := map[int][]float64{
		0: {0.1, 0.2, 0.3},
		1: {0.4, 0.5, 0.6},
	}
	dirs2 := map[int][]float64{
		0: {0.2, 0.3, 0.4},
		2: {0.7, 0.8, 0.9},
	}

	merged := p.MergeDirections([]map[int][]float64{dirs1, dirs2})

	if len(merged) != 3 {
		t.Fatalf("len(merged) = %d, want 3", len(merged))
	}

	if _, ok := merged[0]; !ok {
		t.Fatal("merged should contain layer 0")
	}
}

func TestProfileGroup(t *testing.T) {
	p := NewProgressiveProfiler(4, 2, 0.01)

	pos := [][][]float64{
		{{1.0, 0.0}, {0.9, 0.1}},
	}
	neg := [][][]float64{
		{{0.0, 1.0}, {0.1, 0.9}},
	}

	dirs, shouldContinue := p.ProfileGroup([]int{0, 1}, pos, neg)

	if !shouldContinue {
		t.Fatal("shouldContinue should be true")
	}

	if len(dirs) == 0 {
		t.Fatal("expected directions")
	}
}

func TestDistributedSplitBatches(t *testing.T) {
	p := NewDistributedProfiler(3)

	activations := make([][]float64, 9)
	for i := range activations {
		activations[i] = []float64{float64(i)}
	}

	splits := p.SplitBatches(activations)

	if len(splits) != 3 {
		t.Fatalf("len(splits) = %d, want 3", len(splits))
	}

	for nodeID := 0; nodeID < 3; nodeID++ {
		if len(splits[nodeID]) != 3 {
			t.Errorf("node %d batch size = %d, want 3", nodeID, len(splits[nodeID]))
		}
	}
}

func TestDistributedAggregate(t *testing.T) {
	p := NewDistributedProfiler(2)

	p.CollectResults("node1", map[int][]float64{
		0: {0.5, 0.5},
	})
	p.CollectResults("node2", map[int][]float64{
		0: {0.5, 0.5},
	})

	aggregated := p.AggregateResults()

	if len(aggregated) != 1 {
		t.Fatalf("len(aggregated) = %d, want 1", len(aggregated))
	}
}

func TestDistributedIsComplete(t *testing.T) {
	p := NewDistributedProfiler(2)

	if p.IsComplete() {
		t.Fatal("should not be complete with no results")
	}

	p.CollectResults("node1", nil)
	if p.IsComplete() {
		t.Fatal("should not be complete with 1 of 2 results")
	}

	p.CollectResults("node2", nil)
	if !p.IsComplete() {
		t.Fatal("should be complete with 2 of 2 results")
	}
}
