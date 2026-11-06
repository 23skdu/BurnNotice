package profiler

import (
	"math"
	"testing"
)

func FuzzProgressiveProfiler(f *testing.F) {
	f.Add(16, 4, 0.05, 1.0, 0.2, -0.5, 0.8)
	f.Add(7, 3, 0.01, 0.5, 0.5, -0.5, -0.5)
	f.Add(1, 0, -1.0, 0.0, 1.0, 1.0, 0.0)

	f.Fuzz(func(t *testing.T, totalLayers, groupSize int, thresh, p0, p1, n0, n1 float64) {
		if totalLayers <= 0 || totalLayers > 64 || groupSize > 64 {
			return
		}
		for _, v := range []float64{thresh, p0, p1, n0, n1} {
			if math.IsNaN(v) || math.IsInf(v, 0) || math.Abs(v) > 1e6 {
				return
			}
		}

		pp := NewProgressiveProfiler(totalLayers, groupSize, thresh)
		if err := pp.ValidateLayerGroups(totalLayers); err != nil {
			t.Fatalf("ValidateLayerGroups(%d) failed: %v", totalLayers, err)
		}
		if pp.TotalLayers() != totalLayers {
			t.Fatalf("TotalLayers = %d, want %d", pp.TotalLayers(), totalLayers)
		}

		groups := pp.LayerGroups()
		if len(groups) > 0 {
			pos := [][][]float64{{{p0, p1}}}
			neg := [][][]float64{{{n0, n1}}}
			dirs, _ := pp.ProfileGroup(groups[0], pos, neg)
			_ = pp.MergeDirections([]map[int][]float64{dirs})
		}
	})
}

func FuzzDistributedProfiler(f *testing.F) {
	f.Add(3, 10, 0.5, -0.5)
	f.Add(1, 4, 1.0, 2.0)

	f.Fuzz(func(t *testing.T, nodeCount, batchSize int, v0, v1 float64) {
		if nodeCount <= 0 || nodeCount > 16 || batchSize < 0 || batchSize > 64 {
			return
		}
		if math.IsNaN(v0) || math.IsInf(v0, 0) || math.IsNaN(v1) || math.IsInf(v1, 0) {
			return
		}

		dp := NewDistributedProfiler(nodeCount)
		acts := make([][]float64, batchSize)
		for i := range acts {
			acts[i] = []float64{v0, v1}
		}

		splits := dp.SplitBatches(acts)
		total := 0
		for _, s := range splits {
			total += len(s)
		}
		if total != batchSize {
			t.Fatalf("SplitBatches total = %d, want %d", total, batchSize)
		}

		dp.CollectResults("node-0", map[int][]float64{0: {v0, v1}})
		_ = dp.AggregateResults()
	})
}
