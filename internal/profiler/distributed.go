package profiler

import (
	"sync"
)

// DistributedProfiler coordinates profiling across multiple nodes.
type DistributedProfiler struct {
	mu        sync.RWMutex
	nodeCount int
	results   map[string]map[int][]float64
}

// NewDistributedProfiler creates a distributed profiler.
func NewDistributedProfiler(nodeCount int) *DistributedProfiler {
	return &DistributedProfiler{
		nodeCount: nodeCount,
		results:   make(map[string]map[int][]float64),
	}
}

// SplitBatches divides activations across nodes.
func (p *DistributedProfiler) SplitBatches(activations [][]float64) map[int][][]float64 {
	splits := make(map[int][][]float64)
	for i := 0; i < p.nodeCount; i++ {
		splits[i] = nil
	}

	for i, act := range activations {
		nodeID := i % p.nodeCount
		splits[nodeID] = append(splits[nodeID], act)
	}

	return splits
}

// CollectResults stores results from a node.
func (p *DistributedProfiler) CollectResults(nodeID string, directions map[int][]float64) {
	p.mu.Lock()
	defer p.mu.Unlock()
	p.results[nodeID] = directions
}

// AggregateResults merges results from all nodes.
func (p *DistributedProfiler) AggregateResults() map[int][]float64 {
	p.mu.RLock()
	defer p.mu.RUnlock()

	aggregated := make(map[int][]float64)
	layerCounts := make(map[int]int)
	layerSums := make(map[int][]float64)

	for _, dirs := range p.results {
		for layerIdx, dir := range dirs {
			layerCounts[layerIdx]++
			if existing, ok := layerSums[layerIdx]; ok {
				for i := range dir {
					existing[i] += dir[i]
				}
			} else {
				sum := make([]float64, len(dir))
				copy(sum, dir)
				layerSums[layerIdx] = sum
			}
		}
	}

	for layerIdx, count := range layerCounts {
		sum := layerSums[layerIdx]
		avg := make([]float64, len(sum))
		for i := range sum {
			avg[i] = sum[i] / float64(count)
		}
		aggregated[layerIdx] = normalize(avg)
	}

	return aggregated
}

// NodeCount returns the number of nodes.
func (p *DistributedProfiler) NodeCount() int {
	return p.nodeCount
}

// ResultCount returns the number of nodes that have submitted results.
func (p *DistributedProfiler) ResultCount() int {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.results)
}

// IsComplete returns true if all nodes have submitted results.
func (p *DistributedProfiler) IsComplete() bool {
	p.mu.RLock()
	defer p.mu.RUnlock()
	return len(p.results) == p.nodeCount
}
