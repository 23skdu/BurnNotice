package analysis

// OuroborosDetector measures per-layer deception scores pre/post intervention,
// identifies compensating layers where scores increased, and supports iterative
// refinement targeting only those layers.
type OuroborosDetector struct {
	preScores  map[int]float64
	postScores map[int]float64
}

// NewOuroborosDetector creates a detector for ouroboros-style refinement.
func NewOuroborosDetector() *OuroborosDetector {
	return &OuroborosDetector{
		preScores:  make(map[int]float64),
		postScores: make(map[int]float64),
	}
}

// RecordPre records a per-layer deception score before steering intervention.
func (d *OuroborosDetector) RecordPre(layerIdx int, score float64) {
	d.preScores[layerIdx] = score
}

// RecordPost records a per-layer deception score after steering intervention.
func (d *OuroborosDetector) RecordPost(layerIdx int, score float64) {
	d.postScores[layerIdx] = score
}

// CompensatingLayers returns layer indices where the post-intervention score
// is higher than the pre-intervention score (compensation detected).
func (d *OuroborosDetector) CompensatingLayers() []int {
	var layers []int
	for idx := range d.preScores {
		pre := d.preScores[idx]
		post := d.postScores[idx]
		if post > pre {
			layers = append(layers, idx)
		}
	}
	return layers
}

// NetReduction computes the average reduction across all layers.
// Returns < 0 if net compensation occurred.
func (d *OuroborosDetector) NetReduction() float64 {
	var totalPre, totalPost float64
	var count int
	for idx := range d.preScores {
		totalPre += d.preScores[idx]
		totalPost += d.postScores[idx]
		count++
	}
	if count == 0 || totalPre == 0 {
		return 0
	}
	return (totalPre - totalPost) / totalPre
}

// RequiresRefinement returns true if net reduction is below the threshold.
func (d *OuroborosDetector) RequiresRefinement(threshold float64) bool {
	return d.NetReduction() < threshold
}

// RefinementPasses returns the number of additional passes needed based on
// the number of compensating layers and the required reduction threshold.
func (d *OuroborosDetector) RefinementPasses(threshold float64, maxPasses int) int {
	if !d.RequiresRefinement(threshold) {
		return 0
	}
	compensating := len(d.CompensatingLayers())
	if compensating == 0 {
		return 0
	}
	passes := (compensating + 2) / 3
	if passes > maxPasses {
		passes = maxPasses
	}
	if passes < 1 {
		passes = 1
	}
	return passes
}

// Reset clears all recorded scores for a fresh detection cycle.
func (d *OuroborosDetector) Reset() {
	d.preScores = make(map[int]float64)
	d.postScores = make(map[int]float64)
}
