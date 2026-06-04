package analysis

import (
	"math"
	"sort"
)

// HarmCategory represents a category of harmful or deceptive behavior.
type HarmCategory struct {
	Name       string
	Activations [][]float64
}

// ConeResult holds the geometric analysis of a concept cone.
type ConeResult struct {
	CategoryPairwiseCosines map[string]float64
	IntrinsicDimensionality int
	SingularValues          []float64
	Directions              [][]float64
}

// ConceptConeAnalyzer collects activations across N harm categories and analyzes
// the geometry of the resulting direction subspace.
type ConceptConeAnalyzer struct {
	categories []HarmCategory
	dim        int
}

// NewConceptConeAnalyzer creates an analyzer for the given categories.
func NewConceptConeAnalyzer(categories []HarmCategory) *ConceptConeAnalyzer {
	dim := 0
	if len(categories) > 0 && len(categories[0].Activations) > 0 {
		dim = len(categories[0].Activations[0])
	}
	return &ConceptConeAnalyzer{
		categories: categories,
		dim:        dim,
	}
}

// Analyze computes the full concept cone geometry across all categories.
func (c *ConceptConeAnalyzer) Analyze() *ConeResult {
	if len(c.categories) < 2 || c.dim == 0 {
		return nil
	}

	dirs := c.extractDirections()
	sv := c.computeSingularValues(dirs)
	id := c.estimateIntrinsicDimensionality(sv)
	pairwise := c.computePairwiseCosines(dirs)

	return &ConeResult{
		CategoryPairwiseCosines: pairwise,
		IntrinsicDimensionality: id,
		SingularValues:          sv,
		Directions:              dirs,
	}
}

// PairwiseSimilarities returns cosine similarity matrix between category directions.
func (c *ConceptConeAnalyzer) PairwiseSimilarities() [][]float64 {
	n := len(c.categories)
	if n < 2 {
		return nil
	}
	dirs := c.extractDirections()
	sim := make([][]float64, n)
	for i := 0; i < n; i++ {
		sim[i] = make([]float64, n)
		for j := 0; j < n; j++ {
			sim[i][j] = cosineSimilarity(dirs[i], dirs[j])
		}
	}
	return sim
}

// IntrinsicDimensionality estimates the intrinsic dimensionality of the concept cone
// via PCA scree plot analysis. Returns the number of components needed to explain
// 90% of variance.
func (c *ConceptConeAnalyzer) IntrinsicDimensionality() int {
	sv := c.computeSingularValues(c.extractDirections())
	return c.estimateIntrinsicDimensionality(sv)
}

func (c *ConceptConeAnalyzer) extractDirections() [][]float64 {
	dirs := make([][]float64, len(c.categories))
	for i, cat := range c.categories {
		if len(cat.Activations) == 0 {
			continue
		}
		mean := make([]float64, c.dim)
		for _, act := range cat.Activations {
			for j := 0; j < c.dim; j++ {
				mean[j] += act[j]
			}
		}
		invN := 1.0 / float64(len(cat.Activations))
		for j := 0; j < c.dim; j++ {
			mean[j] *= invN
		}
		dirs[i] = normalize(mean)
	}
	return dirs
}

func (c *ConceptConeAnalyzer) computeSingularValues(dirs [][]float64) []float64 {
	n := len(dirs)
	if n == 0 {
		return nil
	}
	d := c.dim
	centered := make([][]float64, n)
	mean := make([]float64, d)
	count := 0
	for _, dir := range dirs {
		if dir == nil {
			continue
		}
		for j := 0; j < d; j++ {
			mean[j] += dir[j]
		}
		count++
	}
	if count == 0 {
		return nil
	}
	invN := 1.0 / float64(count)
	for j := 0; j < d; j++ {
		mean[j] *= invN
	}
	idx := 0
	for _, dir := range dirs {
		if dir == nil {
			continue
		}
		centered[idx] = make([]float64, d)
		for j := 0; j < d; j++ {
			centered[idx][j] = dir[j] - mean[j]
		}
		idx++
	}
	centered = centered[:count]

	cov := make([][]float64, d)
	for i := 0; i < d; i++ {
		cov[i] = make([]float64, d)
		for k := 0; k < count; k++ {
			for j := 0; j < d; j++ {
				cov[i][j] += centered[k][i] * centered[k][j]
			}
		}
		inv := 1.0 / float64(count-1)
		for j := 0; j < d; j++ {
			cov[i][j] *= inv
		}
	}

	sv := powerIterationSVD(cov, min(d, count))
	return sv
}

func (c *ConceptConeAnalyzer) estimateIntrinsicDimensionality(singularValues []float64) int {
	if len(singularValues) == 0 {
		return 0
	}
	var total float64
	for _, v := range singularValues {
		total += v
	}
	if total == 0 {
		return 0
	}
	var cumulative float64
	for i, v := range singularValues {
		cumulative += v
		if cumulative/total >= 0.9 {
			return i + 1
		}
	}
	return len(singularValues)
}

func (c *ConceptConeAnalyzer) computePairwiseCosines(dirs [][]float64) map[string]float64 {
	result := make(map[string]float64)
	for i := 0; i < len(dirs); i++ {
		for j := i + 1; j < len(dirs); j++ {
			if dirs[i] == nil || dirs[j] == nil {
				continue
			}
			key := c.categories[i].Name + ":" + c.categories[j].Name
			result[key] = cosineSimilarity(dirs[i], dirs[j])
		}
	}
	return result
}

func cosineSimilarity(a, b []float64) float64 {
	var dot, na, nb float64
	for i := 0; i < len(a) && i < len(b); i++ {
		dot += a[i] * b[i]
		na += a[i] * a[i]
		nb += b[i] * b[i]
	}
	if na == 0 || nb == 0 {
		return 0
	}
	return dot / (math.Sqrt(na) * math.Sqrt(nb))
}

func normalize(v []float64) []float64 {
	var norm float64
	for _, x := range v {
		norm += x * x
	}
	norm = math.Sqrt(norm)
	if norm < 1e-10 {
		return v
	}
	out := make([]float64, len(v))
	for i, x := range v {
		out[i] = x / norm
	}
	return out
}

func powerIterationSVD(cov [][]float64, k int) []float64 {
	d := len(cov)
	if d == 0 || k <= 0 {
		return nil
	}
	if k > d {
		k = d
	}

	values := make([]float64, k)
	remainder := make([][]float64, d)
	for i := 0; i < d; i++ {
		remainder[i] = make([]float64, d)
		copy(remainder[i], cov[i])
	}

	for eig := 0; eig < k; eig++ {
		v := make([]float64, d)
		for i := 0; i < d; i++ {
			v[i] = 1.0 / float64(d)
		}

		for iter := 0; iter < 50; iter++ {
			next := make([]float64, d)
			for i := 0; i < d; i++ {
				for j := 0; j < d; j++ {
					next[i] += remainder[i][j] * v[j]
				}
			}
			var norm float64
			for _, x := range next {
				norm += x * x
			}
			norm = math.Sqrt(norm)
			if norm < 1e-15 {
				break
			}
			for i := range next {
				next[i] /= norm
			}
			var diff float64
			for i := range v {
				d := next[i] - v[i]
				diff += d * d
			}
			copy(v, next)
			if math.Sqrt(diff) < 1e-10 {
				break
			}
		}

		var eigenvalue float64
		for i := 0; i < d; i++ {
			var rowSum float64
			for j := 0; j < d; j++ {
				rowSum += remainder[i][j] * v[j]
			}
			eigenvalue += v[i] * rowSum
		}
		values[eig] = math.Sqrt(math.Abs(eigenvalue))

		for i := 0; i < d; i++ {
			for j := 0; j < d; j++ {
				remainder[i][j] -= eigenvalue * v[i] * v[j]
			}
		}
	}

	sort.Slice(values, func(i, j int) bool {
		return values[i] > values[j]
	})
	return values
}

func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}
