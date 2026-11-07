package sae

import (
	"fmt"
	"math"
	"sort"
)

// FeatureAbsorptionMetric quantifies feature absorption for a single semantic concept probe.
type FeatureAbsorptionMetric struct {
	ConceptName           string  `json:"concept_name"`
	PrimaryFeatureIndex   int     `json:"primary_feature_index"`
	DecoderCosine         float64 `json:"decoder_cosine"`
	EncoderActivationRate float64 `json:"encoder_activation_rate"`
	ChildCaptureRatio     float64 `json:"child_capture_ratio"`
	AbsorptionScore       float64 `json:"absorption_score"`
	AbsorbingChildren     []int   `json:"absorbing_children,omitempty"`
	IsAbsorbed            bool    `json:"is_absorbed"`
}

// FeatureAbsorptionReport summarizes automated feature-absorption, polysemanticity, and coverage
// diagnostics for an SAE dictionary.
type FeatureAbsorptionReport struct {
	Family                 SAEFamily                          `json:"family"`
	FeatureCount           int                                `json:"feature_count"`
	InputDim               int                                `json:"input_dim"`
	SampleCount            int                                `json:"sample_count"`
	ActiveCoverage         float64                            `json:"active_coverage"`
	DeadFeatureRatio       float64                            `json:"dead_feature_ratio"`
	MeanL0Sparsity         float64                            `json:"mean_l0_sparsity"`
	MeanReconstructionR2   float64                            `json:"mean_reconstruction_r2"`
	MeanPolysemanticity    float64                            `json:"mean_polysemanticity"`
	AbsorptionRate         float64                            `json:"absorption_rate"`
	FeaturePolysemanticity []float64                          `json:"feature_polysemanticity"`
	ConceptMetrics         map[string]FeatureAbsorptionMetric `json:"concept_metrics"`
}

// AlignedFeaturePair represents a matched safety/semantic feature pair across two SAE architectures.
type AlignedFeaturePair struct {
	SourceFeature    int     `json:"source_feature"`
	TargetFeature    int     `json:"target_feature"`
	ActivationCosine float64 `json:"activation_cosine"`
	ProcrustesCosine float64 `json:"procrustes_cosine"`
	CombinedScore    float64 `json:"combined_score"`
}

// CrossArchAlignmentResult summarizes feature alignment between two SAE dictionaries
// (e.g., Gemma Scope <-> Llama-Scope <-> Qwen-SAE).
type CrossArchAlignmentResult struct {
	SourceFamily        SAEFamily            `json:"source_family"`
	TargetFamily        SAEFamily            `json:"target_family"`
	LinearCKA           float64              `json:"linear_cka"`
	MeanAlignmentCosine float64              `json:"mean_alignment_cosine"`
	AlignedPairs        []AlignedFeaturePair `json:"aligned_pairs"`
}

// AnalyzeFeatureAbsorption runs automated feature-absorption, polysemanticity, and dictionary
// coverage diagnostics on an SAEDictionary over a corpus of activations and concept directions.
func AnalyzeFeatureAbsorption(
	dict SAEDictionary,
	activations [][]float64,
	conceptDirections map[string][]float64,
) (*FeatureAbsorptionReport, error) {
	if dict == nil || dict.Weights() == nil {
		return nil, fmt.Errorf("SAE dictionary weights not loaded")
	}
	w := dict.Weights()
	if len(activations) == 0 {
		return nil, fmt.Errorf("activations corpus cannot be empty")
	}

	nSamples := len(activations)
	featCount := w.FeatureCount
	inDim := w.InputDim

	encodedCorpus := make([][]float64, nSamples)
	activeCounts := make([]int, featCount)
	var totalL0 float64

	// Compute mean activation for R^2 denominator
	meanAct := make([]float64, inDim)
	for s, act := range activations {
		if len(act) != inDim {
			return nil, fmt.Errorf("sample %d dim %d != input dim %d", s, len(act), inDim)
		}
		for j, v := range act {
			meanAct[j] += v / float64(nSamples)
		}
	}

	var totalResSq, totalVarSq float64
	for s, act := range activations {
		feats, err := dict.Encode(act)
		if err != nil {
			return nil, fmt.Errorf("encode sample %d: %w", s, err)
		}
		encodedCorpus[s] = feats

		for fIdx, val := range feats {
			if val > 0 {
				activeCounts[fIdx]++
				totalL0++
			}
		}

		recon, err := dict.Decode(feats)
		if err != nil {
			return nil, fmt.Errorf("decode sample %d: %w", s, err)
		}
		for j := 0; j < inDim; j++ {
			diff := act[j] - recon[j]
			totalResSq += diff * diff
			vDiff := act[j] - meanAct[j]
			totalVarSq += vDiff * vDiff
		}
	}

	activeFeats := 0
	for _, c := range activeCounts {
		if c > 0 {
			activeFeats++
		}
	}
	coverage := float64(activeFeats) / float64(featCount)
	deadRatio := 1.0 - coverage
	meanL0 := totalL0 / float64(nSamples)

	r2 := 1.0
	if totalVarSq > 1e-12 {
		r2 = 1.0 - (totalResSq / totalVarSq)
		if r2 < 0 {
			r2 = 0
		}
	}

	// Sort concept names for deterministic evaluation
	conceptNames := make([]string, 0, len(conceptDirections))
	for name := range conceptDirections {
		conceptNames = append(conceptNames, name)
	}
	sort.Strings(conceptNames)

	conceptMetrics := make(map[string]FeatureAbsorptionMetric, len(conceptNames))
	conceptFeatureMass := make([][]float64, featCount)
	for f := 0; f < featCount; f++ {
		conceptFeatureMass[f] = make([]float64, len(conceptNames))
	}

	absorbedConcepts := 0
	for cIdx, name := range conceptNames {
		dir := conceptDirections[name]
		if len(dir) != inDim {
			return nil, fmt.Errorf("concept %q dim %d != input dim %d", name, len(dir), inDim)
		}
		uDir := normalizeSlice(dir)

		// Find primary feature with maximum decoder cosine alignment to uDir
		bestFeat := 0
		bestCos := -2.0
		decoderCosines := make([]float64, featCount)
		for f := 0; f < featCount; f++ {
			cos := 0.0
			if f < len(w.DecoderWeights) && len(w.DecoderWeights[f]) == inDim {
				cos = sliceCosine(w.DecoderWeights[f], uDir)
			}
			decoderCosines[f] = cos
			if cos > bestCos {
				bestCos = cos
				bestFeat = f
			}
		}
		if bestCos < 0 {
			bestCos = 0
		}

		// Evaluate activation rate and child feature absorption on concept-positive samples
		var posSamples, primaryFires, childCaptured int
		childFreq := make(map[int]int)

		for s, act := range activations {
			proj := sliceDot(act, uDir)
			if proj <= 1e-6 {
				continue
			}
			posSamples++
			feats := encodedCorpus[s]

			for f := 0; f < featCount; f++ {
				if feats[f] > 0 {
					conceptFeatureMass[f][cIdx] += feats[f] * proj
				}
			}

			if feats[bestFeat] > 0 {
				primaryFires++
			} else {
				// Check if secondary aligned child features absorbed the projection
				capturedThisSample := false
				for f := 0; f < featCount; f++ {
					if f != bestFeat && feats[f] > 0 && decoderCosines[f] >= 0.20 {
						childFreq[f]++
						capturedThisSample = true
					}
				}
				if capturedThisSample {
					childCaptured++
				}
			}
		}

		actRate := 1.0
		captureRatio := 0.0
		if posSamples > 0 {
			actRate = float64(primaryFires) / float64(posSamples)
			missed := posSamples - primaryFires
			if missed > 0 {
				captureRatio = float64(childCaptured) / float64(missed)
			}
		}

		absorptionScore := bestCos * (1.0 - actRate) * captureRatio
		isAbsorbed := absorptionScore >= 0.20

		children := make([]int, 0, len(childFreq))
		for ch := range childFreq {
			children = append(children, ch)
		}
		sort.Ints(children)

		if isAbsorbed {
			absorbedConcepts++
		}

		conceptMetrics[name] = FeatureAbsorptionMetric{
			ConceptName:           name,
			PrimaryFeatureIndex:   bestFeat,
			DecoderCosine:         bestCos,
			EncoderActivationRate: actRate,
			ChildCaptureRatio:     captureRatio,
			AbsorptionScore:       absorptionScore,
			AbsorbingChildren:     children,
			IsAbsorbed:            isAbsorbed,
		}
	}

	// Compute per-feature polysemanticity via normalized Shannon entropy across concepts
	polyScores := make([]float64, featCount)
	var sumPoly float64
	numConcepts := len(conceptNames)
	for f := 0; f < featCount; f++ {
		if numConcepts <= 1 {
			polyScores[f] = 0.0
			continue
		}
		var massSum float64
		for _, m := range conceptFeatureMass[f] {
			massSum += m
		}
		if massSum <= 1e-12 {
			continue
		}
		var ent float64
		for _, m := range conceptFeatureMass[f] {
			if m > 0 {
				p := m / massSum
				ent -= p * math.Log(p)
			}
		}
		poly := ent / math.Log(float64(numConcepts))
		if poly > 1.0 {
			poly = 1.0
		}
		polyScores[f] = poly
		sumPoly += poly
	}

	meanPoly := 0.0
	if featCount > 0 {
		meanPoly = sumPoly / float64(featCount)
	}
	absRate := 0.0
	if numConcepts > 0 {
		absRate = float64(absorbedConcepts) / float64(numConcepts)
	}

	return &FeatureAbsorptionReport{
		Family:                 dict.Family(),
		FeatureCount:           featCount,
		InputDim:               inDim,
		SampleCount:            nSamples,
		ActiveCoverage:         coverage,
		DeadFeatureRatio:       deadRatio,
		MeanL0Sparsity:         meanL0,
		MeanReconstructionR2:   r2,
		MeanPolysemanticity:    meanPoly,
		AbsorptionRate:         absRate,
		FeaturePolysemanticity: polyScores,
		ConceptMetrics:         conceptMetrics,
	}, nil
}

// AlignSAEDictionaries aligns features across two SAE dictionaries (e.g., Gemma Scope, Llama-Scope, Qwen-SAE)
// using paired corpus activations, Linear CKA, and Procrustes-style structural cosine matching.
func AlignSAEDictionaries(
	source, target SAEDictionary,
	sourceActs, targetActs [][]float64,
	topK int,
) (*CrossArchAlignmentResult, error) {
	if source == nil || source.Weights() == nil || target == nil || target.Weights() == nil {
		return nil, fmt.Errorf("both source and target SAE dictionaries must be loaded")
	}
	if len(sourceActs) == 0 || len(sourceActs) != len(targetActs) {
		return nil, fmt.Errorf("sourceActs (%d) and targetActs (%d) must be non-empty and equal length", len(sourceActs), len(targetActs))
	}

	nSamples := len(sourceActs)
	srcW := source.Weights()
	tgtW := target.Weights()

	srcFeatsMat := make([][]float64, nSamples)
	tgtFeatsMat := make([][]float64, nSamples)

	for s := 0; s < nSamples; s++ {
		sf, err := source.Encode(sourceActs[s])
		if err != nil {
			return nil, fmt.Errorf("encode source sample %d: %w", s, err)
		}
		tf, err := target.Encode(targetActs[s])
		if err != nil {
			return nil, fmt.Errorf("encode target sample %d: %w", s, err)
		}
		srcFeatsMat[s] = sf
		tgtFeatsMat[s] = tf
	}

	cka := computeLinearCKAMatrices(srcFeatsMat, tgtFeatsMat)

	// Build feature-wise activation profiles: srcProfiles[i][s], tgtProfiles[j][s]
	srcProfiles := transposeMatrix(srcFeatsMat, srcW.FeatureCount)
	tgtProfiles := transposeMatrix(tgtFeatsMat, tgtW.FeatureCount)

	// For each source feature, find best matching target feature combining activation profile cosine
	// and decoder structural cosine (aligned via common prefix dimension or normalized spectrum)
	pairs := make([]AlignedFeaturePair, 0, srcW.FeatureCount)
	for i := 0; i < srcW.FeatureCount; i++ {
		bestJ := 0
		bestCombined := -1.0
		bestActCos := 0.0
		bestProcCos := 0.0

		for j := 0; j < tgtW.FeatureCount; j++ {
			actCos := math.Max(0, sliceCosine(srcProfiles[i], tgtProfiles[j]))
			procCos := 0.0
			if i < len(srcW.DecoderWeights) && j < len(tgtW.DecoderWeights) {
				procCos = math.Abs(alignedDecoderCosine(srcW.DecoderWeights[i], tgtW.DecoderWeights[j]))
			}
			combined := 0.70*actCos + 0.30*procCos
			if combined > bestCombined {
				bestCombined = combined
				bestJ = j
				bestActCos = actCos
				bestProcCos = procCos
			}
		}

		pairs = append(pairs, AlignedFeaturePair{
			SourceFeature:    i,
			TargetFeature:    bestJ,
			ActivationCosine: bestActCos,
			ProcrustesCosine: bestProcCos,
			CombinedScore:    bestCombined,
		})
	}

	sort.Slice(pairs, func(a, b int) bool {
		if math.Abs(pairs[a].CombinedScore-pairs[b].CombinedScore) > 1e-12 {
			return pairs[a].CombinedScore > pairs[b].CombinedScore
		}
		return pairs[a].SourceFeature < pairs[b].SourceFeature
	})

	if topK > 0 && topK < len(pairs) {
		pairs = pairs[:topK]
	}

	var sumCos float64
	for _, p := range pairs {
		sumCos += p.CombinedScore
	}
	meanCos := 0.0
	if len(pairs) > 0 {
		meanCos = sumCos / float64(len(pairs))
	}

	return &CrossArchAlignmentResult{
		SourceFamily:        source.Family(),
		TargetFamily:        target.Family(),
		LinearCKA:           cka,
		MeanAlignmentCosine: meanCos,
		AlignedPairs:        pairs,
	}, nil
}

func computeLinearCKAMatrices(X, Y [][]float64) float64 {
	n := len(X)
	if n == 0 || len(Y) != n {
		return 0.0
	}
	cX := centerColumns(X)
	cY := centerColumns(Y)

	// Compute Gram matrices K = cX * cX^T and L = cY * cY^T
	var hsicXY, normK, normL float64
	for i := 0; i < n; i++ {
		for j := 0; j < n; j++ {
			kij := sliceDot(cX[i], cX[j])
			lij := sliceDot(cY[i], cY[j])
			hsicXY += kij * lij
			normK += kij * kij
			normL += lij * lij
		}
	}
	denom := math.Sqrt(normK * normL)
	if denom < 1e-12 {
		return 0.0
	}
	cka := hsicXY / denom
	if cka < 0 {
		return 0.0
	}
	if cka > 1.0 {
		return 1.0
	}
	return cka
}

func centerColumns(M [][]float64) [][]float64 {
	n := len(M)
	if n == 0 || len(M[0]) == 0 {
		return M
	}
	d := len(M[0])
	means := make([]float64, d)
	for i := 0; i < n; i++ {
		for j := 0; j < d; j++ {
			means[j] += M[i][j] / float64(n)
		}
	}
	out := make([][]float64, n)
	for i := 0; i < n; i++ {
		out[i] = make([]float64, d)
		for j := 0; j < d; j++ {
			out[i][j] = M[i][j] - means[j]
		}
	}
	return out
}

func transposeMatrix(M [][]float64, cols int) [][]float64 {
	rows := len(M)
	out := make([][]float64, cols)
	for c := 0; c < cols; c++ {
		out[c] = make([]float64, rows)
		for r := 0; r < rows; r++ {
			if c < len(M[r]) {
				out[c][r] = M[r][c]
			}
		}
	}
	return out
}

func alignedDecoderCosine(a, b []float64) float64 {
	minLen := len(a)
	if len(b) < minLen {
		minLen = len(b)
	}
	if minLen == 0 {
		return 0.0
	}
	return sliceCosine(a[:minLen], b[:minLen])
}

func sliceDot(a, b []float64) float64 {
	var s float64
	for i := range a {
		s += a[i] * b[i]
	}
	return s
}

func sliceNorm(a []float64) float64 {
	return math.Sqrt(sliceDot(a, a))
}

func normalizeSlice(a []float64) []float64 {
	n := sliceNorm(a)
	out := make([]float64, len(a))
	if n < 1e-12 {
		return out
	}
	for i, v := range a {
		out[i] = v / n
	}
	return out
}

func sliceCosine(a, b []float64) float64 {
	na := sliceNorm(a)
	nb := sliceNorm(b)
	if na < 1e-12 || nb < 1e-12 {
		return 0.0
	}
	return sliceDot(a, b) / (na * nb)
}
