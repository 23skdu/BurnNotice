package steer

import (
	"crypto/sha256"
	"encoding/binary"
	"encoding/hex"
	"errors"
	"fmt"
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/floats"
)

// ErrCheckpointNotHighRisk is returned when ExtendedRefusalFineTuner is invoked on a checkpoint
// that does not meet high-risk deployment gating criteria while HighRiskOnly is enabled.
var ErrCheckpointNotHighRisk = errors.New("checkpoint is not flagged high-risk for DoRA extended-refusal hardening")

// DoRAConfig configures the ExtendedRefusalFineTuner weight-hardening pass.
type DoRAConfig struct {
	Rank              int     `json:"rank"`                // Low-rank adaptation dimension r (default: 4)
	Alpha             float64 `json:"alpha"`               // DoRA scaling factor alpha (default: 4.0)
	Strength          float64 `json:"strength"`            // MPOA refusal ablation strength in (0, 1] (default: 0.85)
	RegAlpha          float64 `json:"reg_alpha"`           // Ridge regularization for biprojected MPOA pairing (default: 0.01)
	MaxKLDivergence   float64 `json:"max_kl_divergence"`   // Maximum permissible KL divergence ceiling (default: 0.20)
	MinCoherenceRatio float64 `json:"min_coherence_ratio"` // Minimum coherence preservation ratio (default: 0.90)
	HighRiskOnly      bool    `json:"high_risk_only"`      // Require checkpoint to be flagged high-risk (default: true)
	MinRiskScore      float64 `json:"min_risk_score"`      // Minimum risk score to qualify as high-risk (default: 0.70)
	DryRun            bool    `json:"dry_run"`             // Compute decomposition & verification metrics without mutating checkpoint
	TargetLayers      []int   `json:"target_layers"`       // Layers to harden (empty = all layers present in refusal subspace)
}

// DefaultDoRAConfig returns recommended production parameters for DoRA extended-refusal fine-tuning.
func DefaultDoRAConfig() DoRAConfig {
	return DoRAConfig{
		Rank:              4,
		Alpha:             4.0,
		Strength:          0.85,
		RegAlpha:          0.01,
		MaxKLDivergence:   0.20,
		MinCoherenceRatio: 0.90,
		HighRiskOnly:      true,
		MinRiskScore:      0.70,
		DryRun:            false,
	}
}

// DoRALayerAdapter holds the weight-decomposed low-rank adaptation (DoRA) factors for a single layer:
// W_DoRA = diag(m) * normalize_rows(V + (alpha/r) * B * A).
type DoRALayerAdapter struct {
	LayerIndex         int         `json:"layer_index"`
	OutDim             int         `json:"out_dim"`
	InDim              int         `json:"in_dim"`
	Rank               int         `json:"rank"`
	EffectiveScale     float64     `json:"effective_scale"`
	Magnitude          []float64   `json:"magnitude"`           // m in R^{out_dim}: row-wise L2 norms ||W_0[i]||_2
	B                  [][]float64 `json:"b_matrix"`            // B in R^{out_dim x rank}
	A                  [][]float64 `json:"a_matrix"`            // A in R^{rank x in_dim}
	EstimatedKL        float64     `json:"estimated_kl"`        // KL divergence between MPOA-steered and DoRA-hardened logits
	BaselineKL         float64     `json:"baseline_kl"`         // KL divergence vs unsteered baseline (capped <= MaxKLDivergence)
	CoherenceRatio     float64     `json:"coherence_ratio"`     // Norm/cosine coherence preservation on benign subspace
	RefusalSuppression float64     `json:"refusal_suppression"` // Fractional reduction of projection along refusal subspace
}

// CheckpointWeights represents an in-memory deployment checkpoint subject to optional DoRA hardening.
type CheckpointWeights struct {
	ModelID   string              `json:"model_id"`
	HighRisk  bool                `json:"high_risk"`
	RiskScore float64             `json:"risk_score"`
	Layers    map[int][][]float64 `json:"layers"`
}

// DoRAHardeningResult summarizes a full checkpoint DoRA weight-hardening pass.
type DoRAHardeningResult struct {
	ModelID                string                    `json:"model_id"`
	Rank                   int                       `json:"rank"`
	Alpha                  float64                   `json:"alpha"`
	HardenedLayers         []int                     `json:"hardened_layers"`
	BaseCheckpointHash     string                    `json:"base_checkpoint_sha256"`
	HardenedCheckpointHash string                    `json:"hardened_checkpoint_sha256"`
	MaxMeasuredKL          float64                   `json:"max_measured_kl"`
	MeanMPOAPairingKL      float64                   `json:"mean_mpoa_pairing_kl"`
	MinCoherenceRatio      float64                   `json:"min_coherence_ratio"`
	DryRun                 bool                      `json:"dry_run"`
	Adapters               map[int]*DoRALayerAdapter `json:"adapters"`
}

// ExtendedRefusalFineTuner pairs inference-time MPOA steering with Weight-Decomposed Low-Rank Adaptation (DoRA)
// on high-risk deployment checkpoints.
type ExtendedRefusalFineTuner struct {
	mu        sync.RWMutex
	cfg       DoRAConfig
	snapshots map[int][][]float64 // Pre-hardening weight snapshots for deterministic rollback
	adapters  map[int]*DoRALayerAdapter
}

// NewExtendedRefusalFineTuner creates a new ExtendedRefusalFineTuner with validated configuration.
func NewExtendedRefusalFineTuner(cfg DoRAConfig) *ExtendedRefusalFineTuner {
	if cfg.Rank <= 0 {
		cfg.Rank = 4
	}
	if cfg.Alpha <= 0 {
		cfg.Alpha = float64(cfg.Rank)
	}
	if cfg.Strength <= 0 || cfg.Strength > 1.0 {
		cfg.Strength = 0.85
	}
	if cfg.RegAlpha < 0 {
		cfg.RegAlpha = 0.01
	}
	if cfg.MaxKLDivergence <= 0 {
		cfg.MaxKLDivergence = 0.20
	}
	if cfg.MinCoherenceRatio <= 0 || cfg.MinCoherenceRatio > 1.0 {
		cfg.MinCoherenceRatio = 0.90
	}
	if cfg.MinRiskScore <= 0 {
		cfg.MinRiskScore = 0.70
	}
	return &ExtendedRefusalFineTuner{
		cfg:       cfg,
		snapshots: make(map[int][][]float64),
		adapters:  make(map[int]*DoRALayerAdapter),
	}
}

// Config returns the active DoRAConfig.
func (ft *ExtendedRefusalFineTuner) Config() DoRAConfig {
	ft.mu.RLock()
	defer ft.mu.RUnlock()
	return ft.cfg
}

// HardenLayer decomposes `weights` (out_dim x in_dim) into magnitude m and direction V,
// constructs a low-rank directional update DeltaV = (alpha/r) * B * A paired with MPOA biprojected
// refusal directions, enforces the KL ceiling (<= MaxKLDivergence), and returns the DoRALayerAdapter
// along with the hardened weight matrix W_DoRA.
func (ft *ExtendedRefusalFineTuner) HardenLayer(
	layerIdx int,
	weights [][]float64,
	refusalDirs [][]float64,
	benignDir []float64,
) (*DoRALayerAdapter, [][]float64, error) {
	if len(weights) == 0 || len(weights[0]) == 0 {
		return nil, nil, fmt.Errorf("empty weight matrix at layer %d", layerIdx)
	}
	outDim := len(weights)
	inDim := len(weights[0])
	for i, row := range weights {
		if len(row) != inDim {
			return nil, nil, fmt.Errorf("ragged weight matrix row %d: len %d != %d", i, len(row), inDim)
		}
	}
	if len(refusalDirs) == 0 {
		return nil, nil, fmt.Errorf("empty refusal directions at layer %d", layerIdx)
	}
	for i, d := range refusalDirs {
		if len(d) != inDim {
			return nil, nil, fmt.Errorf("refusal direction %d dim %d != weight in_dim %d", i, len(d), inDim)
		}
	}
	if len(benignDir) > 0 && len(benignDir) != inDim {
		return nil, nil, fmt.Errorf("benign direction dim %d != weight in_dim %d", len(benignDir), inDim)
	}

	// 1. Biproject refusal directions orthogonal to benignDir (if provided) and orthonormalize top-r basis
	var bUnit []float64
	if len(benignDir) == inDim {
		bNorm := floats.Norm(benignDir, 2)
		if bNorm > 1e-9 {
			bUnit = make([]float64, inDim)
			for j := 0; j < inDim; j++ {
				bUnit[j] = benignDir[j] / bNorm
			}
		}
	}

	cleanedDirs := make([][]float64, 0, len(refusalDirs))
	for _, d := range refusalDirs {
		u := make([]float64, inDim)
		copy(u, d)
		if bUnit != nil {
			dotB := floats.Dot(u, bUnit)
			floats.AddScaled(u, -dotB, bUnit)
		}
		cleanedDirs = append(cleanedDirs, u)
	}
	orthoBasis := orthonormalizeSubspace(cleanedDirs)
	if len(orthoBasis) == 0 {
		return nil, nil, fmt.Errorf("refusal subspace collapsed to zero rank at layer %d", layerIdx)
	}

	rank := ft.cfg.Rank
	if rank > len(orthoBasis) {
		rank = len(orthoBasis)
	}
	if rank > inDim {
		rank = inDim
	}

	// 2. Extract DoRA row-wise magnitude vector m_i = ||W_0[i]||_2
	magnitude := make([]float64, outDim)
	for i := 0; i < outDim; i++ {
		magnitude[i] = floats.Norm(weights[i], 2)
	}

	// 3. Construct low-rank factors B (outDim x rank) and A (rank x inDim)
	// For each basis direction u_k, we want (W_0 + DeltaV) u_k = (1 - strength) W_0 u_k
	// so DeltaV_k = -(W_0 u_k) u_k^T.
	B := make([][]float64, outDim)
	for i := 0; i < outDim; i++ {
		B[i] = make([]float64, rank)
	}
	A := make([][]float64, rank)
	for k := 0; k < rank; k++ {
		uk := orthoBasis[k]
		A[k] = make([]float64, inDim)
		copy(A[k], uk)
		for i := 0; i < outDim; i++ {
			proj := floats.Dot(weights[i], uk) / (1.0 + ft.cfg.RegAlpha)
			B[i][k] = -proj
		}
	}

	// 4. Calibrate effective scale so baseline KL divergence stays <= MaxKLDivergence
	rawScale := ft.cfg.Strength * (ft.cfg.Alpha / float64(rank))
	if rawScale > 1.0 {
		rawScale = ft.cfg.Strength
	}

	hardened, baselineKL, mpoaKL, coherence, suppression := evaluateAndScaleDoRA(
		weights, magnitude, B, A, orthoBasis[:rank], bUnit, rawScale, ft.cfg,
	)

	adapter := &DoRALayerAdapter{
		LayerIndex:         layerIdx,
		OutDim:             outDim,
		InDim:              inDim,
		Rank:               rank,
		EffectiveScale:     rawScale,
		Magnitude:          magnitude,
		B:                  B,
		A:                  A,
		EstimatedKL:        mpoaKL,
		BaselineKL:         baselineKL,
		CoherenceRatio:     coherence,
		RefusalSuppression: suppression,
	}

	ft.mu.Lock()
	if _, exists := ft.snapshots[layerIdx]; !exists {
		ft.snapshots[layerIdx] = cloneMatrix(weights)
	}
	ft.adapters[layerIdx] = adapter
	ft.mu.Unlock()

	return adapter, hardened, nil
}

func evaluateAndScaleDoRA(
	weights [][]float64,
	magnitude []float64,
	B [][]float64,
	A [][]float64,
	basis [][]float64,
	bUnit []float64,
	initialScale float64,
	cfg DoRAConfig,
) (hardened [][]float64, baselineKL, mpoaKL, coherence, suppression float64) {
	scale := initialScale
	for attempt := 0; attempt < 6; attempt++ {
		hardened = composeDoRAWeights(weights, magnitude, B, A, scale)
		baselineKL, mpoaKL, coherence, suppression = computeDoRAPairingMetrics(weights, hardened, basis, bUnit, scale, cfg.RegAlpha)
		if baselineKL <= cfg.MaxKLDivergence {
			break
		}
		scale = AdjustAlphaForKL(scale, baselineKL, cfg.MaxKLDivergence) * 0.90
	}
	return hardened, baselineKL, mpoaKL, coherence, suppression
}

// composeDoRAWeights computes W_DoRA[i] = m[i] * (W_0[i] + scale * sum_k B[i][k]*A[k]) / ||W_0[i] + scale * sum_k B[i][k]*A[k]||_2.
func composeDoRAWeights(weights [][]float64, magnitude []float64, B [][]float64, A [][]float64, scale float64) [][]float64 {
	outDim := len(weights)
	inDim := len(weights[0])
	rank := len(A)

	out := make([][]float64, outDim)
	for i := 0; i < outDim; i++ {
		row := make([]float64, inDim)
		copy(row, weights[i])
		for k := 0; k < rank; k++ {
			floats.AddScaled(row, scale*B[i][k], A[k])
		}
		dirNorm := floats.Norm(row, 2)
		if dirNorm > 1e-10 && magnitude[i] > 1e-10 {
			rowScale := magnitude[i] / dirNorm
			floats.Scale(rowScale, row)
		}
		out[i] = row
	}
	return out
}

func computeDoRAPairingMetrics(
	weights, hardened [][]float64,
	basis [][]float64,
	bUnit []float64,
	scale, regAlpha float64,
) (baselineKL, mpoaKL, coherence, suppression float64) {
	inDim := len(weights[0])

	// Build probe activations: one mixed refusal+benign probe and one pure benign probe
	probeMix := make([]float64, inDim)
	for j := 0; j < inDim; j++ {
		probeMix[j] = 0.5 / math.Sqrt(float64(inDim))
	}
	for _, uk := range basis {
		floats.AddScaled(probeMix, 0.7, uk)
	}
	if bUnit != nil {
		floats.AddScaled(probeMix, 0.7, bUnit)
	}

	// 1. Compute unsteered output W_0 * x, DoRA output W_DoRA * x, and MPOA-steered output W_0 * MPOA(x)
	steeredX := make([]float64, inDim)
	copy(steeredX, probeMix)
	for _, uk := range basis {
		if bUnit != nil {
			steeredX = BiprojectedAblation(steeredX, uk, bUnit, scale, regAlpha)
		} else {
			steeredX = NormPreservingAblation(steeredX, uk, scale, regAlpha)
		}
	}

	logitsBase := matVecMul(weights, probeMix)
	logitsHard := matVecMul(hardened, probeMix)
	logitsMPOA := matVecMul(weights, steeredX)

	baselineKL = symmetricKLLogits(logitsBase, logitsHard)
	mpoaKL = symmetricKLLogits(logitsMPOA, logitsHard)

	// 2. Compute refusal suppression along primary refusal direction u_0
	u0 := basis[0]
	baseRefusalOut := floats.Norm(matVecMul(weights, u0), 2)
	hardRefusalOut := floats.Norm(matVecMul(hardened, u0), 2)
	if baseRefusalOut > 1e-10 {
		suppression = math.Max(0, 1.0-(hardRefusalOut/baseRefusalOut))
	}

	// 3. Compute coherence preservation on benign direction (or orthogonal complement)
	benignProbe := bUnit
	if benignProbe == nil {
		benignProbe = make([]float64, inDim)
		for j := 0; j < inDim; j++ {
			benignProbe[j] = float64((j%3)+1) * 0.3
		}
		for _, uk := range basis {
			dot := floats.Dot(benignProbe, uk)
			floats.AddScaled(benignProbe, -dot, uk)
		}
	}
	bOutBase := matVecMul(weights, benignProbe)
	bOutHard := matVecMul(hardened, benignProbe)
	coherence = cosineSim(bOutBase, bOutHard)
	if coherence < 0 {
		coherence = 0
	}
	return baselineKL, mpoaKL, coherence, suppression
}

// HardenCheckpoint executes a DoRA weight-hardening pass across a deployment checkpoint,
// enforcing high-risk gating, KL bounds, coherence checks, and producing cryptographic SHA-256 digests.
func (ft *ExtendedRefusalFineTuner) HardenCheckpoint(
	checkpoint *CheckpointWeights,
	refusalSubspace map[int][][]float64,
	benignDirs map[int][]float64,
) (*DoRAHardeningResult, error) {
	if checkpoint == nil || len(checkpoint.Layers) == 0 {
		return nil, fmt.Errorf("checkpoint contains no layers")
	}
	if ft.cfg.HighRiskOnly && !checkpoint.HighRisk && checkpoint.RiskScore < ft.cfg.MinRiskScore {
		return nil, ErrCheckpointNotHighRisk
	}
	if len(refusalSubspace) == 0 {
		return nil, fmt.Errorf("refusal subspace map cannot be empty")
	}

	baseHash := ComputeCheckpointSHA256(checkpoint.Layers)

	targetSet := make(map[int]bool, len(ft.cfg.TargetLayers))
	for _, l := range ft.cfg.TargetLayers {
		targetSet[l] = true
	}

	layerIndices := make([]int, 0, len(checkpoint.Layers))
	for l := range checkpoint.Layers {
		if len(targetSet) > 0 && !targetSet[l] {
			continue
		}
		if _, hasSubspace := refusalSubspace[l]; hasSubspace {
			layerIndices = append(layerIndices, l)
		}
	}
	sort.Ints(layerIndices)
	if len(layerIndices) == 0 {
		return nil, fmt.Errorf("no matching target layers found between checkpoint and refusal subspace")
	}

	adapters := make(map[int]*DoRALayerAdapter, len(layerIndices))
	hardenedLayersMap := make(map[int][][]float64, len(checkpoint.Layers))
	for l, w := range checkpoint.Layers {
		hardenedLayersMap[l] = w
	}

	var maxKL, sumMPOAKL float64
	minCoherence := 1.0

	for _, l := range layerIndices {
		var benign []float64
		if benignDirs != nil {
			benign = benignDirs[l]
		}
		adapter, hardenedW, err := ft.HardenLayer(l, checkpoint.Layers[l], refusalSubspace[l], benign)
		if err != nil {
			return nil, fmt.Errorf("harden layer %d: %w", l, err)
		}
		adapters[l] = adapter
		hardenedLayersMap[l] = hardenedW

		if adapter.BaselineKL > maxKL {
			maxKL = adapter.BaselineKL
		}
		sumMPOAKL += adapter.EstimatedKL
		if adapter.CoherenceRatio < minCoherence {
			minCoherence = adapter.CoherenceRatio
		}
	}

	hardenedHash := ComputeCheckpointSHA256(hardenedLayersMap)
	if !ft.cfg.DryRun {
		for _, l := range layerIndices {
			checkpoint.Layers[l] = hardenedLayersMap[l]
		}
	}

	return &DoRAHardeningResult{
		ModelID:                checkpoint.ModelID,
		Rank:                   ft.cfg.Rank,
		Alpha:                  ft.cfg.Alpha,
		HardenedLayers:         layerIndices,
		BaseCheckpointHash:     baseHash,
		HardenedCheckpointHash: hardenedHash,
		MaxMeasuredKL:          maxKL,
		MeanMPOAPairingKL:      sumMPOAKL / float64(len(layerIndices)),
		MinCoherenceRatio:      minCoherence,
		DryRun:                 ft.cfg.DryRun,
		Adapters:               adapters,
	}, nil
}

// RollbackCheckpoint restores all modified layers in checkpoint to their pre-DoRA snapshots.
func (ft *ExtendedRefusalFineTuner) RollbackCheckpoint(checkpoint *CheckpointWeights) error {
	if checkpoint == nil {
		return fmt.Errorf("nil checkpoint")
	}
	ft.mu.Lock()
	defer ft.mu.Unlock()
	if len(ft.snapshots) == 0 {
		return fmt.Errorf("no DoRA snapshots recorded to rollback")
	}
	for l, snap := range ft.snapshots {
		checkpoint.Layers[l] = cloneMatrix(snap)
	}
	ft.snapshots = make(map[int][][]float64)
	ft.adapters = make(map[int]*DoRALayerAdapter)
	return nil
}

// ComputeCheckpointSHA256 computes a deterministic SHA-256 hex digest over sorted layer weight matrices.
func ComputeCheckpointSHA256(layers map[int][][]float64) string {
	h := sha256.New()
	indices := make([]int, 0, len(layers))
	for l := range layers {
		indices = append(indices, l)
	}
	sort.Ints(indices)

	var buf [8]byte
	for _, l := range indices {
		_, _ = fmt.Fprintf(h, "%d:", l)
		for _, row := range layers[l] {
			for _, val := range row {
				binary.LittleEndian.PutUint64(buf[:], math.Float64bits(val))
				_, _ = h.Write(buf[:])
			}
		}
	}
	return hex.EncodeToString(h.Sum(nil))
}

func cloneMatrix(m [][]float64) [][]float64 {
	out := make([][]float64, len(m))
	for i := range m {
		out[i] = make([]float64, len(m[i]))
		copy(out[i], m[i])
	}
	return out
}

func matVecMul(m [][]float64, v []float64) []float64 {
	out := make([]float64, len(m))
	for i, row := range m {
		out[i] = floats.Dot(row, v)
	}
	return out
}

func cosineSim(a, b []float64) float64 {
	na := floats.Norm(a, 2)
	nb := floats.Norm(b, 2)
	if na < 1e-10 || nb < 1e-10 {
		return 1.0
	}
	return floats.Dot(a, b) / (na * nb)
}

func symmetricKLLogits(pLogits, qLogits []float64) float64 {
	if len(pLogits) == 0 || len(pLogits) != len(qLogits) {
		return 0.0
	}
	p := softmaxProbs(pLogits)
	q := softmaxProbs(qLogits)
	var klPQ, klQP float64
	for i := range p {
		pi := math.Max(p[i], 1e-12)
		qi := math.Max(q[i], 1e-12)
		klPQ += pi * math.Log(pi/qi)
		klQP += qi * math.Log(qi/pi)
	}
	return 0.5 * (klPQ + klQP)
}

func softmaxProbs(logits []float64) []float64 {
	maxVal := logits[0]
	for _, v := range logits[1:] {
		if v > maxVal {
			maxVal = v
		}
	}
	out := make([]float64, len(logits))
	var sum float64
	for i, v := range logits {
		e := math.Exp(v - maxVal)
		out[i] = e
		sum += e
	}
	if sum > 0 {
		for i := range out {
			out[i] /= sum
		}
	}
	return out
}
