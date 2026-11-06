package steer

import "math"

// Method defines the steering intervention type.
type Method int

const (
	MethodAddition Method = iota
	MethodAngular
	MethodSelective
	MethodAblation
	MethodNormPreserving
	MethodBiprojection
)

// Command holds a steering intervention to apply.
type Command struct {
	Vector              []float64
	SecondaryVector     []float64 // Benign / harmless reference direction for biprojection
	Strength            float64
	Method              Method
	RegularizationAlpha float64
	AttentionStrength   float64
	MLPStrength         float64
	DirectionMix        float64
}

// NormPreservingAblation performs norm-preserving biprojected ablation (MPOA):
// it removes the component along direction and restores the original L2 norm
// of the activation, preventing representational degradation:
//
//	h_ortho = h - (h · v)/(v · v + lambda) * v
//	h' = h_ortho * (||h||_2 / ||h_ortho||_2)
func NormPreservingAblation(activation, direction []float64, alpha float64, regAlpha float64) []float64 {
	origNorm := vecNorm(activation)
	if origNorm < 1e-10 {
		return activation
	}

	vNormSq := vecDot(direction, direction)
	if vNormSq < 1e-10 {
		return activation
	}

	dot := vecDot(activation, direction)
	proj := dot / (vNormSq + regAlpha)

	ortho := make([]float64, len(activation))
	for i := range activation {
		ortho[i] = activation[i] - proj*direction[i]
	}

	orthoNorm := vecNorm(ortho)
	if orthoNorm < 1e-10 {
		return activation
	}

	// Restore original norm
	scale := origNorm / orthoNorm
	restored := make([]float64, len(activation))
	for i := range activation {
		restored[i] = ortho[i] * scale
	}

	if alpha >= 1.0 {
		return restored
	}
	if alpha <= 0.0 {
		return activation
	}

	// Interpolate between original and restored
	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = (1.0-alpha)*activation[i] + alpha*restored[i]
	}
	return result
}

// BiprojectedAblation removes the harmful direction while explicitly preserving/restoring
// the benign direction component and the original activation norm:
//
//	h_harm_free = h - (h · v_harm)/(||v_harm||^2 + reg) * v_harm
//	h_benign_restored = h_harm_free + (h · v_benign)/(||v_benign||^2 + reg) * v_benign
//	h' = h_benign_restored * (||h||_2 / ||h_benign_restored||_2)
func BiprojectedAblation(activation, harmDir, benignDir []float64, alpha, regAlpha float64) []float64 {
	origNorm := vecNorm(activation)
	if origNorm < 1e-10 || len(harmDir) != len(activation) {
		return activation
	}

	harmNormSq := vecDot(harmDir, harmDir)
	if harmNormSq < 1e-10 {
		return activation
	}

	// 1. Remove harmful projection
	dotHarm := vecDot(activation, harmDir)
	projHarm := dotHarm / (harmNormSq + regAlpha)

	hOrtho := make([]float64, len(activation))
	for i := range activation {
		hOrtho[i] = activation[i] - projHarm*harmDir[i]
	}

	// 2. If benign direction is provided, preserve its original component
	if len(benignDir) == len(activation) {
		benignNormSq := vecDot(benignDir, benignDir)
		if benignNormSq >= 1e-10 {
			origBenignDot := vecDot(activation, benignDir)
			curBenignDot := vecDot(hOrtho, benignDir)
			deficit := origBenignDot - curBenignDot
			scale := deficit / (benignNormSq + regAlpha)
			for i := range hOrtho {
				hOrtho[i] += scale * benignDir[i]
			}
		}
	}

	// 3. Restore original L2 norm
	orthoNorm := vecNorm(hOrtho)
	if orthoNorm < 1e-10 {
		return activation
	}
	normScale := origNorm / orthoNorm
	restored := make([]float64, len(activation))
	for i := range activation {
		restored[i] = hOrtho[i] * normScale
	}

	if alpha >= 1.0 {
		return restored
	}
	if alpha <= 0.0 {
		return activation
	}

	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = (1.0-alpha)*activation[i] + alpha*restored[i]
	}
	return result
}

// ResolveComponentStrengths computes the effective steering strength for a transformer sub-component.
// If component-specific strengths are not configured, default to attention=1.0 and MLP=0.5 (as MLP carries sensitive knowledge).
func ResolveComponentStrengths(cmd Command, component TransformerComponent) float64 {
	baseStrength := cmd.Strength
	if baseStrength == 0 {
		baseStrength = 1.0
	}

	switch component {
	case ComponentMLPDown, ComponentMLPUp, ComponentMLPGate:
		if cmd.MLPStrength > 0 {
			return baseStrength * cmd.MLPStrength
		}
		return baseStrength * 0.5 // Default MLP scaling
	case ComponentAttentionQuery, ComponentAttentionKey, ComponentAttentionValue, ComponentAttentionOut:
		if cmd.AttentionStrength > 0 {
			return baseStrength * cmd.AttentionStrength
		}
		return baseStrength * 1.0 // Default Attention scaling
	default:
		return baseStrength
	}
}

// ApplyCommand executes the configured steering command on the activation.
func ApplyCommand(activation []float64, cmd Command) []float64 {
	switch cmd.Method {
	case MethodAddition:
		return ActAdd(activation, cmd.Vector, cmd.Strength)
	case MethodAngular:
		return Angular(activation, cmd.Vector, cmd.Strength)
	case MethodAblation:
		return Ablate(activation, cmd.Vector)
	case MethodNormPreserving:
		strength := cmd.Strength
		if strength == 0 {
			strength = 1.0
		}
		return NormPreservingAblation(activation, cmd.Vector, strength, cmd.RegularizationAlpha)
	case MethodBiprojection:
		strength := cmd.Strength
		if strength == 0 {
			strength = 1.0
		}
		return BiprojectedAblation(activation, cmd.Vector, cmd.SecondaryVector, strength, cmd.RegularizationAlpha)
	default:
		return activation
	}
}

// ActAdd performs activation addition: h' = h + alpha * v
func ActAdd(activation, direction []float64, alpha float64) []float64 {
	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = activation[i] + alpha*direction[i]
	}
	return result
}

// Angular performs norm-preserving angular steering.
func Angular(activation, direction []float64, alpha float64) []float64 {
	aNorm := vecNorm(activation)
	if aNorm < 1e-10 {
		return activation
	}

	// Normalize the steering direction to make it a unit vector
	u := normalizeVec(direction)
	uNorm := vecNorm(u)
	if uNorm < 1e-10 {
		return activation
	}

	// Since u is a unit vector, proj is the scalar projection
	proj := vecDot(activation, u)

	// Compute the component orthogonal to u
	ortho := make([]float64, len(activation))
	for i := range activation {
		ortho[i] = activation[i] - proj*u[i]
	}
	orthoNorm := vecNorm(ortho)
	if orthoNorm < 1e-10 {
		return activation
	}

	theta := alpha * math.Pi
	cosT := math.Cos(theta)
	sinT := math.Sin(theta)

	// Compute the orthonormal basis vector w (orthogonal to activation, same norm)
	// w = orthoNorm * u - proj * (ortho / orthoNorm)
	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = cosT*activation[i] + sinT*(orthoNorm*u[i]-proj*ortho[i]/orthoNorm)
	}

	return result
}

// Ablate projects the activation orthogonal to the direction: h' = h - (h·v)/(v·v) * v
func Ablate(activation, direction []float64) []float64 {
	vNormSq := vecDot(direction, direction)
	if vNormSq < 1e-10 {
		return activation
	}

	dot := vecDot(activation, direction)
	proj := dot / vNormSq

	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = activation[i] - proj*direction[i]
	}
	return result
}

// SelectiveSteerer applies steering only to discriminative layers.
type SelectiveSteerer struct {
	TruthDir     map[int][]float64
	DeceptionDir map[int][]float64
	TargetLayers map[int]bool
}

// NewSelectiveSteerer creates a selective steerer.
func NewSelectiveSteerer(truthDir, deceptionDir map[int][]float64, layers []int) *SelectiveSteerer {
	layerSet := make(map[int]bool, len(layers))
	for _, l := range layers {
		layerSet[l] = true
	}
	return &SelectiveSteerer{
		TruthDir:     truthDir,
		DeceptionDir: deceptionDir,
		TargetLayers: layerSet,
	}
}

// IsDiscriminative returns true if truth and deception projections have opposite signs.
func (s *SelectiveSteerer) IsDiscriminative(layerIdx int, activation []float64) bool {
	if s.TruthDir == nil || s.DeceptionDir == nil {
		return false
	}
	truth, ok1 := s.TruthDir[layerIdx]
	deception, ok2 := s.DeceptionDir[layerIdx]
	if !ok1 || !ok2 || len(truth) != len(activation) || len(deception) != len(activation) {
		return false
	}
	truthProj := vecDot(activation, truth)
	deceptionProj := vecDot(activation, deception)
	return truthProj*deceptionProj < 0
}

// ComputeSteering returns a steering command if the layer is discriminative and deception-leaning.
func (s *SelectiveSteerer) ComputeSteering(layerIdx int, activation []float64, strength float64) *Command {
	if !s.TargetLayers[layerIdx] {
		return nil
	}
	if s.TruthDir == nil || s.DeceptionDir == nil {
		return nil
	}
	truth, ok1 := s.TruthDir[layerIdx]
	deception, ok2 := s.DeceptionDir[layerIdx]
	if !ok1 || !ok2 || len(truth) != len(activation) || len(deception) != len(activation) {
		return nil
	}
	if !s.IsDiscriminative(layerIdx, activation) {
		return nil
	}

	v := make([]float64, len(truth))
	for i := range truth {
		v[i] = truth[i] - deception[i]
	}
	v = normalizeVec(v)

	proj := vecDot(activation, v)
	if proj >= 0 {
		return nil
	}

	clamped := proj
	if clamped < -1 {
		clamped = -1
	}
	if clamped > 1 {
		clamped = 1
	}
	theta := strength * math.Asin(-clamped)

	steered := rotate2D(activation, v, theta)
	return &Command{
		Vector:   steered,
		Strength: strength,
		Method:   MethodSelective,
	}
}

func rotate2D(h, v []float64, theta float64) []float64 {
	proj := vecDot(h, v)
	ortho := make([]float64, len(h))
	for i := range h {
		ortho[i] = h[i] - proj*v[i]
	}
	orthoNorm := vecNorm(ortho)
	if orthoNorm < 1e-10 {
		return h
	}

	cosT := math.Cos(theta)
	sinT := math.Sin(theta)

	result := make([]float64, len(h))
	for i := range h {
		result[i] = cosT*h[i] + sinT*(orthoNorm*v[i]-proj*ortho[i]/orthoNorm)
	}
	return result
}

func vecNorm(v []float64) float64 {
	var sum float64
	for _, x := range v {
		sum += x * x
	}
	return math.Sqrt(sum)
}

func vecDot(a, b []float64) float64 {
	var sum float64
	for i := range a {
		sum += a[i] * b[i]
	}
	return sum
}

func normalizeVec(v []float64) []float64 {
	n := vecNorm(v)
	if n < 1e-10 {
		return v
	}
	result := make([]float64, len(v))
	for i := range v {
		result[i] = v[i] / n
	}
	return result
}
