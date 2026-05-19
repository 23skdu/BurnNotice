package steer

import "math"

// Method defines the steering intervention type.
type Method int

const (
	MethodAddition Method = iota
	MethodAngular
	MethodSelective
	MethodAblation
)

// Command holds a steering intervention to apply.
type Command struct {
	Vector   []float64
	Strength float64
	Method   Method
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
		result[i] = cosT*activation[i] + sinT*(orthoNorm*u[i] - proj*ortho[i]/orthoNorm)
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
		result[i] = cosT*h[i] + sinT*(orthoNorm*v[i] - proj*ortho[i]/orthoNorm)
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
