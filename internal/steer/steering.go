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
	vNorm := vecNorm(direction)
	if vNorm < 1e-10 || aNorm < 1e-10 {
		return activation
	}

	dot := vecDot(activation, direction)
	parallelScale := dot / (vNorm * vNorm)

	parallel := make([]float64, len(direction))
	for i := range direction {
		parallel[i] = parallelScale * direction[i]
	}

	ortho := make([]float64, len(activation))
	for i := range activation {
		ortho[i] = activation[i] - parallel[i]
	}
	orthoNorm := vecNorm(ortho)

	theta := alpha * math.Pi

	cosT := math.Cos(theta)
	sinT := math.Sin(theta)

	result := make([]float64, len(activation))
	for i := range activation {
		result[i] = cosT*activation[i] + sinT*(orthoNorm*direction[i]/vNorm - parallelScale*ortho[i]/orthoNorm)
	}

	resultNorm := vecNorm(result)
	if resultNorm > 1e-10 {
		scale := aNorm / resultNorm
		for i := range result {
			result[i] *= scale
		}
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
	truthProj := vecDot(activation, s.TruthDir[layerIdx])
	deceptionProj := vecDot(activation, s.DeceptionDir[layerIdx])
	return truthProj*deceptionProj < 0
}

// ComputeSteering returns a steering command if the layer is discriminative and deception-leaning.
func (s *SelectiveSteerer) ComputeSteering(layerIdx int, activation []float64, strength float64) *Command {
	if !s.TargetLayers[layerIdx] {
		return nil
	}
	if !s.IsDiscriminative(layerIdx, activation) {
		return nil
	}

	truth := s.TruthDir[layerIdx]
	deception := s.DeceptionDir[layerIdx]

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
