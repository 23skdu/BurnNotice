package steer

import "math"

// Dispatcher routes steering commands to the appropriate method.
type Dispatcher struct {
	TargetLayers map[int]bool
	Methods      map[int]Method
}

// NewDispatcher creates a dispatcher with the given target layers and per-layer methods.
func NewDispatcher(layers []int, methods map[int]Method) *Dispatcher {
	layerSet := make(map[int]bool, len(layers))
	for _, l := range layers {
		layerSet[l] = true
	}
	return &Dispatcher{
		TargetLayers: layerSet,
		Methods:      methods,
	}
}

// Apply routes the activation through the correct steering method for the given layer.
func (d *Dispatcher) Apply(layerIdx int, activation, direction []float64, strength float64) []float64 {
	if !d.TargetLayers[layerIdx] {
		return activation
	}
	method, ok := d.Methods[layerIdx]
	if !ok {
		method = MethodAddition
	}
	switch method {
	case MethodAddition:
		return ActAdd(activation, direction, strength)
	case MethodAngular:
		return Angular(activation, direction, strength)
	case MethodAblation:
		return Ablate(activation, direction)
	case MethodSelective:
		// Selective steering: dynamically check alignment projection along direction
		// direction is assumed to be the difference vector (truth - deception)
		v := normalizeVec(direction)
		proj := vecDot(activation, v)
		if proj >= 0 {
			// Compliant / not leaning towards deception, do not steer
			return activation
		}
		clamped := proj
		if clamped < -1 {
			clamped = -1
		}
		if clamped > 1 {
			clamped = 1
		}
		// Calculate the steering angle theta based on how far it is unaligned
		theta := strength * math.Asin(-clamped)
		return rotate2D(activation, v, theta)
	default:
		return activation
	}
}
