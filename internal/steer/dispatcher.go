package steer

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
	default:
		return activation
	}
}
