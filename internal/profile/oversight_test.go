package profile

import (
	"math"
	"math/rand"
	"testing"
)

func TestExtractOversightDirection_Empty(t *testing.T) {
	_, err := ExtractOversightDirection(nil, nil)
	if err == nil {
		t.Error("expected error for nil activations, got nil")
	}

	_, err = ExtractOversightDirection([][][]float64{}, [][][]float64{})
	if err == nil {
		t.Error("expected error for empty activations, got nil")
	}
}

func TestExtractOversightDirection_Basic(t *testing.T) {
	// 2 samples, 2 layers, 3 dimensions
	monitored := [][][]float64{
		{
			{1.0, 2.0, 3.0}, // Layer 0
			{4.0, 5.0, 6.0}, // Layer 1
		},
		{
			{1.5, 2.5, 3.5}, // Layer 0
			{4.5, 5.5, 6.5}, // Layer 1
		},
	}

	deployed := [][][]float64{
		{
			{0.0, 0.0, 0.0}, // Layer 0
			{1.0, 1.0, 1.0}, // Layer 1
		},
		{
			{0.5, 0.5, 0.5}, // Layer 0
			{1.5, 1.5, 1.5}, // Layer 1
		},
	}

	directions, err := ExtractOversightDirection(monitored, deployed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if len(directions) != 2 {
		t.Fatalf("expected 2 layers, got %d", len(directions))
	}

	for layerIdx, dir := range directions {
		if len(dir) != 3 {
			t.Errorf("layer %d: expected direction dimension 3, got %d", layerIdx, len(dir))
		}

		// Verify norm is approximately 1.0
		norm := 0.0
		for _, x := range dir {
			norm += x * x
		}
		norm = math.Sqrt(norm)
		if math.Abs(norm-1.0) > 1e-9 {
			t.Errorf("layer %d: expected unit vector, got norm %f", layerIdx, norm)
		}
	}
}

func TestOversightAblator_Ablate(t *testing.T) {
	// Setup oversight direction pointing along x-axis for Layer 0, and y-axis for Layer 1
	directions := map[int][]float64{
		0: {1.0, 0.0, 0.0},
		1: {0.0, 1.0, 0.0},
	}

	oa := NewOversightAblator(directions)

	// Test Layer 0: Activation has components along x and y.
	// Ablation should completely remove x component.
	act0 := []float64{3.0, 4.0, 5.0}
	ablated0 := oa.Ablate(0, act0)
	expected0 := []float64{0.0, 4.0, 5.0}

	for i := range ablated0 {
		if math.Abs(ablated0[i]-expected0[i]) > 1e-9 {
			t.Errorf("layer 0 dim %d: got %f, want %f", i, ablated0[i], expected0[i])
		}
	}

	// Verify dot product of ablated activation and oversight direction is 0
	dot := 0.0
	for i := range ablated0 {
		dot += ablated0[i] * directions[0][i]
	}
	if math.Abs(dot) > 1e-9 {
		t.Errorf("expected orthogonal output (dot product 0), got %f", dot)
	}

	// Test Layer 1: Ablation should completely remove y component.
	act1 := []float64{3.0, 4.0, 5.0}
	ablated1 := oa.Ablate(1, act1)
	expected1 := []float64{3.0, 0.0, 5.0}

	for i := range ablated1 {
		if math.Abs(ablated1[i]-expected1[i]) > 1e-9 {
			t.Errorf("layer 1 dim %d: got %f, want %f", i, ablated1[i], expected1[i])
		}
	}

	// Test Layer 2 (no direction registered): Should remain unchanged
	act2 := []float64{3.0, 4.0, 5.0}
	ablated2 := oa.Ablate(2, act2)
	for i := range ablated2 {
		if ablated2[i] != act2[i] {
			t.Errorf("unregistered layer: got %f, want %f", ablated2[i], act2[i])
		}
	}
}

func TestOversightAblator_AblateActivations(t *testing.T) {
	directions := map[int][]float64{
		0: {1.0, 0.0},
		1: {0.0, 1.0},
	}
	oa := NewOversightAblator(directions)

	activations := [][][]float64{
		{
			{2.0, 3.0},
			{4.0, 5.0},
		},
		{
			{-1.0, 10.0},
			{8.0, -2.0},
		},
	}

	ablated := oa.AblateActivations(activations)

	if len(ablated) != 2 {
		t.Fatalf("expected 2 samples, got %d", len(ablated))
	}

	// Sample 0, Layer 0: x-component should be ablated (remain y = 3.0)
	if math.Abs(ablated[0][0][0]) > 1e-9 || math.Abs(ablated[0][0][1]-3.0) > 1e-9 {
		t.Errorf("Sample 0 Layer 0 got %v, want [0, 3]", ablated[0][0])
	}

	// Sample 0, Layer 1: y-component should be ablated (remain x = 4.0)
	if math.Abs(ablated[0][1][0]-4.0) > 1e-9 || math.Abs(ablated[0][1][1]) > 1e-9 {
		t.Errorf("Sample 0 Layer 1 got %v, want [4, 0]", ablated[0][1])
	}
}

func TestOversightAblator_RandomizedFuzz(t *testing.T) {
	r := rand.New(rand.NewSource(42))

	numLayers := 5
	dim := 128
	numSamples := 10

	monitored := make([][][]float64, numSamples)
	deployed := make([][][]float64, numSamples)

	for s := 0; s < numSamples; s++ {
		monitored[s] = make([][]float64, numLayers)
		deployed[s] = make([][]float64, numLayers)
		for l := 0; l < numLayers; l++ {
			monitored[s][l] = make([]float64, dim)
			deployed[s][l] = make([]float64, dim)
			for d := 0; d < dim; d++ {
				monitored[s][l][d] = r.Float64()*10 - 5
				deployed[s][l][d] = r.Float64()*10 - 5
			}
		}
	}

	directions, err := ExtractOversightDirection(monitored, deployed)
	if err != nil {
		t.Fatalf("unexpected error in extraction: %v", err)
	}

	oa := NewOversightAblator(directions)

	// Verify projection properties on arbitrary test vectors
	for l := 0; l < numLayers; l++ {
		testVec := make([]float64, dim)
		for d := 0; d < dim; d++ {
			testVec[d] = r.Float64()*20 - 10
		}

		ablated := oa.Ablate(l, testVec)

		// Dot product with extracted direction should be extremely close to 0
		dir := directions[l]
		dot := 0.0
		for d := 0; d < dim; d++ {
			dot += ablated[d] * dir[d]
		}

		if math.Abs(dot) > 1e-12 {
			t.Errorf("Layer %d: dot product after ablation is %e, expected close to 0", l, dot)
		}
	}
}

func TestNewOversightAblator_NilDirections(t *testing.T) {
	oa := NewOversightAblator(nil)
	if oa.OversightDirections == nil {
		t.Fatal("expected non-nil OversightDirections map")
	}
}

func TestExtractOversightDirection_ZeroNorm(t *testing.T) {
	// monitored and deployed are identical, so difference in means should be zero (norm <= 1e-10)
	monitored := [][][]float64{{{1.0, 2.0}}}
	deployed := [][][]float64{{{1.0, 2.0}}}
	dirs, err := ExtractOversightDirection(monitored, deployed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if len(dirs) != 1 {
		t.Fatalf("expected 1 layer, got %d", len(dirs))
	}
	for i, val := range dirs[0] {
		if val != 0.0 {
			t.Errorf("expected 0 at index %d, got %f", i, val)
		}
	}
}

func TestOversightAblator_Ablate_NilAndDimensionMismatch(t *testing.T) {
	// Nil directions map
	oa := &OversightAblator{OversightDirections: nil}
	act := []float64{1.0, 2.0}
	res := oa.Ablate(0, act)
	if len(res) != len(act) || res[0] != act[0] || res[1] != act[1] {
		t.Error("expected unchanged activation when directions are nil")
	}

	// Dimension mismatch
	oa2 := NewOversightAblator(map[int][]float64{0: {1.0, 0.0, 0.0}})
	res2 := oa2.Ablate(0, act)
	if len(res2) != len(act) || res2[0] != act[0] || res2[1] != act[1] {
		t.Error("expected unchanged activation on dimension mismatch")
	}
}

func TestOversightAblator_AblateActivations_Empty(t *testing.T) {
	oa := NewOversightAblator(nil)
	res := oa.AblateActivations(nil)
	if res != nil {
		t.Error("expected nil output for nil activations")
	}

	res2 := oa.AblateActivations([][][]float64{})
	if len(res2) != 0 {
		t.Error("expected empty output for empty activations")
	}
}

func TestExtractOversightDirection_MismatchedLayers(t *testing.T) {
	monitored := [][][]float64{
		{{1.0, 2.0}, {3.0, 4.0}},
		{{1.0, 2.0}}, // only 1 layer
	}
	deployed := [][][]float64{
		{{1.0, 2.0}, {3.0, 4.0}},
	}
	dirs, err := ExtractOversightDirection(monitored, deployed)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// For layer 1, pos will only have monitored[0][1], not monitored[1][1] (which doesn't exist)
	if len(dirs) != 2 {
		t.Errorf("expected 2 layers, got %d", len(dirs))
	}
}

