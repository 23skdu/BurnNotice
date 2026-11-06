package analysis

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestProjectLatentSpace_PCA(t *testing.T) {
	// Synthesize 4D points with 2 distinct clusters
	points := []LatentPoint{
		{ID: "t1", Vector: []float64{2.0, 3.0, 0.1, -0.2}, Category: CategoryTruthful, Layer: 0, Token: "The"},
		{ID: "t2", Vector: []float64{2.1, 2.9, 0.2, -0.1}, Category: CategoryTruthful, Layer: 1, Token: "The"},
		{ID: "t3", Vector: []float64{1.9, 3.1, 0.0, -0.3}, Category: CategoryTruthful, Layer: 2, Token: "The"},
		{ID: "d1", Vector: []float64{-2.0, -1.0, 0.5, 0.4}, Category: CategoryDeceptive, Layer: 0, Token: "Lie"},
		{ID: "d2", Vector: []float64{-2.2, -0.9, 0.4, 0.6}, Category: CategoryDeceptive, Layer: 1, Token: "Lie"},
		{ID: "d3", Vector: []float64{-1.8, -1.1, 0.6, 0.3}, Category: CategoryDeceptive, Layer: 2, Token: "Lie"},
	}

	proj, err := ProjectLatentSpace(points, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if proj.Method != "pca" {
		t.Errorf("expected method 'pca', got '%s'", proj.Method)
	}

	if len(proj.Points) != len(points) {
		t.Errorf("expected %d points, got %d", len(points), len(proj.Points))
	}

	// Variance explained must be valid positive ratios
	if proj.VarianceExplained[0] <= 0 || proj.VarianceExplained[1] <= 0 {
		t.Errorf("expected positive variance explained, got %v", proj.VarianceExplained)
	}

	// Check truthful cluster centroid vs deceptive cluster centroid
	truthCluster := proj.Clusters[CategoryTruthful]
	deceptCluster := proj.Clusters[CategoryDeceptive]

	if truthCluster.PointCount != 3 || deceptCluster.PointCount != 3 {
		t.Errorf("expected 3 points per cluster, got %d and %d", truthCluster.PointCount, deceptCluster.PointCount)
	}

	// Centroids should be well-separated along principal component
	diffX := truthCluster.Centroid[0] - deceptCluster.Centroid[0]
	diffY := truthCluster.Centroid[1] - deceptCluster.Centroid[1]
	dist := diffX*diffX + diffY*diffY
	if dist < 1.0 {
		t.Errorf("expected well-separated cluster centroids, squared distance = %f", dist)
	}
}

func TestProjectLatentSpace_BasisProjection(t *testing.T) {
	basisX := []float64{1.0, 0.0, 0.0, 0.0}
	basisY := []float64{0.0, 1.0, 0.0, 0.0}

	points := []LatentPoint{
		{ID: "p1", Vector: []float64{3.0, 4.0, 10.0, -5.0}, Category: CategoryRefusal},
		{ID: "p2", Vector: []float64{-2.0, 5.0, 8.0, 2.0}, Category: CategoryCreative},
	}

	proj, err := ProjectLatentSpace(points, basisX, basisY)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if proj.Method != "basis_projection" {
		t.Errorf("expected method 'basis_projection', got '%s'", proj.Method)
	}

	if proj.Points[0].ProjectedX != 3.0 || proj.Points[0].ProjectedY != 4.0 {
		t.Errorf("expected (3, 4), got (%f, %f)", proj.Points[0].ProjectedX, proj.Points[0].ProjectedY)
	}
	if proj.Points[1].ProjectedX != -2.0 || proj.Points[1].ProjectedY != 5.0 {
		t.Errorf("expected (-2, 5), got (%f, %f)", proj.Points[1].ProjectedX, proj.Points[1].ProjectedY)
	}
}

func TestBuildTokenTrajectories(t *testing.T) {
	// Points unordered by layer
	points := []LatentPoint{
		{PromptID: "prompt1", Token: "word", Layer: 2, ProjectedX: 4.0, ProjectedY: 0.0, Category: CategoryHarmful},
		{PromptID: "prompt1", Token: "word", Layer: 0, ProjectedX: 0.0, ProjectedY: 0.0, Category: CategoryHarmful},
		{PromptID: "prompt1", Token: "word", Layer: 1, ProjectedX: 2.0, ProjectedY: 0.0, Category: CategoryHarmful},
	}

	trajs := BuildTokenTrajectories(points)
	if len(trajs) != 1 {
		t.Fatalf("expected 1 trajectory, got %d", len(trajs))
	}

	traj := trajs[0]
	if len(traj.Points) != 3 {
		t.Fatalf("expected 3 points in trajectory, got %d", len(traj.Points))
	}

	// Should be sorted by Layer ascending: 0, 1, 2
	for i := 0; i < 3; i++ {
		if traj.Points[i].Layer != i {
			t.Errorf("expected point %d to have layer %d, got %d", i, i, traj.Points[i].Layer)
		}
	}

	// Velocity between 0->1 is 2.0, between 1->2 is 2.0
	if len(traj.Velocity) != 2 {
		t.Fatalf("expected 2 velocity entries, got %d", len(traj.Velocity))
	}
	if traj.Velocity[0] != 2.0 || traj.Velocity[1] != 2.0 {
		t.Errorf("expected velocity [2.0, 2.0], got %v", traj.Velocity)
	}
}

func TestGenerateSVG(t *testing.T) {
	points := []LatentPoint{
		{ID: "1", PromptID: "pr1", Token: "test", Layer: 0, ProjectedX: -1.5, ProjectedY: 2.0, Category: CategoryTruthful},
		{ID: "2", PromptID: "pr1", Token: "test", Layer: 1, ProjectedX: 0.5, ProjectedY: 1.0, Category: CategoryTruthful},
		{ID: "3", PromptID: "pr2", Token: "fake", Layer: 0, ProjectedX: 2.0, ProjectedY: -1.5, Category: CategoryDeceptive},
		{ID: "4", PromptID: "pr2", Token: "fake", Layer: 1, ProjectedX: 3.5, ProjectedY: -2.0, Category: CategoryDeceptive},
	}

	proj, err := ProjectLatentSpace(points, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	cfg := DefaultVisualizerConfig()
	svg, err := GenerateSVG(proj, cfg)
	if err != nil {
		t.Fatalf("failed to generate SVG: %v", err)
	}

	if !strings.HasPrefix(svg, "<svg") || !strings.HasSuffix(strings.TrimSpace(svg), "</svg>") {
		t.Errorf("expected valid SVG root tags")
	}

	if !strings.Contains(svg, "Semantic Latent Space Manifold") {
		t.Errorf("expected title in SVG")
	}

	if !strings.Contains(svg, CategoryColors[CategoryTruthful]) {
		t.Errorf("expected truthful category color in SVG")
	}

	if !strings.Contains(svg, "trajectory-path") {
		t.Errorf("expected trajectory-path class in SVG")
	}

	// Test light mode as well
	cfg.DarkMode = false
	svgLight, err := GenerateSVG(proj, cfg)
	if err != nil {
		t.Fatalf("failed to generate light mode SVG: %v", err)
	}
	if !strings.Contains(svgLight, "#f8fafc") {
		t.Errorf("expected light background in SVG")
	}
}

func TestExportJSON(t *testing.T) {
	points := []LatentPoint{
		{ID: "p1", ProjectedX: 1.0, ProjectedY: 2.0, Category: CategoryRefusal},
	}
	proj, err := ProjectLatentSpace(points, nil, nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	data, err := ExportJSON(proj)
	if err != nil {
		t.Fatalf("unexpected JSON error: %v", err)
	}

	var parsed LatentSpaceProjection
	if err := json.Unmarshal(data, &parsed); err != nil {
		t.Fatalf("failed to unmarshal exported JSON: %v", err)
	}

	if len(parsed.Points) != 1 {
		t.Errorf("expected 1 point in parsed JSON, got %d", len(parsed.Points))
	}
}

func TestProjectLatentSpace_CacheHitAndEviction(t *testing.T) {
	ClearLatentSVDCache()
	points := []LatentPoint{
		{ID: "t1", Vector: []float64{2.0, 3.0, 0.1, -0.2}, Category: CategoryTruthful, Layer: 0, Token: "The"},
		{ID: "t2", Vector: []float64{2.1, 2.9, 0.2, -0.1}, Category: CategoryTruthful, Layer: 1, Token: "The"},
		{ID: "d1", Vector: []float64{-2.0, -1.0, 0.5, 0.4}, Category: CategoryDeceptive, Layer: 0, Token: "Lie"},
		{ID: "d2", Vector: []float64{-2.2, -0.9, 0.4, 0.6}, Category: CategoryDeceptive, Layer: 1, Token: "Lie"},
	}

	proj1, err := ProjectLatentSpace(points, nil, nil)
	if err != nil {
		t.Fatalf("first ProjectLatentSpace failed: %v", err)
	}
	proj2, err := ProjectLatentSpace(points, nil, nil)
	if err != nil {
		t.Fatalf("second ProjectLatentSpace (cached) failed: %v", err)
	}

	if proj1.VarianceExplained != proj2.VarianceExplained {
		t.Errorf("cached variance mismatch: %v vs %v", proj1.VarianceExplained, proj2.VarianceExplained)
	}
	for i := range proj1.Points {
		if proj1.Points[i].ProjectedX != proj2.Points[i].ProjectedX || proj1.Points[i].ProjectedY != proj2.Points[i].ProjectedY {
			t.Errorf("point %d cached projection mismatch", i)
		}
	}

	// Trigger cache rollover beyond maxSVDCacheEntries
	for k := 0; k < maxSVDCacheEntries+2; k++ {
		pts := []LatentPoint{
			{ID: "a", Vector: []float64{float64(k) + 1.0, 2.0, 3.0}, Category: CategoryTruthful},
			{ID: "b", Vector: []float64{-float64(k) - 1.0, -2.0, -3.0}, Category: CategoryDeceptive},
		}
		if _, err := ProjectLatentSpace(pts, nil, nil); err != nil {
			t.Fatalf("rollover iteration %d failed: %v", k, err)
		}
	}
}

func makeBenchmarkLatentPoints(n, dim int) []LatentPoint {
	pts := make([]LatentPoint, n)
	for i := 0; i < n; i++ {
		vec := make([]float64, dim)
		for j := 0; j < dim; j++ {
			vec[j] = float64((i+1)*(j+1)%37)*0.1 - 1.8
		}
		cat := CategoryTruthful
		if i%2 == 1 {
			cat = CategoryDeceptive
		}
		pts[i] = LatentPoint{
			ID:       "pt",
			PromptID: "p1",
			Token:    "tok",
			Layer:    i % 8,
			Category: cat,
			Vector:   vec,
		}
	}
	return pts
}

func BenchmarkProjectLatentSpacePCA(b *testing.B) {
	pts := makeBenchmarkLatentPoints(64, 128)
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		ClearLatentSVDCache()
		if _, err := ProjectLatentSpace(pts, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProjectLatentSpaceCached(b *testing.B) {
	pts := makeBenchmarkLatentPoints(64, 128)
	ClearLatentSVDCache()
	if _, err := ProjectLatentSpace(pts, nil, nil); err != nil {
		b.Fatal(err)
	}
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ProjectLatentSpace(pts, nil, nil); err != nil {
			b.Fatal(err)
		}
	}
}

func BenchmarkProjectLatentSpaceBasis(b *testing.B) {
	pts := makeBenchmarkLatentPoints(64, 128)
	basisX := make([]float64, 128)
	basisY := make([]float64, 128)
	basisX[0] = 1.0
	basisY[1] = 1.0
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := ProjectLatentSpace(pts, basisX, basisY); err != nil {
			b.Fatal(err)
		}
	}
}
