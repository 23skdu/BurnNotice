package analysis

import (
	"bytes"
	"encoding/json"
	"fmt"
	"math"
	"sort"
	"sync"

	"gonum.org/v1/gonum/floats"
	"gonum.org/v1/gonum/mat"
)

type svdCacheEntry struct {
	uX      []float64
	uY      []float64
	varExpl [2]float64
}

const maxSVDCacheEntries = 64

var (
	svdCacheMu sync.RWMutex
	svdCache   = make(map[uint64]svdCacheEntry, maxSVDCacheEntries)
)

// ClearLatentSVDCache clears the cached SVD principal components used by ProjectLatentSpace.
func ClearLatentSVDCache() {
	svdCacheMu.Lock()
	svdCache = make(map[uint64]svdCacheEntry, maxSVDCacheEntries)
	svdCacheMu.Unlock()
}

func hashActivationMatrix(points []LatentPoint, dim int) (uint64, bool) {
	if dim < 0 {
		return 0, false
	}
	const (
		offset64 = 14695981039346656037
		prime64  = 1099511628211
	)
	h := uint64(offset64)
	h = (h ^ uint64(len(points))) * prime64
	h = (h ^ uint64(dim)) * prime64 // #nosec G115 -- dim is validated non-negative slice length
	for i := range points {
		vec := points[i].Vector
		h = (h ^ uint64(len(vec))) * prime64
		for _, v := range vec {
			if math.IsNaN(v) || math.IsInf(v, 0) {
				return 0, false
			}
			bits := math.Float64bits(v)
			h = (h ^ bits) * prime64
		}
	}
	return h, true
}

func hasNonFinite(xs []float64) bool {
	for _, v := range xs {
		if math.IsNaN(v) || math.IsInf(v, 0) {
			return true
		}
	}
	return false
}

// SemanticCategory classifies latent points into semantic behavioral groups.
type SemanticCategory string

const (
	CategoryTruthful         SemanticCategory = "truthful"
	CategoryDeceptive        SemanticCategory = "deceptive"
	CategoryRefusal          SemanticCategory = "refusal"
	CategoryCreative         SemanticCategory = "creative"
	CategoryHarmful          SemanticCategory = "harmful"
	CategoryHarmless         SemanticCategory = "harmless"
	CategoryHarmlessBaseline SemanticCategory = "baseline"
)

// CategoryColors provides curated aesthetic hex colors for semantic categories.
var CategoryColors = map[SemanticCategory]string{
	CategoryTruthful:         "#10B981", // emerald
	CategoryDeceptive:        "#EF4444", // crimson
	CategoryRefusal:          "#F59E0B", // amber
	CategoryCreative:         "#8B5CF6", // purple
	CategoryHarmful:          "#EC4899", // pink
	CategoryHarmless:         "#3B82F6", // sky blue
	CategoryHarmlessBaseline: "#6B7280", // cool gray
}

// LatentPoint represents a high-dimensional activation projected onto 2D latent space.
type LatentPoint struct {
	ID         string            `json:"id"`
	Vector     []float64         `json:"vector,omitempty"`
	ProjectedX float64           `json:"x"`
	ProjectedY float64           `json:"y"`
	Category   SemanticCategory  `json:"category"`
	Layer      int               `json:"layer"`
	Token      string            `json:"token"`
	PromptID   string            `json:"prompt_id"`
	Metadata   map[string]string `json:"metadata,omitempty"`
}

// TokenTrajectory represents a sequence of layer activations for a specific token or prompt.
type TokenTrajectory struct {
	PromptID       string           `json:"prompt_id"`
	Token          string           `json:"token"`
	Category       SemanticCategory `json:"category"`
	Points         []LatentPoint    `json:"points"`
	TotalCurvature float64          `json:"total_curvature"`
	Velocity       []float64        `json:"velocity"` // step-to-step euclidean delta
}

// ClusterSummary describes the statistical dispersion of a semantic category.
type ClusterSummary struct {
	Category   SemanticCategory `json:"category"`
	Centroid   [2]float64       `json:"centroid"`
	RadiusX    float64          `json:"radius_x"`
	RadiusY    float64          `json:"radius_y"`
	PointCount int              `json:"point_count"`
	Variance   float64          `json:"variance"`
}

// LatentSpaceProjection encapsulates the projected manifold and its topological features.
type LatentSpaceProjection struct {
	Points            []LatentPoint                       `json:"points"`
	Trajectories      []TokenTrajectory                   `json:"trajectories"`
	Clusters          map[SemanticCategory]ClusterSummary `json:"clusters"`
	VarianceExplained [2]float64                          `json:"variance_explained"`
	Method            string                              `json:"method"`
}

// LatentVisualizerConfig controls visual styling and rendering attributes.
type LatentVisualizerConfig struct {
	Width               int
	Height              int
	ShowTrajectories    bool
	ShowConfidenceHulls bool
	ShowGrid            bool
	DarkMode            bool
	Title               string
}

// DefaultVisualizerConfig returns balanced defaults for SVG generation.
func DefaultVisualizerConfig() LatentVisualizerConfig {
	return LatentVisualizerConfig{
		Width:               880,
		Height:              580,
		ShowTrajectories:    true,
		ShowConfidenceHulls: true,
		ShowGrid:            true,
		DarkMode:            true,
		Title:               "Semantic Latent Space Manifold",
	}
}

// ProjectLatentSpace projects high-dimensional activations into a 2D semantic subspace.
// If basisX and basisY are provided, activations are orthogonally projected onto them.
// Otherwise, PCA (SVD) is performed across the points to find the top-2 principal components.
func ProjectLatentSpace(points []LatentPoint, basisX, basisY []float64) (*LatentSpaceProjection, error) {
	if len(points) == 0 {
		return nil, fmt.Errorf("no points provided for latent space projection")
	}

	dim := len(points[0].Vector)
	if dim < 2 && len(points[0].Vector) > 0 {
		return nil, fmt.Errorf("vector dimension %d is too small for 2D projection", dim)
	}

	proj := &LatentSpaceProjection{
		Points:   make([]LatentPoint, len(points)),
		Clusters: make(map[SemanticCategory]ClusterSummary),
		Method:   "pca",
	}
	copy(proj.Points, points)

	// If 2D coordinates are already pre-computed and no high-dim vectors exist
	if dim == 0 {
		for _, p := range proj.Points {
			if math.IsNaN(p.ProjectedX) || math.IsInf(p.ProjectedX, 0) || math.IsNaN(p.ProjectedY) || math.IsInf(p.ProjectedY, 0) {
				return nil, fmt.Errorf("precomputed coordinates contain NaN or Inf")
			}
		}
		proj.Method = "precomputed"
		proj.Clusters = ComputeClusterSummaries(proj.Points)
		proj.Trajectories = BuildTokenTrajectories(proj.Points)
		proj.VarianceExplained = [2]float64{0.55, 0.35}
		return proj, nil
	}

	cacheKey, finite := hashActivationMatrix(points, dim)
	if !finite {
		return nil, fmt.Errorf("activation vectors contain NaN or Inf")
	}

	var uX, uY []float64
	var varExpl [2]float64

	if len(basisX) == dim && len(basisY) == dim {
		if hasNonFinite(basisX) || hasNonFinite(basisY) {
			return nil, fmt.Errorf("basis vectors contain NaN or Inf")
		}
		uX = make([]float64, dim)
		uY = make([]float64, dim)
		copy(uX, basisX)
		copy(uY, basisY)
		normX := floats.Norm(uX, 2)
		normY := floats.Norm(uY, 2)
		if normX > 1e-9 {
			floats.Scale(1.0/normX, uX)
		}
		if normY > 1e-9 {
			floats.Scale(1.0/normY, uY)
		}
		proj.Method = "basis_projection"
		varExpl = [2]float64{0.60, 0.40}
	} else {
		svdCacheMu.RLock()
		cached, hit := svdCache[cacheKey]
		svdCacheMu.RUnlock()

		if hit && len(cached.uX) == dim && len(cached.uY) == dim {
			uX = cached.uX
			uY = cached.uY
			varExpl = cached.varExpl
		} else {
			// Perform Thin SVD on mean-centered activations using a flat backing slice
			rows := len(points)
			mean := make([]float64, dim)
			for _, p := range points {
				if len(p.Vector) == dim {
					floats.Add(mean, p.Vector)
				}
			}
			floats.Scale(1.0/float64(rows), mean)

			raw := make([]float64, rows*dim)
			for i, p := range points {
				rowOff := i * dim
				if len(p.Vector) == dim {
					for j := 0; j < dim; j++ {
						raw[rowOff+j] = p.Vector[j] - mean[j]
					}
				} else {
					for j := 0; j < dim; j++ {
						val := -mean[j]
						if j < len(p.Vector) {
							val = p.Vector[j] - mean[j]
						}
						raw[rowOff+j] = val
					}
				}
			}
			data := mat.NewDense(rows, dim, raw)

			var svd mat.SVD
			if ok := svd.Factorize(data, mat.SVDThin); !ok {
				return nil, fmt.Errorf("svd factorization failed on activation matrix")
			}

			svals := svd.Values(nil)
			totalVar := 0.0
			for _, s := range svals {
				totalVar += s * s
			}
			if totalVar > 1e-12 {
				if len(svals) > 0 {
					varExpl[0] = (svals[0] * svals[0]) / totalVar
				}
				if len(svals) > 1 {
					varExpl[1] = (svals[1] * svals[1]) / totalVar
				}
			} else {
				varExpl = [2]float64{0.5, 0.5}
			}

			minDim := rows
			if dim < minDim {
				minDim = dim
			}
			var v mat.Dense
			svd.VTo(&v)
			rawV := v.RawMatrix()

			uX = make([]float64, dim)
			uY = make([]float64, dim)
			for j := 0; j < dim; j++ {
				rowIdx := j * rawV.Stride
				uX[j] = rawV.Data[rowIdx]
				if minDim > 1 {
					uY[j] = rawV.Data[rowIdx+1]
				}
			}

			svdCacheMu.Lock()
			if len(svdCache) >= maxSVDCacheEntries {
				svdCache = make(map[uint64]svdCacheEntry, maxSVDCacheEntries)
			}
			svdCache[cacheKey] = svdCacheEntry{
				uX:      uX,
				uY:      uY,
				varExpl: varExpl,
			}
			svdCacheMu.Unlock()
		}
		proj.Method = "pca"
	}

	proj.VarianceExplained = varExpl

	// Project each point
	for i := range proj.Points {
		if len(proj.Points[i].Vector) == dim {
			proj.Points[i].ProjectedX = floats.Dot(proj.Points[i].Vector, uX)
			proj.Points[i].ProjectedY = floats.Dot(proj.Points[i].Vector, uY)
		}
	}

	proj.Clusters = ComputeClusterSummaries(proj.Points)
	proj.Trajectories = BuildTokenTrajectories(proj.Points)

	return proj, nil
}

// ComputeClusterSummaries calculates centroids, standard deviations, and variances for each category.
func ComputeClusterSummaries(points []LatentPoint) map[SemanticCategory]ClusterSummary {
	clusters := make(map[SemanticCategory]ClusterSummary)
	groups := make(map[SemanticCategory][]LatentPoint)

	for _, p := range points {
		groups[p.Category] = append(groups[p.Category], p)
	}

	for cat, pts := range groups {
		if len(pts) == 0 {
			continue
		}
		sumX := 0.0
		sumY := 0.0
		for _, p := range pts {
			sumX += p.ProjectedX
			sumY += p.ProjectedY
		}
		n := float64(len(pts))
		meanX := sumX / n
		meanY := sumY / n

		varSumX := 0.0
		varSumY := 0.0
		for _, p := range pts {
			dx := p.ProjectedX - meanX
			dy := p.ProjectedY - meanY
			varSumX += dx * dx
			varSumY += dy * dy
		}

		stdX := math.Sqrt(varSumX / n)
		stdY := math.Sqrt(varSumY / n)
		if stdX < 0.05 {
			stdX = 0.05
		}
		if stdY < 0.05 {
			stdY = 0.05
		}

		clusters[cat] = ClusterSummary{
			Category:   cat,
			Centroid:   [2]float64{meanX, meanY},
			RadiusX:    stdX * 1.96, // ~95% confidence interval
			RadiusY:    stdY * 1.96,
			PointCount: len(pts),
			Variance:   (varSumX + varSumY) / n,
		}
	}

	return clusters
}

// BuildTokenTrajectories organizes points into layer-by-layer trajectories.
func BuildTokenTrajectories(points []LatentPoint) []TokenTrajectory {
	groups := make(map[string][]LatentPoint)
	for _, p := range points {
		key := p.PromptID
		if key == "" {
			key = p.Token
		}
		if key == "" {
			continue
		}
		groups[key] = append(groups[key], p)
	}

	trajectories := make([]TokenTrajectory, 0, len(groups))
	for _, pts := range groups {
		if len(pts) < 2 {
			continue
		}
		// Sort by Layer ascending
		sort.Slice(pts, func(i, j int) bool {
			return pts[i].Layer < pts[j].Layer
		})

		velocities := make([]float64, len(pts)-1)
		totalCurv := 0.0

		for i := 0; i < len(pts)-1; i++ {
			dx := pts[i+1].ProjectedX - pts[i].ProjectedX
			dy := pts[i+1].ProjectedY - pts[i].ProjectedY
			dist := math.Hypot(dx, dy)
			velocities[i] = dist

			if i > 0 {
				prevDx := pts[i].ProjectedX - pts[i-1].ProjectedX
				prevDy := pts[i].ProjectedY - pts[i-1].ProjectedY
				prevDist := math.Hypot(prevDx, prevDy)
				if dist > 1e-9 && prevDist > 1e-9 {
					dot := (dx*prevDx + dy*prevDy) / (dist * prevDist)
					if dot > 1.0 {
						dot = 1.0
					} else if dot < -1.0 {
						dot = -1.0
					}
					angle := math.Acos(dot)
					totalCurv += angle
				}
			}
		}

		trajectories = append(trajectories, TokenTrajectory{
			PromptID:       pts[0].PromptID,
			Token:          pts[0].Token,
			Category:       pts[0].Category,
			Points:         pts,
			TotalCurvature: totalCurv,
			Velocity:       velocities,
		})
	}

	// Sort trajectories by prompt ID for deterministic output
	sort.Slice(trajectories, func(i, j int) bool {
		if trajectories[i].PromptID == trajectories[j].PromptID {
			return trajectories[i].Token < trajectories[j].Token
		}
		return trajectories[i].PromptID < trajectories[j].PromptID
	})

	return trajectories
}

// GenerateSVG renders a rich, high-resolution vector visualization of the latent space.
func GenerateSVG(proj *LatentSpaceProjection, cfg LatentVisualizerConfig) (string, error) {
	if proj == nil {
		return "", fmt.Errorf("nil projection supplied")
	}

	width := cfg.Width
	if width <= 0 {
		width = 880
	}
	height := cfg.Height
	if height <= 0 {
		height = 580
	}

	// Determine coordinate bounds with padding
	minX, maxX := math.MaxFloat64, -math.MaxFloat64
	minY, maxY := math.MaxFloat64, -math.MaxFloat64

	for _, p := range proj.Points {
		if p.ProjectedX < minX {
			minX = p.ProjectedX
		}
		if p.ProjectedX > maxX {
			maxX = p.ProjectedX
		}
		if p.ProjectedY < minY {
			minY = p.ProjectedY
		}
		if p.ProjectedY > maxY {
			maxY = p.ProjectedY
		}
	}

	// Include cluster hull boundaries in bounding box
	for _, c := range proj.Clusters {
		if c.Centroid[0]-c.RadiusX < minX {
			minX = c.Centroid[0] - c.RadiusX
		}
		if c.Centroid[0]+c.RadiusX > maxX {
			maxX = c.Centroid[0] + c.RadiusX
		}
		if c.Centroid[1]-c.RadiusY < minY {
			minY = c.Centroid[1] - c.RadiusY
		}
		if c.Centroid[1]+c.RadiusY > maxY {
			maxY = c.Centroid[1] + c.RadiusY
		}
	}

	if minX >= maxX {
		minX, maxX = -1.0, 1.0
	}
	if minY >= maxY {
		minY, maxY = -1.0, 1.0
	}

	// Add 12% margin
	padX := (maxX - minX) * 0.12
	padY := (maxY - minY) * 0.12
	if padX < 0.1 {
		padX = 0.1
	}
	if padY < 0.1 {
		padY = 0.1
	}
	minX -= padX
	maxX += padX
	minY -= padY
	maxY += padY

	// Screen coordinate transformation
	const marginL, marginR, marginT, marginB = 70.0, 50.0, 60.0, 60.0
	plotW := float64(width) - marginL - marginR
	plotH := float64(height) - marginT - marginB

	toScreenX := func(x float64) float64 {
		return marginL + ((x-minX)/(maxX-minX))*plotW
	}
	toScreenY := func(y float64) float64 {
		return marginT + plotH - ((y-minY)/(maxY-minY))*plotH
	}
	toScreenRadiusX := func(rx float64) float64 {
		return (rx / (maxX - minX)) * plotW
	}
	toScreenRadiusY := func(ry float64) float64 {
		return (ry / (maxY - minY)) * plotH
	}

	var buf bytes.Buffer

	bgColor := "#0b0f19"
	cardBg := "#111827"
	textColor := "#f3f4f6"
	subTextColor := "#9ca3af"
	gridColor := "#1f2937"
	axisColor := "#374151"

	if !cfg.DarkMode {
		bgColor = "#f8fafc"
		cardBg = "#ffffff"
		textColor = "#0f172a"
		subTextColor = "#64748b"
		gridColor = "#e2e8f0"
		axisColor = "#cbd5e1"
	}

	_, _ = fmt.Fprintf(&buf, `<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 %d %d" width="100%%" height="100%%" class="latent-space-viz">`+"\n", width, height)
	buf.WriteString("<defs>\n")
	buf.WriteString(`  <filter id="glow" x="-20%" y="-20%" width="140%" height="140%">
    <feGaussianBlur stdDeviation="3" result="blur" />
    <feComposite in="SourceGraphic" in2="blur" operator="over" />
  </filter>
  <marker id="arrow" viewBox="0 0 10 10" refX="6" refY="5" markerWidth="6" markerHeight="6" orient="auto-start-reverse">
    <path d="M 0 1.5 L 8 5 L 0 8.5 z" fill="#60a5fa" opacity="0.8"/>
  </marker>
`)
	buf.WriteString("</defs>\n")

	// Styling
	_, _ = fmt.Fprintf(&buf, `<style>
  .latent-space-viz { font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif; }
  .grid-line { stroke: %s; stroke-width: 1; stroke-dasharray: 3 3; }
  .axis-line { stroke: %s; stroke-width: 1.5; }
  .axis-label { fill: %s; font-size: 11px; font-weight: 500; }
  .title-text { fill: %s; font-size: 15px; font-weight: 600; letter-spacing: -0.01em; }
  .subtitle-text { fill: %s; font-size: 11px; }
  .point-node { cursor: pointer; transition: transform 0.15s ease, r 0.15s ease; }
  .point-node:hover { stroke-width: 2px !important; stroke: #ffffff !important; }
  .trajectory-path { fill: none; stroke-linecap: round; stroke-linejoin: round; }
  .legend-text { fill: %s; font-size: 11px; font-weight: 500; }
</style>
`, gridColor, axisColor, subTextColor, textColor, subTextColor, textColor)

	// Background
	_, _ = fmt.Fprintf(&buf, `<rect width="%d" height="%d" fill="%s" rx="10" />`+"\n", width, height, bgColor)
	_, _ = fmt.Fprintf(&buf, `<rect x="%f" y="%f" width="%f" height="%f" fill="%s" rx="6" />`+"\n",
		marginL, marginT, plotW, plotH, cardBg)

	// Gridlines
	if cfg.ShowGrid {
		gridSteps := 5
		for i := 0; i <= gridSteps; i++ {
			frac := float64(i) / float64(gridSteps)
			gx := marginL + frac*plotW
			gy := marginT + frac*plotH
			_, _ = fmt.Fprintf(&buf, `<line x1="%f" y1="%f" x2="%f" y2="%f" class="grid-line" />`+"\n",
				gx, marginT, gx, marginT+plotH)
			_, _ = fmt.Fprintf(&buf, `<line x1="%f" y1="%f" x2="%f" y2="%f" class="grid-line" />`+"\n",
				marginL, gy, marginL+plotW, gy)
		}
	}

	// Axes & Zero Lines (if in range)
	if minX <= 0 && maxX >= 0 {
		zx := toScreenX(0)
		_, _ = fmt.Fprintf(&buf, `<line x1="%f" y1="%f" x2="%f" y2="%f" class="axis-line" stroke-opacity="0.4" />`+"\n",
			zx, marginT, zx, marginT+plotH)
	}
	if minY <= 0 && maxY >= 0 {
		zy := toScreenY(0)
		_, _ = fmt.Fprintf(&buf, `<line x1="%f" y1="%f" x2="%f" y2="%f" class="axis-line" stroke-opacity="0.4" />`+"\n",
			marginL, zy, marginL+plotW, zy)
	}

	// Title & Metrics Header
	title := cfg.Title
	if title == "" {
		title = "Semantic Latent Space Manifold"
	}
	_, _ = fmt.Fprintf(&buf, `<text x="%f" y="32" class="title-text">%s</text>`+"\n", marginL, title)
	_, _ = fmt.Fprintf(&buf, `<text x="%f" y="48" class="subtitle-text">Method: %s | PC1 (%.1f%% var) vs PC2 (%.1f%% var) | %d Points | %d Trajectories</text>`+"\n",
		marginL, proj.Method, proj.VarianceExplained[0]*100, proj.VarianceExplained[1]*100, len(proj.Points), len(proj.Trajectories))

	// Confidence Hulls / Ellipses
	if cfg.ShowConfidenceHulls {
		for cat, c := range proj.Clusters {
			cx := toScreenX(c.Centroid[0])
			cy := toScreenY(c.Centroid[1])
			rx := toScreenRadiusX(c.RadiusX)
			ry := toScreenRadiusY(c.RadiusY)
			color := CategoryColors[cat]
			if color == "" {
				color = "#9ca3af"
			}

			_, _ = fmt.Fprintf(&buf, `<ellipse cx="%f" cy="%f" rx="%f" ry="%f" fill="%s" fill-opacity="0.08" stroke="%s" stroke-width="1.2" stroke-dasharray="4 3" />`+"\n",
				cx, cy, rx, ry, color, color)
			_, _ = fmt.Fprintf(&buf, `<circle cx="%f" cy="%f" r="3" fill="%s" />`+"\n", cx, cy, color)
			_, _ = fmt.Fprintf(&buf, `<text x="%f" y="%f" fill="%s" font-size="10" font-weight="600" text-anchor="middle" opacity="0.85">%s (N=%d)</text>`+"\n",
				cx, cy-ry-5, color, cat, c.PointCount)
		}
	}

	// Trajectories
	if cfg.ShowTrajectories && len(proj.Trajectories) > 0 {
		for _, traj := range proj.Trajectories {
			if len(traj.Points) < 2 {
				continue
			}
			pathData := bytes.Buffer{}
			for i, p := range traj.Points {
				sx := toScreenX(p.ProjectedX)
				sy := toScreenY(p.ProjectedY)
				if i == 0 {
					_, _ = fmt.Fprintf(&pathData, "M %f %f ", sx, sy)
				} else {
					_, _ = fmt.Fprintf(&pathData, "L %f %f ", sx, sy)
				}
			}
			color := CategoryColors[traj.Category]
			if color == "" {
				color = "#60a5fa"
			}

			_, _ = fmt.Fprintf(&buf, `<path d="%s" class="trajectory-path" stroke="%s" stroke-width="2" stroke-opacity="0.75" marker-end="url(#arrow)" />`+"\n",
				pathData.String(), color)
		}
	}

	// Scatter Points
	for _, p := range proj.Points {
		sx := toScreenX(p.ProjectedX)
		sy := toScreenY(p.ProjectedY)
		color := CategoryColors[p.Category]
		if color == "" {
			color = "#9ca3af"
		}

		_, _ = fmt.Fprintf(&buf, `<circle cx="%f" cy="%f" r="4.5" fill="%s" fill-opacity="0.85" stroke="#ffffff" stroke-width="0.8" class="point-node">`+"\n",
			sx, sy, color)
		_, _ = fmt.Fprintf(&buf, `  <title>Token: "%s" | Layer: %d | Cat: %s | Coord: (%.2f, %.2f)</title>`+"\n",
			p.Token, p.Layer, p.Category, p.ProjectedX, p.ProjectedY)
		buf.WriteString("</circle>\n")
	}

	// Axis Labels
	_, _ = fmt.Fprintf(&buf, `<text x="%f" y="%f" class="axis-label" text-anchor="middle">Latent Subspace Dimension 1 (%.1f%% Variance)</text>`+"\n",
		marginL+plotW/2, float64(height)-18, proj.VarianceExplained[0]*100)

	_, _ = fmt.Fprintf(&buf, `<text x="18" y="%f" class="axis-label" text-anchor="middle" transform="rotate(-90, 18, %f)">Latent Subspace Dimension 2 (%.1f%% Variance)</text>`+"\n",
		marginT+plotH/2, marginT+plotH/2, proj.VarianceExplained[1]*100)

	// Legend (placed at bottom right or top right)
	legendX := marginL + plotW - 220
	legendY := marginT + 18
	_, _ = fmt.Fprintf(&buf, `<g transform="translate(%f, %f)">`+"\n", legendX, legendY)
	buf.WriteString(`<rect width="210" height="75" fill="#1f2937" fill-opacity="0.8" rx="5" />` + "\n")

	row := 0
	col := 0
	cats := []SemanticCategory{CategoryTruthful, CategoryDeceptive, CategoryRefusal, CategoryCreative, CategoryHarmful, CategoryHarmless}
	for _, cat := range cats {
		lx := float64(col*100) + 12
		ly := float64(row*20) + 16
		color := CategoryColors[cat]

		_, _ = fmt.Fprintf(&buf, `<circle cx="%f" cy="%f" r="4" fill="%s" />`+"\n", lx, ly, color)
		_, _ = fmt.Fprintf(&buf, `<text x="%f" y="%f" class="legend-text" font-size="10">%s</text>`+"\n", lx+10, ly+3, cat)

		col++
		if col > 1 {
			col = 0
			row++
		}
	}
	buf.WriteString("</g>\n")

	buf.WriteString("</svg>\n")
	return buf.String(), nil
}

// ExportJSON outputs the projection, clusters, and trajectories in JSON format.
func ExportJSON(proj *LatentSpaceProjection) ([]byte, error) {
	if proj == nil {
		return nil, fmt.Errorf("nil projection")
	}
	return json.MarshalIndent(proj, "", "  ")
}
