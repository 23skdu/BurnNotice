package web

import (
	"fmt"
	"math"

	"burnnotice/internal/analysis"
)

func sampleHeatmapSVG() string {
	svg := ""
	layers := 32
	tokens := 20
	cellW := 700.0 / float64(tokens)
	cellH := 350.0 / float64(layers)
	for l := 0; l < layers; l++ {
		for t := 0; t < tokens; t++ {
			score := float64(l*t) / float64(layers*tokens)
			r := int(score * 255)
			g := int((1 - score) * 200)
			b := 50
			svg += fmt.Sprintf(`<rect x="%.1f" y="%.1f" width="%.1f" height="%.1f" fill="rgb(%d,%d,%d)" rx="1"/>`,
				float64(t)*cellW+50, float64(l)*cellH+25, cellW-1, cellH-1, r, g, b)
		}
	}
	return svg
}

func sampleScatterPoints() string {
	points := ""
	for i := 0; i < 50; i++ {
		x := 50 + float64(i)*10
		y := 50 + float64(i)*6
		color := "#4ade80"
		if i%3 == 0 {
			color = "#f87171"
		}
		points += fmt.Sprintf(`<circle cx="%.1f" cy="%.1f" r="4" fill="%s" opacity="0.7"/>`, x, y, color)
	}
	return points
}

func sampleCircuitGraph() string {
	svg := ""
	nodes := 12
	for i := 0; i < nodes; i++ {
		x := 100 + float64(i%4)*200
		y := 100 + float64(i/4)*150
		color := "#4ade80"
		if i%3 == 0 {
			color = "#f87171"
		}
		svg += fmt.Sprintf(`<circle cx="%.1f" cy="%.1f" r="20" fill="%s" opacity="0.8"/>`, x, y, color)
	}
	for i := 0; i < nodes-1; i++ {
		x1 := 100 + float64(i%4)*200
		y1 := 100 + float64(i/4)*150
		x2 := 100 + float64((i+1)%4)*200
		y2 := 100 + float64((i+1)/4)*150
		svg += fmt.Sprintf(`<line x1="%.1f" y1="%.1f" x2="%.1f" y2="%.1f" stroke="#666" stroke-width="2"/>`, x1, y1, x2, y2)
	}
	return svg
}

// sampleLatentSpaceSVG generates a fine-grained 2D manifold visualization of the model's semantic latent space.
func sampleLatentSpaceSVG(modelID string) string {
	points := make([]analysis.LatentPoint, 0, 64)

	// Truthful cluster & trajectory across 8 layers
	for l := 0; l < 8; l++ {
		fl := float64(l)
		points = append(points, analysis.LatentPoint{
			ID:         fmt.Sprintf("t-%d", l),
			PromptID:   "prompt-truthful-1",
			Token:      "verified",
			Layer:      l,
			Category:   analysis.CategoryTruthful,
			ProjectedX: 2.5 + 0.15*fl + 0.05*math.Sin(fl),
			ProjectedY: 1.8 + 0.20*fl + 0.04*math.Cos(fl),
		})
	}

	// Deceptive cluster & trajectory across 8 layers
	for l := 0; l < 8; l++ {
		fl := float64(l)
		points = append(points, analysis.LatentPoint{
			ID:         fmt.Sprintf("d-%d", l),
			PromptID:   "prompt-deceptive-1",
			Token:      "misleading",
			Layer:      l,
			Category:   analysis.CategoryDeceptive,
			ProjectedX: -2.2 - 0.18*fl + 0.06*math.Cos(fl),
			ProjectedY: -1.5 - 0.22*fl + 0.05*math.Sin(fl),
		})
	}

	// Refusal cluster
	for i := 0; i < 10; i++ {
		fi := float64(i)
		points = append(points, analysis.LatentPoint{
			ID:         fmt.Sprintf("r-%d", i),
			PromptID:   "prompt-refusal",
			Token:      "cannot",
			Layer:      16,
			Category:   analysis.CategoryRefusal,
			ProjectedX: -0.5 + 0.3*math.Sin(fi*1.2),
			ProjectedY: 2.5 + 0.25*math.Cos(fi*1.2),
		})
	}

	// Creative cluster
	for i := 0; i < 8; i++ {
		fi := float64(i)
		points = append(points, analysis.LatentPoint{
			ID:         fmt.Sprintf("c-%d", i),
			PromptID:   "prompt-creative",
			Token:      "imagine",
			Layer:      12,
			Category:   analysis.CategoryCreative,
			ProjectedX: 0.8 + 0.4*math.Cos(fi*0.9),
			ProjectedY: -2.0 + 0.3*math.Sin(fi*0.9),
		})
	}

	// Harmful cluster
	for i := 0; i < 8; i++ {
		fi := float64(i)
		points = append(points, analysis.LatentPoint{
			ID:         fmt.Sprintf("h-%d", i),
			PromptID:   "prompt-harmful",
			Token:      "exploit",
			Layer:      20,
			Category:   analysis.CategoryHarmful,
			ProjectedX: -3.5 + 0.3*math.Sin(fi*0.8),
			ProjectedY: 0.5 + 0.35*math.Cos(fi*0.8),
		})
	}

	proj, err := analysis.ProjectLatentSpace(points, nil, nil)
	if err != nil {
		return "<div class='text-red-400'>Failed to project latent space</div>"
	}

	cfg := analysis.DefaultVisualizerConfig()
	cfg.Title = fmt.Sprintf("Semantic Latent Space: %s", modelID)
	cfg.Width = 840
	cfg.Height = 520

	svg, err := analysis.GenerateSVG(proj, cfg)
	if err != nil {
		return fmt.Sprintf("<div class='text-red-400'>Error generating SVG: %v</div>", err)
	}

	return svg
}
