package web

import "fmt"

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
