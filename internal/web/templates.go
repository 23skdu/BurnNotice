package web

import "fmt"

func dashboardHTML(activeModels, totalProfiles int) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>BurnNotice Dashboard</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/ext/sse.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Dashboard</h1>
		<div id="summary" hx-get="/api/summary" hx-trigger="every 5s" hx-swap="innerHTML" class="grid grid-cols-4 gap-4 mb-8">
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Active Models</div>
				<div class="text-3xl font-bold text-red-400">%d</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Total Profiles</div>
				<div class="text-3xl font-bold text-blue-400">%d</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">System Status</div>
				<div class="text-3xl font-bold text-green-400">Healthy</div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Active Sessions</div>
				<div class="text-3xl font-bold text-yellow-400">0</div>
			</div>
		</div>
		<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
			<h2 class="text-lg font-semibold mb-4">Quick Links</h2>
			<div class="grid grid-cols-3 gap-4">
				<a href="/profiler" class="bg-gray-700 rounded p-4 text-center hover:bg-gray-600">Run Profiler</a>
				<a href="/alerts" class="bg-gray-700 rounded p-4 text-center hover:bg-gray-600">View Alerts</a>
				<a href="/policies" class="bg-gray-700 rounded p-4 text-center hover:bg-gray-600">Edit Policies</a>
			</div>
		</div>
	</main>
</body>
</html>`, activeModels, totalProfiles)
}

func profilerHTML() string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Profiler - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/ext/sse.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Profiler Status</h1>
		<div hx-ext="sse" sse-connect="/sse" class="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
			<table class="w-full">
				<thead class="bg-gray-700">
					<tr>
						<th class="px-4 py-2 text-left">Model ID</th>
						<th class="px-4 py-2 text-left">Status</th>
						<th class="px-4 py-2 text-left">Progress</th>
						<th class="px-4 py-2 text-left">Actions</th>
					</tr>
				</thead>
				<tbody id="job-queue">
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2 font-mono">llama-3-8b</td>
						<td class="px-4 py-2"><span class="px-2 py-1 rounded bg-green-900 text-green-300 text-sm">complete</span></td>
						<td class="px-4 py-2">
							<div class="w-full bg-gray-700 rounded h-2"><div class="bg-green-500 h-2 rounded" style="width:100%"></div></div>
						</td>
						<td class="px-4 py-2"><a href="/model/llama-3-8b" class="text-blue-400 hover:underline">View</a></td>
					</tr>
				</tbody>
			</table>
		</div>
	</main>
</body>
</html>`
}

func modelProfileHTML(modelID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Model Profile - %s - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/ext/sse.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Model Profile: <span class="font-mono">%[1]s</span></h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-6">
			<h2 class="text-lg font-semibold mb-4">Extracted Directions</h2>
			<table class="w-full">
				<thead class="bg-gray-700">
					<tr><th class="px-4 py-2 text-left">Direction</th><th class="px-4 py-2 text-left">Layers</th><th class="px-4 py-2 text-left">Visualizations</th></tr>
				</thead>
				<tbody>
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2">Deception</td>
						<td class="px-4 py-2">32</td>
						<td class="px-4 py-2">
							<a href="/model/%[1]s/heatmap" class="text-blue-400 hover:underline mr-3">Heatmap</a>
							<a href="/model/%[1]s/subspace" class="text-blue-400 hover:underline mr-3">Subspace</a>
							<a href="/model/%[1]s/circuits" class="text-blue-400 hover:underline">Circuits</a>
						</td>
					</tr>
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2">Truth</td>
						<td class="px-4 py-2">32</td>
						<td class="px-4 py-2">
							<a href="/model/%[1]s/heatmap" class="text-blue-400 hover:underline mr-3">Heatmap</a>
							<a href="/model/%[1]s/subspace" class="text-blue-400 hover:underline mr-3">Subspace</a>
							<a href="/model/%[1]s/circuits" class="text-blue-400 hover:underline">Circuits</a>
						</td>
					</tr>
				</tbody>
			</table>
		</div>
	</main>
</body>
</html>`, modelID)
}

func monitorHTML(sessionID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Monitor %s - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/ext/sse.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Live Monitor: <span class="font-mono">%s</span></h1>
		<div hx-ext="sse" sse-connect="/sse" class="grid grid-cols-4 gap-4">
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Deception</div>
				<div id="deception-score" class="text-3xl font-bold text-green-400">0.12</div>
				<div class="w-full bg-gray-700 rounded h-2 mt-2"><div class="bg-green-500 h-2 rounded" style="width:12%%"></div></div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Conflict</div>
				<div id="conflict-score" class="text-3xl font-bold text-yellow-400">0.34</div>
				<div class="w-full bg-gray-700 rounded h-2 mt-2"><div class="bg-yellow-500 h-2 rounded" style="width:34%%"></div></div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Truth</div>
				<div id="truth-score" class="text-3xl font-bold text-green-400">0.89</div>
				<div class="w-full bg-gray-700 rounded h-2 mt-2"><div class="bg-green-500 h-2 rounded" style="width:89%%"></div></div>
			</div>
			<div class="bg-gray-800 rounded-lg p-4 border border-gray-700">
				<div class="text-gray-400 text-sm">Compliance</div>
				<div id="compliance-score" class="text-3xl font-bold text-green-400">0.95</div>
				<div class="w-full bg-gray-700 rounded h-2 mt-2"><div class="bg-green-500 h-2 rounded" style="width:95%%"></div></div>
			</div>
		</div>
	</main>
</body>
</html>`, sessionID, sessionID)
}

func heatmapHTML(modelID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Heatmap - %s - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Layer x Token Heatmap: <span class="font-mono">%s</span></h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 p-4 overflow-x-auto">
			<svg width="800" height="400" viewBox="0 0 800 400">
				%s
			</svg>
		</div>
	</main>
</body>
</html>`, modelID, modelID, sampleHeatmapSVG())
}

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

func subspaceHTML(modelID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Subspace - %s - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Subspace Projection: <span class="font-mono">%s</span></h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
			<svg width="600" height="400" viewBox="0 0 600 400">
				<line x1="50" y1="200" x2="550" y2="200" stroke="#666" stroke-width="1"/>
				<line x1="300" y1="50" x2="300" y2="350" stroke="#666" stroke-width="1"/>
				<text x="540" y="190" fill="#999" font-size="12">Truth</text>
				<text x="310" y="60" fill="#999" font-size="12">Deception</text>
				<line x1="50" y1="350" x2="550" y2="50" stroke="#f44" stroke-width="1" stroke-dasharray="4"/>
				%s
			</svg>
		</div>
	</main>
</body>
</html>`, modelID, modelID, sampleScatterPoints())
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

func circuitsHTML(modelID string) string {
	return fmt.Sprintf(`<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Circuits - %s - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Neural Circuits: <span class="font-mono">%s</span></h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 p-4">
			<svg width="800" height="500" viewBox="0 0 800 500">
				%s
			</svg>
		</div>
	</main>
</body>
</html>`, modelID, modelID, sampleCircuitGraph())
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

func policiesHTML() string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Policies - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Policy Configuration</h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 p-4 mb-6">
			<h2 class="text-lg font-semibold mb-4">Add Policy</h2>
			<form hx-post="/api/policies" hx-swap="outerHTML" class="grid grid-cols-5 gap-4">
				<input name="name" placeholder="Policy name" class="bg-gray-700 rounded px-3 py-2 border border-gray-600"/>
				<select name="metric" class="bg-gray-700 rounded px-3 py-2 border border-gray-600">
					<option value="deception_score">Deception Score</option>
					<option value="conflict_score">Conflict Score</option>
					<option value="truth_score">Truth Score</option>
					<option value="compliance">Compliance</option>
				</select>
				<input name="threshold" type="number" step="0.1" min="0" max="1" placeholder="Threshold" class="bg-gray-700 rounded px-3 py-2 border border-gray-600"/>
				<select name="action" class="bg-gray-700 rounded px-3 py-2 border border-gray-600">
					<option value="steer">Steer</option>
					<option value="block">Block</option>
					<option value="log">Log</option>
					<option value="terminate">Terminate</option>
				</select>
				<button type="submit" class="bg-red-600 hover:bg-red-500 rounded px-4 py-2">Add Policy</button>
			</form>
		</div>
		<div class="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
			<table class="w-full">
				<thead class="bg-gray-700">
					<tr><th class="px-4 py-2 text-left">Name</th><th class="px-4 py-2 text-left">Metric</th><th class="px-4 py-2 text-left">Threshold</th><th class="px-4 py-2 text-left">Action</th><th class="px-4 py-2 text-left">Actions</th></tr>
				</thead>
				<tbody>
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2">deception_threshold</td>
						<td class="px-4 py-2">deception_score</td>
						<td class="px-4 py-2">0.7</td>
						<td class="px-4 py-2"><span class="px-2 py-1 rounded bg-blue-900 text-blue-300 text-sm">steer</span></td>
						<td class="px-4 py-2"><button class="text-red-400 hover:underline">Delete</button></td>
					</tr>
				</tbody>
			</table>
		</div>
	</main>
</body>
</html>`
}

func alertsHTML() string {
	return `<!DOCTYPE html>
<html lang="en">
<head>
	<meta charset="UTF-8"/>
	<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
	<title>Alerts - BurnNotice</title>
	<script src="https://unpkg.com/htmx.org@2.0.0/dist/htmx.min.js"></script>
	<script src="https://cdn.tailwindcss.com"></script>
</head>
<body class="bg-gray-900 text-gray-100 min-h-screen">
	<nav class="bg-gray-800 border-b border-gray-700 px-6 py-3">
		<div class="flex items-center justify-between">
			<a href="/" class="text-xl font-bold text-red-400">BurnNotice</a>
			<div class="flex gap-4">
				<a href="/" class="hover:text-red-400">Dashboard</a>
				<a href="/profiler" class="hover:text-red-400">Profiler</a>
				<a href="/alerts" class="hover:text-red-400">Alerts</a>
				<a href="/policies" class="hover:text-red-400">Policies</a>
			</div>
		</div>
	</nav>
	<main class="p-6">
		<h1 class="text-2xl font-bold mb-6">Alert History</h1>
		<div class="bg-gray-800 rounded-lg border border-gray-700 overflow-hidden">
			<table class="w-full">
				<thead class="bg-gray-700">
					<tr><th class="px-4 py-2 text-left">Timestamp</th><th class="px-4 py-2 text-left">Metric</th><th class="px-4 py-2 text-left">Score</th><th class="px-4 py-2 text-left">Threshold</th><th class="px-4 py-2 text-left">Action</th><th class="px-4 py-2 text-left">Session</th></tr>
				</thead>
				<tbody>
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2 font-mono text-sm">2026-05-18T10:30:00Z</td>
						<td class="px-4 py-2">deception_score</td>
						<td class="px-4 py-2 text-red-400">0.85</td>
						<td class="px-4 py-2">0.7</td>
						<td class="px-4 py-2"><span class="px-2 py-1 rounded bg-blue-900 text-blue-300 text-sm">steer</span></td>
						<td class="px-4 py-2 font-mono text-sm">sess-001</td>
					</tr>
					<tr class="border-t border-gray-700">
						<td class="px-4 py-2 font-mono text-sm">2026-05-18T10:31:00Z</td>
						<td class="px-4 py-2">conflict_score</td>
						<td class="px-4 py-2 text-yellow-400">0.72</td>
						<td class="px-4 py-2">0.8</td>
						<td class="px-4 py-2"><span class="px-2 py-1 rounded bg-gray-600 text-gray-300 text-sm">log</span></td>
						<td class="px-4 py-2 font-mono text-sm">sess-002</td>
					</tr>
				</tbody>
			</table>
		</div>
	</main>
</body>
</html>`
}
