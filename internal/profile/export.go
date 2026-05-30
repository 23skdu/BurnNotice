package profile

import (
	"encoding/gob"
	"encoding/json"
	"fmt"
	"time"
)

// ProfileMetadata holds metadata about a profile.
type Metadata struct {
	CreatedAt         time.Time `json:"created_at"`
	ModelArchitecture string    `json:"model_architecture"`
	DatasetSize       int       `json:"dataset_size"`
	ValidationScore   float64   `json:"validation_score"`
	Version           int       `json:"version"`
}

// ExportableProfile is the exportable version of ModelProfile with metadata.
type ExportableProfile struct {
	ModelID         string                       `json:"model_id"`
	Directions      map[string]map[int][]float64 `json:"directions"`
	LayerScores     map[string][]float64         `json:"layer_scores"`
	TruthNeurons    []int                        `json:"truth_neurons"`
	MonitoredLayers []int                        `json:"monitored_layers"`
	Metadata        Metadata                     `json:"metadata"`
}

// ToExportable converts a ModelProfile to an exportable format.
func (p *ModelProfile) ToExportable(meta Metadata) *ExportableProfile {
	return &ExportableProfile{
		ModelID:         p.ModelID,
		Directions:      p.Directions,
		LayerScores:     p.LayerScores,
		TruthNeurons:    p.TruthNeurons,
		MonitoredLayers: p.MonitoredLayers,
		Metadata:        meta,
	}
}

// FromExportable converts an ExportableProfile back to a ModelProfile.
func (ep *ExportableProfile) FromExportable() *ModelProfile {
	return &ModelProfile{
		ModelID:         ep.ModelID,
		Directions:      ep.Directions,
		LayerScores:     ep.LayerScores,
		TruthNeurons:    ep.TruthNeurons,
		MonitoredLayers: ep.MonitoredLayers,
	}
}

// Export serializes a profile to the given format.
func (p *ModelProfile) Export(meta Metadata, format string) ([]byte, error) {
	ep := p.ToExportable(meta)

	switch format {
	case "json":
		return json.Marshal(ep)
	case "gob":
		var buf []byte
		enc := gob.NewEncoder(nilWriter{&buf})
		if err := enc.Encode(ep); err != nil {
			return nil, fmt.Errorf("gob encode: %w", err)
		}
		return buf, nil
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}
}

// ImportProfile deserializes a profile from the given format.
func ImportProfile(data []byte, format string) (*ExportableProfile, error) {
	var ep ExportableProfile

	switch format {
	case "json":
		if err := json.Unmarshal(data, &ep); err != nil {
			return nil, fmt.Errorf("json decode: %w", err)
		}
	case "gob":
		r := &nilReader{data: data}
		dec := gob.NewDecoder(r)
		if err := dec.Decode(&ep); err != nil {
			return nil, fmt.Errorf("gob decode: %w", err)
		}
	default:
		return nil, fmt.Errorf("unsupported format: %s", format)
	}

	return &ep, nil
}

// nilWriter wraps a byte slice pointer for gob encoding.
type nilWriter struct {
	buf *[]byte
}

func (w nilWriter) Write(p []byte) (int, error) {
	*w.buf = append(*w.buf, p...)
	return len(p), nil
}

// nilReader wraps a byte slice for gob decoding.
type nilReader struct {
	data []byte
	pos  int
}

func (r *nilReader) Read(p []byte) (int, error) {
	if r.pos >= len(r.data) {
		return 0, fmt.Errorf("EOF")
	}
	n := copy(p, r.data[r.pos:])
	r.pos += n
	return n, nil
}

// ComputeDiff computes the cosine distance between direction vectors of two profiles.
func ComputeDiff(old, new *ExportableProfile) map[string]map[int]float64 {
	diff := make(map[string]map[int]float64)

	for name, oldDirs := range old.Directions {
		newDirs, ok := new.Directions[name]
		if !ok {
			continue
		}

		diff[name] = make(map[int]float64)
		for layerIdx, oldVec := range oldDirs {
			newVec, ok := newDirs[layerIdx]
			if !ok {
				continue
			}

			diff[name][layerIdx] = cosineDistance(oldVec, newVec)
		}
	}

	return diff
}

// DetectDrift flags significant direction changes (> threshold cosine distance).
func DetectDrift(diff map[string]map[int]float64, threshold float64) []DriftAlert {
	var alerts []DriftAlert

	for name, layers := range diff {
		for layerIdx, distance := range layers {
			if distance > threshold {
				alerts = append(alerts, DriftAlert{
					DirectionName: name,
					LayerIndex:    layerIdx,
					Distance:      distance,
				})
			}
		}
	}

	return alerts
}

// DriftAlert represents a significant direction change.
type DriftAlert struct {
	DirectionName string
	LayerIndex    int
	Distance      float64
}

func cosineDistance(a, b []float64) float64 {
	if len(a) != len(b) {
		return 1.0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	if normA == 0 || normB == 0 {
		return 1.0
	}

	similarity := dot / (sqrt(normA) * sqrt(normB))
	return 1.0 - similarity
}

func sqrt(x float64) float64 {
	if x <= 0 {
		return 0
	}
	z := x
	for i := 0; i < 10; i++ {
		z = (z + x/z) / 2
	}
	return z
}
