package profile

import (
	"encoding/gob"
	"fmt"
	"os"
	"path/filepath"
)

// ModelProfile stores extracted behavioral directions for a specific model.
type ModelProfile struct {
	ModelID         string
	Directions      map[string]map[int][]float64 // direction_name -> layer_idx -> vector
	LayerScores     map[string][]float64         // direction_name -> layer scores
	TruthNeurons    []int
	MonitoredLayers []int
}

// Save persists the profile to disk using gob serialization.
func (p *ModelProfile) Save(dir string) error {
	if err := os.MkdirAll(dir, 0750); err != nil {
		return fmt.Errorf("create profile dir: %w", err)
	}

	path := filepath.Join(dir, "profile.gob")
	f, err := os.Create(path) // #nosec G304 -- profile dir is controlled by application config
	if err != nil {
		return fmt.Errorf("create file: %w", err)
	}
	defer f.Close()

	enc := gob.NewEncoder(f)
	return enc.Encode(p)
}

// LoadProfile loads a profile from disk.
func LoadProfile(dir string) (*ModelProfile, error) {
	path := filepath.Join(dir, "profile.gob")
	f, err := os.Open(path) // #nosec G304 -- profile dir is controlled by application config
	if err != nil {
		return nil, fmt.Errorf("open profile: %w", err)
	}
	defer f.Close()

	var p ModelProfile
	if err := gob.NewDecoder(f).Decode(&p); err != nil {
		return nil, fmt.Errorf("decode profile: %w", err)
	}
	return &p, nil
}
