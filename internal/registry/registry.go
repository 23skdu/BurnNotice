package registry

import (
	"fmt"
	"os"
	"path/filepath"
	"sort"
	"strings"
	"sync"
	"time"

	"burnnotice/internal/profile"
)

// Metadata holds profile metadata.
type Metadata struct {
	ModelID         string    `json:"model_id"`
	Architecture    string    `json:"architecture"`
	Date            time.Time `json:"date"`
	DirectionCount  int       `json:"direction_count"`
	ValidationScore float64   `json:"validation_score"`
	Version         int       `json:"version"`
	Path            string    `json:"path"`
}

// Registry manages a searchable index of profiles.
type Registry struct {
	mu       sync.RWMutex
	profiles map[string]*Metadata
	baseDir  string
}

// NewRegistry creates a registry.
func NewRegistry(baseDir string) *Registry {
	return &Registry{
		profiles: make(map[string]*Metadata),
		baseDir:  baseDir,
	}
}

// Scan discovers profiles in the base directory.
func (r *Registry) Scan() error {
	r.mu.Lock()
	defer r.mu.Unlock()

	entries, err := os.ReadDir(r.baseDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil
		}
		return err
	}

	for _, entry := range entries {
		if !entry.IsDir() {
			continue
		}

		modelDir := filepath.Join(r.baseDir, entry.Name())
		profilePath := filepath.Join(modelDir, "profile.gob")

		if _, err := os.Stat(profilePath); os.IsNotExist(err) {
			continue
		}

		p, err := profile.LoadProfile(modelDir)
		if err != nil {
			continue
		}

		dirCount := 0
		for _, layers := range p.Directions {
			dirCount += len(layers)
		}

		r.profiles[p.ModelID] = &Metadata{
			ModelID:         p.ModelID,
			Architecture:    "unknown",
			Date:            time.Now(),
			DirectionCount:  dirCount,
			ValidationScore: 0.0,
			Version:         1,
			Path:            modelDir,
		}
	}

	return nil
}

// List returns all profiles with optional filtering.
func (r *Registry) List(filter string) []Metadata {
	r.mu.RLock()
	defer r.mu.RUnlock()

	var result []Metadata
	for _, meta := range r.profiles {
		if filter == "" || strings.Contains(meta.ModelID, filter) {
			result = append(result, *meta)
		}
	}

	sort.Slice(result, func(i, j int) bool {
		return result[i].Date.After(result[j].Date)
	})

	return result
}

// Get loads a profile by ID.
func (r *Registry) Get(modelID string) (*profile.ModelProfile, error) {
	r.mu.RLock()
	meta, ok := r.profiles[modelID]
	r.mu.RUnlock()

	if !ok {
		return nil, fmt.Errorf("profile not found: %s", modelID)
	}

	return profile.LoadProfile(meta.Path)
}

// Delete removes a profile and all versions.
func (r *Registry) Delete(modelID string) error {
	r.mu.Lock()
	meta, ok := r.profiles[modelID]
	if !ok {
		r.mu.Unlock()
		return fmt.Errorf("profile not found: %s", modelID)
	}
	delete(r.profiles, modelID)
	r.mu.Unlock()

	return os.RemoveAll(meta.Path)
}

// Add manually registers a profile.
func (r *Registry) Add(meta Metadata) {
	r.mu.Lock()
	defer r.mu.Unlock()
	r.profiles[meta.ModelID] = &meta
}

// Count returns the number of profiles.
func (r *Registry) Count() int {
	r.mu.RLock()
	defer r.mu.RUnlock()
	return len(r.profiles)
}
