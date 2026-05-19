package profile

import (
	"encoding/json"
	"fmt"
	"math"
	"os"
	"path/filepath"
	"time"
)

// Version holds profile version metadata.
type Version struct {
	Version   int       `json:"version"`
	CreatedAt time.Time `json:"created_at"`
	ModelID   string    `json:"model_id"`
	DiffFrom  string    `json:"diff_from"`
}

// VersionManager manages profile versioning.
type VersionManager struct {
	profileDir string
}

// NewVersionManager creates a version manager.
func NewVersionManager(profileDir string) *VersionManager {
	return &VersionManager{profileDir: profileDir}
}

// NextVersion returns the next version number.
func (vm *VersionManager) NextVersion() (int, error) {
	versionsDir := filepath.Join(vm.profileDir, "versions")
	entries, err := os.ReadDir(versionsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return 1, nil
		}
		return 0, err
	}

	maxVersion := 0
	for _, entry := range entries {
		if entry.IsDir() {
			var v int
			if _, err := fmt.Sscanf(entry.Name(), "v%d", &v); err == nil && v > maxVersion {
				maxVersion = v
			}
		}
	}

	return maxVersion + 1, nil
}

// SaveVersion saves a profile version.
func (vm *VersionManager) SaveVersion(profile *ModelProfile, diffFrom string) (*Version, error) {
	version, err := vm.NextVersion()
	if err != nil {
		return nil, fmt.Errorf("get next version: %w", err)
	}

	versionsDir := filepath.Join(vm.profileDir, "versions")
	versionDir := filepath.Join(versionsDir, fmt.Sprintf("v%d", version))

	if err := os.MkdirAll(versionDir, 0750); err != nil {
		return nil, fmt.Errorf("create version dir: %w", err)
	}

	if err := profile.Save(versionDir); err != nil {
		return nil, fmt.Errorf("save profile: %w", err)
	}

	v := &Version{
		Version:   version,
		CreatedAt: time.Now(),
		ModelID:   profile.ModelID,
		DiffFrom:  diffFrom,
	}

	metaPath := filepath.Join(versionDir, "version.json")
	data, err := json.MarshalIndent(v, "", "  ")
	if err != nil {
		return nil, fmt.Errorf("marshal version: %w", err)
	}

	if err := os.WriteFile(metaPath, data, 0600); err != nil {
		return nil, fmt.Errorf("write version metadata: %w", err)
	}

	return v, nil
}

// LoadVersion loads a specific profile version.
func (vm *VersionManager) LoadVersion(version int) (*ModelProfile, error) {
	versionDir := filepath.Join(vm.profileDir, "versions", fmt.Sprintf("v%d", version))
	return LoadProfile(versionDir)
}

// ListVersions returns all available versions.
func (vm *VersionManager) ListVersions() ([]Version, error) {
	versionsDir := filepath.Join(vm.profileDir, "versions")
	entries, err := os.ReadDir(versionsDir)
	if err != nil {
		if os.IsNotExist(err) {
			return nil, nil
		}
		return nil, err
	}

	var versions []Version
	for _, entry := range entries {
		if entry.IsDir() {
			metaPath := filepath.Join(versionsDir, entry.Name(), "version.json")
			// #nosec G304 -- version files are internally managed
			data, err := os.ReadFile(metaPath)
			if err != nil {
				continue
			}

			var v Version
			if err := json.Unmarshal(data, &v); err == nil {
				versions = append(versions, v)
			}
		}
	}

	return versions, nil
}

// Diff computes the cosine distance between two profile versions.
func Diff(old, new *ModelProfile) map[string]float64 {
	distances := make(map[string]float64)

	for dirName, oldLayers := range old.Directions {
		if newLayers, ok := new.Directions[dirName]; ok {
			for layerIdx, oldDir := range oldLayers {
				if newDir, ok := newLayers[layerIdx]; ok {
					distances[fmt.Sprintf("%s:%d", dirName, layerIdx)] = versionCosineDistance(oldDir, newDir)
				}
			}
		}
	}

	return distances
}

// DetectVersionDrift flags directions with cosine distance above threshold.
func DetectVersionDrift(distances map[string]float64, threshold float64) []string {
	var drifts []string
	for key, dist := range distances {
		if dist > threshold {
			drifts = append(drifts, key)
		}
	}
	return drifts
}

func versionCosineDistance(a, b []float64) float64 {
	if len(a) != len(b) || len(a) == 0 {
		return 0
	}

	var dot, normA, normB float64
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}

	normA = math.Sqrt(normA)
	normB = math.Sqrt(normB)

	if normA < 1e-10 || normB < 1e-10 {
		return 0
	}

	similarity := dot / (normA * normB)
	return 1.0 - similarity
}
