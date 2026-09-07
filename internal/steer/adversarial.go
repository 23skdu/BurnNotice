package steer

import (
	"crypto/rand"
	"encoding/binary"
	"math"
	"strings"
	"sync"
	"unicode"

	"gonum.org/v1/gonum/floats"
)

// AdversarialShieldConfig specifies operational settings for the adversarial evasion defenses.
type AdversarialShieldConfig struct {
	NoiseRatio                float64 // ratio of vector norm to inject as noise (default 0.02 = 2%)
	AnomalousEntropyThreshold float64 // Shannon entropy threshold for suspicious token suffixes
	EnableTokenSanitizer      bool
	EnableLatentNoise         bool
	EnableGradientMasking     bool
}

// DefaultAdversarialShieldConfig returns recommended production parameters.
func DefaultAdversarialShieldConfig() AdversarialShieldConfig {
	return AdversarialShieldConfig{
		NoiseRatio:                0.02,
		AnomalousEntropyThreshold: 4.8,
		EnableTokenSanitizer:      true,
		EnableLatentNoise:         true,
		EnableGradientMasking:     true,
	}
}

// AdversarialShieldStats maintains thread-safe live shield metrics for dashboard telemetry.
type AdversarialShieldStats struct {
	BlockedSuffixes     int64   `json:"blocked_suffixes"`
	PerturbationsAdded  int64   `json:"perturbations_added"`
	MaskedGradients     int64   `json:"masked_gradients"`
	MeanNoiseMagnitude  float64 `json:"mean_noise_magnitude"`
}

// AdversarialShield provides multi-layered defenses against gradient-based suffix optimization (GCG)
// and activation probe inversion.
type AdversarialShield struct {
	mu            sync.RWMutex
	cfg           AdversarialShieldConfig
	stats         AdversarialShieldStats
	totalNoiseMag float64
}

// NewAdversarialShield creates and initializes an adversarial shield.
func NewAdversarialShield(cfg AdversarialShieldConfig) *AdversarialShield {
	if cfg.NoiseRatio <= 0 {
		cfg.NoiseRatio = 0.02
	}
	if cfg.AnomalousEntropyThreshold <= 0 {
		cfg.AnomalousEntropyThreshold = 4.8
	}
	return &AdversarialShield{
		cfg: cfg,
	}
}

// SanitizePrompt evaluates whether an input prompt contains anomalous suffix patterns characteristic of GCG attacks.
// Flags repeating non-alphanumeric punctuation bursts, excessive unigram entropy, and known adversarial delimiters.
func (s *AdversarialShield) SanitizePrompt(prompt string) (flagged bool, reason string) {
	if !s.cfg.EnableTokenSanitizer || len(prompt) == 0 {
		return false, ""
	}

	trimmed := strings.TrimSpace(prompt)
	if len(trimmed) < 15 {
		return false, ""
	}

	// 1. Check for dense repeating punctuation bursts (e.g. "==//==//!!??")
	punctCount := 0
	nonAlphaPunctStreak := 0
	maxStreak := 0
	for _, r := range trimmed {
		if unicode.IsPunct(r) || unicode.IsSymbol(r) {
			punctCount++
			nonAlphaPunctStreak++
			if nonAlphaPunctStreak > maxStreak {
				maxStreak = nonAlphaPunctStreak
			}
		} else if !unicode.IsSpace(r) {
			nonAlphaPunctStreak = 0
		}
	}

	if maxStreak >= 8 {
		s.incrementBlockedSuffix()
		return true, "dense non-alphanumeric adversarial burst detected"
	}

	// 2. High punctuation-to-character ratio on long inputs (> 45% punctuation)
	if len(trimmed) > 30 && float64(punctCount)/float64(len(trimmed)) > 0.45 {
		s.incrementBlockedSuffix()
		return true, "anomalous punctuation density"
	}

	// 3. Shannon entropy check on suffix (last 80 characters)
	suffix := trimmed
	if len(suffix) > 80 {
		suffix = suffix[len(suffix)-80:]
	}
	entropy := computeShannonEntropy(suffix)
	if entropy > s.cfg.AnomalousEntropyThreshold && punctCount > 10 {
		s.incrementBlockedSuffix()
		return true, "anomalous suffix Shannon entropy"
	}

	return false, ""
}

// InjectLatentNoise adds controlled stochastic Gaussian noise: S(a + eta) with ||eta|| = NoiseRatio * ||a||.
// This preserves macro-directionality while breaking micro-gradient adversarial inversion attacks.
func (s *AdversarialShield) InjectLatentNoise(activation []float64) ([]float64, error) {
	if !s.cfg.EnableLatentNoise || len(activation) == 0 {
		res := make([]float64, len(activation))
		copy(res, activation)
		return res, nil
	}

	dim := len(activation)
	norm := floats.Norm(activation, 2)
	if norm < 1e-9 {
		res := make([]float64, dim)
		copy(res, activation)
		return res, nil
	}

	targetNoiseNorm := s.cfg.NoiseRatio * norm
	noise := make([]float64, dim)

	// Sample pseudo-random Gaussian values via Box-Muller transform
	for i := 0; i < dim; i += 2 {
		u1 := secureUniformFloat()
		u2 := secureUniformFloat()
		if u1 <= 1e-15 {
			u1 = 1e-15
		}
		mag := math.Sqrt(-2.0 * math.Log(u1))
		z0 := mag * math.Cos(2.0*math.Pi*u2)
		z1 := mag * math.Sin(2.0*math.Pi*u2)

		noise[i] = z0
		if i+1 < dim {
			noise[i+1] = z1
		}
	}

	// Scale noise vector to target norm
	currentNoiseNorm := floats.Norm(noise, 2)
	if currentNoiseNorm > 1e-9 {
		floats.Scale(targetNoiseNorm/currentNoiseNorm, noise)
	}

	// Add noise to activation
	result := make([]float64, dim)
	for i := 0; i < dim; i++ {
		result[i] = activation[i] + noise[i]
	}

	s.recordNoise(targetNoiseNorm)
	return result, nil
}

// MaskSteeringGradient perturbs gradients that exhibit high alignment with sensitive concept probes,
// generating decoy gradient distributions to resist black-box probe localization.
func (s *AdversarialShield) MaskSteeringGradient(grad, probeDirection []float64) ([]float64, bool) {
	if !s.cfg.EnableGradientMasking || len(grad) == 0 || len(grad) != len(probeDirection) {
		res := make([]float64, len(grad))
		copy(res, grad)
		return res, false
	}

	normG := floats.Norm(grad, 2)
	normP := floats.Norm(probeDirection, 2)
	if normG < 1e-9 || normP < 1e-9 {
		res := make([]float64, len(grad))
		copy(res, grad)
		return res, false
	}

	cosSim := math.Abs(floats.Dot(grad, probeDirection) / (normG * normP))

	// If gradient is suspiciously aligned with probe (> 0.80 cosine similarity), mask it
	if cosSim > 0.80 {
		masked := make([]float64, len(grad))
		// Project out probe component and inject orthogonal decoy
		proj := floats.Dot(grad, probeDirection) / (normP * normP)
		for i := 0; i < len(grad); i++ {
			masked[i] = grad[i] - proj*probeDirection[i]
		}
		// Scale to preserve gradient energy
		normMasked := floats.Norm(masked, 2)
		if normMasked > 1e-9 {
			floats.Scale(normG/normMasked, masked)
		}

		s.incrementMaskedGradient()
		return masked, true
	}

	res := make([]float64, len(grad))
	copy(res, grad)
	return res, false
}

// Stats returns a snapshot of the current shield metrics.
func (s *AdversarialShield) Stats() AdversarialShieldStats {
	s.mu.RLock()
	defer s.mu.RUnlock()
	return s.stats
}

func (s *AdversarialShield) incrementBlockedSuffix() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.BlockedSuffixes++
}

func (s *AdversarialShield) incrementMaskedGradient() {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.MaskedGradients++
}

func (s *AdversarialShield) recordNoise(mag float64) {
	s.mu.Lock()
	defer s.mu.Unlock()
	s.stats.PerturbationsAdded++
	s.totalNoiseMag += mag
	s.stats.MeanNoiseMagnitude = s.totalNoiseMag / float64(s.stats.PerturbationsAdded)
}

func computeShannonEntropy(s string) float64 {
	if len(s) == 0 {
		return 0.0
	}
	freq := make(map[rune]float64)
	for _, r := range s {
		freq[r]++
	}
	n := float64(len([]rune(s)))
	entropy := 0.0
	for _, count := range freq {
		p := count / n
		if p > 0 {
			entropy -= p * math.Log2(p)
		}
	}
	return entropy
}

func secureUniformFloat() float64 {
	var b [8]byte
	_, err := rand.Read(b[:])
	if err != nil {
		return 0.5
	}
	val := binary.BigEndian.Uint64(b[:])
	return float64(val) / float64(math.MaxUint64)
}
