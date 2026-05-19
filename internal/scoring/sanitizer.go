package scoring

import (
	"errors"
	"strings"
)

// Sanitizer provides input validation and jailbreak detection patterns.
type Sanitizer struct {
	blocklist []string
}

// NewSanitizer creates a new sanitizer with default blocklist terms.
func NewSanitizer() *Sanitizer {
	return &Sanitizer{
		blocklist: []string{
			"ignore previous instructions",
			"dan mode",
			"jailbreak",
			"do anything now",
			"system override",
			"bypass safety",
		},
	}
}

// Sanitize checks prompt strings against security policies and blocklist entries.
func (s *Sanitizer) Sanitize(prompt string) error {
	lowerPrompt := strings.ToLower(prompt)
	for _, term := range s.blocklist {
		if strings.Contains(lowerPrompt, term) {
			return errors.New("malicious prompt pattern detected: potential jailbreak attempt")
		}
	}
	return nil
}
