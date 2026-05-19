package checkpoint

import (
	"encoding/json"
	"fmt"
	"os"
	"path/filepath"
	"sync"
	"time"
)

// Manager handles periodic state persistence with atomic writes.
type Manager struct {
	mu       sync.RWMutex
	interval time.Duration
	path     string
	state    interface{}
	stopCh   chan struct{}
	ticker   *time.Ticker
}

// NewManager creates a checkpoint manager.
func NewManager(interval time.Duration, path string) *Manager {
	return &Manager{
		interval: interval,
		path:     path,
		stopCh:   make(chan struct{}),
	}
}

// Start begins periodic checkpointing.
func (m *Manager) Start(state interface{}) {
	m.mu.Lock()
	m.state = state
	m.mu.Unlock()

	if err := m.save(state); err != nil {
		return
	}

	m.ticker = time.NewTicker(m.interval)
	go func() {
		for {
			select {
			case <-m.ticker.C:
				m.mu.RLock()
				s := m.state
				m.mu.RUnlock()
				_ = m.save(s)
			case <-m.stopCh:
				m.ticker.Stop()
				return
			}
		}
	}()
}

// Stop stops the checkpoint ticker.
func (m *Manager) Stop() {
	close(m.stopCh)
}

// Save performs an immediate checkpoint.
func (m *Manager) Save(state interface{}) error {
	m.mu.Lock()
	m.state = state
	m.mu.Unlock()
	return m.save(state)
}

func (m *Manager) save(state interface{}) error {
	if err := os.MkdirAll(filepath.Dir(m.path), 0750); err != nil {
		return fmt.Errorf("create checkpoint dir: %w", err)
	}

	tmpPath := m.path + ".tmp"

	data, err := json.Marshal(state)
	if err != nil {
		return fmt.Errorf("marshal state: %w", err)
	}

	if err := os.WriteFile(tmpPath, data, 0600); err != nil {
		return fmt.Errorf("write temp file: %w", err)
	}

	if err := os.Rename(tmpPath, m.path); err != nil {
		_ = os.Remove(tmpPath)
		return fmt.Errorf("atomic rename: %w", err)
	}

	return nil
}

// Load decodes a checkpoint file and returns the state.
func Load(path string) (interface{}, error) {
	// #nosec G304 -- checkpoint path is internally managed
	data, err := os.ReadFile(path)
	if err != nil {
		return nil, fmt.Errorf("open checkpoint: %w", err)
	}

	var state interface{}
	if err := json.Unmarshal(data, &state); err != nil {
		return nil, fmt.Errorf("unmarshal checkpoint: %w", err)
	}

	return state, nil
}

// Exists checks if a checkpoint file exists.
func Exists(path string) bool {
	_, err := os.Stat(path)
	return err == nil
}
