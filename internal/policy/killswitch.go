package policy

import (
	"context"
	"sync"
)

// KillSwitch provides an emergency stop for inference.
type KillSwitch struct {
	mu      sync.RWMutex
	armed   bool
	cancel  context.CancelFunc
	ctx     context.Context
	alertFn func(string)
}

// NewKillSwitch creates an armed kill switch.
func NewKillSwitch(alertFn func(string)) (*KillSwitch, context.Context) {
	ctx, cancel := context.WithCancel(context.Background())
	return &KillSwitch{
		armed:   true,
		cancel:  cancel,
		ctx:     ctx,
		alertFn: alertFn,
	}, ctx
}

// Trigger cancels the inference context and sends an alert.
func (k *KillSwitch) Trigger(reason string) {
	k.mu.Lock()
	defer k.mu.Unlock()

	if !k.armed {
		return
	}
	k.cancel()
	k.armed = false

	if k.alertFn != nil {
		k.alertFn(reason)
	}
}

// Reset re-arms the kill switch with a fresh context.
func (k *KillSwitch) Reset() context.Context {
	k.mu.Lock()
	defer k.mu.Unlock()

	ctx, cancel := context.WithCancel(context.Background())
	k.ctx = ctx
	k.cancel = cancel
	k.armed = true
	return ctx
}

// IsArmed returns whether the kill switch is currently armed.
func (k *KillSwitch) IsArmed() bool {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.armed
}

// Context returns the current inference context.
func (k *KillSwitch) Context() context.Context {
	k.mu.RLock()
	defer k.mu.RUnlock()
	return k.ctx
}
