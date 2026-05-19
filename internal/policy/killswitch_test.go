package policy

import (
	"context"
	"sync"
	"testing"
	"time"
)

func TestNewKillSwitch(t *testing.T) {
	var alerts []string
	alertFn := func(reason string) {
		alerts = append(alerts, reason)
	}

	ks, ctx := NewKillSwitch(alertFn)

	if !ks.IsArmed() {
		t.Error("kill switch should be armed initially")
	}

	if ctx == nil {
		t.Fatal("context should not be nil")
	}

	ks.Trigger("test reason")

	if ks.IsArmed() {
		t.Error("kill switch should be disarmed after trigger")
	}

	if len(alerts) != 1 || alerts[0] != "test reason" {
		t.Errorf("alerts = %v, want [test reason]", alerts)
	}
}

func TestKillSwitch_ContextCancellation(t *testing.T) {
	ks, ctx := NewKillSwitch(nil)

	done := make(chan struct{})
	go func() {
		<-ctx.Done()
		close(done)
	}()

	ks.Trigger("cancelled")

	select {
	case <-done:
		// context was cancelled as expected
	case <-time.After(time.Second):
		t.Fatal("context was not cancelled within timeout")
	}
}

func TestKillSwitch_Reset(t *testing.T) {
	ks, ctx1 := NewKillSwitch(nil)

	ks.Trigger("triggered")
	if ks.IsArmed() {
		t.Fatal("should be disarmed after trigger")
	}

	ctx2 := ks.Reset()
	if !ks.IsArmed() {
		t.Fatal("should be armed after reset")
	}

	if ctx1 == ctx2 {
		t.Fatal("reset should return a new context")
	}

	select {
	case <-ctx1.Done():
		// old context should be cancelled
	default:
		t.Error("old context should be cancelled after reset")
	}

	select {
	case <-ctx2.Done():
		t.Error("new context should not be done")
	default:
		// new context is still active
	}
}

func TestKillSwitch_TriggerWhenNotArmed(t *testing.T) {
	var alertCount int
	alertFn := func(_ string) {
		alertCount++
	}

	ks, _ := NewKillSwitch(alertFn)

	ks.Trigger("first")
	ks.Trigger("second")

	if alertCount != 1 {
		t.Errorf("alertCount = %d, want 1 (second trigger should be ignored)", alertCount)
	}
}

func TestKillSwitch_NilAlertFn(t *testing.T) {
	ks, _ := NewKillSwitch(nil)

	ks.Trigger("no crash")

	if ks.IsArmed() {
		t.Error("should be disarmed after trigger")
	}
}

func TestKillSwitch_ContextReturnsCorrectContext(t *testing.T) {
	ks, originalCtx := NewKillSwitch(nil)

	retrievedCtx := ks.Context()
	if retrievedCtx != originalCtx {
		t.Error("Context() should return the same context")
	}

	ks.Reset()
	newCtx := ks.Context()
	if newCtx == retrievedCtx {
		t.Error("Context() should return new context after reset")
	}
}

func TestKillSwitch_ConcurrentTriggerReset(t *testing.T) {
	ks, _ := NewKillSwitch(nil)

	var wg sync.WaitGroup
	for i := 0; i < 10; i++ {
		wg.Add(2)
		go func() {
			defer wg.Done()
			ks.Trigger("concurrent trigger")
		}()
		go func() {
			defer wg.Done()
			ks.Reset()
		}()
	}
	wg.Wait()
}

func TestKillSwitch_ContextDoneAfterTrigger(t *testing.T) {
	ks, ctx := NewKillSwitch(nil)

	ks.Trigger("test")

	if ctx.Err() != context.Canceled {
		t.Errorf("ctx.Err() = %v, want context.Canceled", ctx.Err())
	}
}
