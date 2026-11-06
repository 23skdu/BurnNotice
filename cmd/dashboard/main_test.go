package main

import (
	"os"
	"testing"
	"time"
)

func TestMainDashboard(t *testing.T) {
	// Start the server in a goroutine
	_ = os.Setenv("PORT", "8081") // Use a different port
	defer func() { _ = os.Unsetenv("PORT") }()

	go func() {
		// Mock os.Exit or panic to prevent actual fatal on failure
		defer func() {
			_ = recover()
		}()
		main()
	}()

	// Wait for server to start
	time.Sleep(100 * time.Millisecond)
}
