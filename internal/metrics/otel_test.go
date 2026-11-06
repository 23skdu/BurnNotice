package metrics

import (
	"context"
	"net/http"
	"net/http/httptest"
	"os"
	"testing"
	"time"
)

func TestOTelConfigFromEnv(t *testing.T) {
	// Set test environment variables
	_ = os.Setenv(EnvOTelEndpoint, "http://localhost:4318")
	_ = os.Setenv(EnvOTelInsecure, "true")
	_ = os.Setenv(EnvServiceName, "test-service")
	defer func() {
		_ = os.Unsetenv(EnvOTelEndpoint)
		_ = os.Unsetenv(EnvOTelInsecure)
		_ = os.Unsetenv(EnvServiceName)
	}()

	cfg := OTelConfigFromEnv()
	if cfg.Endpoint != "http://localhost:4318" {
		t.Errorf("expected Endpoint = http://localhost:4318, got %s", cfg.Endpoint)
	}
	if !cfg.Insecure {
		t.Error("expected Insecure = true")
	}
	if cfg.ServiceName != "test-service" {
		t.Errorf("expected ServiceName = test-service, got %s", cfg.ServiceName)
	}
	if cfg.Interval != DefaultExportInterval {
		t.Errorf("expected Interval = %v, got %v", DefaultExportInterval, cfg.Interval)
	}
	if cfg.Timeout != DefaultTimeout {
		t.Errorf("expected Timeout = %v, got %v", DefaultTimeout, cfg.Timeout)
	}
}

func TestInitOTelExporter_NilConfig(t *testing.T) {
	// Should load from env (which is empty here, returning a no-op shutdown and no error)
	shutdown, err := InitOTelExporter(nil)
	if err != nil {
		t.Fatalf("InitOTelExporter(nil) failed: %v", err)
	}
	if shutdown == nil {
		t.Fatal("expected non-nil shutdown function")
	}
	// Try calling it
	if err := shutdown(context.Background()); err != nil {
		t.Errorf("shutdown failed: %v", err)
	}
}

func TestInitOTelExporter_EmptyEndpoint(t *testing.T) {
	cfg := &OTelConfig{Endpoint: ""}
	shutdown, err := InitOTelExporter(cfg)
	if err != nil {
		t.Fatalf("InitOTelExporter with empty endpoint failed: %v", err)
	}
	if shutdown == nil {
		t.Fatal("expected non-nil shutdown function")
	}
	if err := shutdown(context.Background()); err != nil {
		t.Errorf("shutdown failed: %v", err)
	}
}

func TestInitOTelExporter_InvalidURL(t *testing.T) {
	cfg := &OTelConfig{Endpoint: "::invalid-url::"}
	_, err := InitOTelExporter(cfg)
	if err == nil {
		t.Error("expected error for invalid URL endpoint")
	}
}

func TestInitOTelExporter_ValidEndpoint(t *testing.T) {
	// Create a mock HTTP server to act as OTel collector
	server := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer server.Close()

	cfg := &OTelConfig{
		Endpoint:    server.URL + "/v1/metrics",
		Insecure:    true,
		ServiceName: "test-service",
		Interval:    10 * time.Millisecond,
		Timeout:     50 * time.Millisecond,
	}

	shutdown, err := InitOTelExporter(cfg)
	if err != nil {
		t.Fatalf("failed to initialize exporter: %v", err)
	}
	if shutdown == nil {
		t.Fatal("expected non-nil shutdown function")
	}

	err = ShutdownOTel(context.Background())
	if err != nil {
		t.Errorf("ShutdownOTel failed: %v", err)
	}
}

func TestShutdownOTel_NotInitialized(t *testing.T) {
	// Reset global shutdown
	shutdownOTel = nil
	err := ShutdownOTel(context.Background())
	if err != nil {
		t.Errorf("ShutdownOTel when not initialized returned error: %v", err)
	}
}
