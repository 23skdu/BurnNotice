package metrics

import (
	"context"
	"errors"
	"net/url"
	"os"
	"time"

	"go.opentelemetry.io/otel"
	"go.opentelemetry.io/otel/exporters/otlp/otlpmetric/otlpmetrichttp"
	"go.opentelemetry.io/otel/sdk/metric"
	"go.opentelemetry.io/otel/sdk/resource"
	semconv "go.opentelemetry.io/otel/semconv/v1.30.0"
)

const (
	EnvOTelEndpoint       = "BURNNOTICE_OTEL_ENDPOINT"
	EnvOTelInsecure       = "BURNNOTICE_OTEL_INSECURE"
	EnvOTelExportInterval = "BURNNOTICE_OTEL_EXPORT_INTERVAL"
	EnvServiceName        = "OTEL_SERVICE_NAME"

	DefaultServiceName    = "burnnotice"
	DefaultExportInterval = 30 * time.Second
	DefaultTimeout        = 10 * time.Second
)

var shutdownOTel func(context.Context) error

type OTelConfig struct {
	Endpoint    string
	Insecure    bool
	ServiceName string
	Interval    time.Duration
	Timeout     time.Duration
}

func OTelConfigFromEnv() *OTelConfig {
	cfg := &OTelConfig{
		Endpoint:    os.Getenv(EnvOTelEndpoint),
		Insecure:    os.Getenv(EnvOTelInsecure) == "true",
		ServiceName: DefaultServiceName,
		Interval:    DefaultExportInterval,
		Timeout:     DefaultTimeout,
	}
	if s := os.Getenv(EnvServiceName); s != "" {
		cfg.ServiceName = s
	}
	return cfg
}

func InitOTelExporter(cfg *OTelConfig) (func(context.Context) error, error) {
	if cfg == nil {
		cfg = OTelConfigFromEnv()
	}
	if cfg.Endpoint == "" {
		return func(context.Context) error { return nil }, nil
	}

	parsedURL, err := url.Parse(cfg.Endpoint)
	if err != nil {
		return nil, err
	}

	opts := []otlpmetrichttp.Option{
		otlpmetrichttp.WithEndpoint(parsedURL.Host),
	}
	if cfg.Insecure {
		opts = append(opts, otlpmetrichttp.WithInsecure())
	}
	if parsedURL.Path != "" && parsedURL.Path != "/" {
		opts = append(opts, otlpmetrichttp.WithURLPath(parsedURL.Path))
	}
	if cfg.Timeout > 0 {
		opts = append(opts, otlpmetrichttp.WithTimeout(cfg.Timeout))
	}

	ctx, cancel := context.WithTimeout(context.Background(), cfg.Timeout)
	defer cancel()

	exporter, err := otlpmetrichttp.New(ctx, opts...)
	if err != nil {
		return nil, err
	}

	res, err := resource.New(ctx,
		resource.WithAttributes(
			semconv.ServiceName(cfg.ServiceName),
		),
	)
	if err != nil {
		return nil, err
	}

	reader := metric.NewPeriodicReader(exporter,
		metric.WithInterval(cfg.Interval),
	)

	provider := metric.NewMeterProvider(
		metric.WithResource(res),
		metric.WithReader(reader),
	)

	otel.SetMeterProvider(provider)
	shutdownOTel = provider.Shutdown

	return provider.Shutdown, nil
}

func ShutdownOTel(ctx context.Context) error {
	if shutdownOTel != nil {
		return shutdownOTel(ctx)
	}
	return nil
}

var ErrOTelNotInitialized = errors.New("OTel exporter not initialized")
