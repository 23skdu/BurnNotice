package metrics

import (
	"net/http"
	"time"

	"github.com/prometheus/client_golang/prometheus/promhttp"
)

// Handler returns an http.Handler that serves Prometheus-format metrics.
// This endpoint can be scraped by a Prometheus server or an OpenTelemetry
// Collector using the Prometheus Receiver for OTel-native metric pipelines.
func Handler() http.Handler {
	return promhttp.Handler()
}

// HTTPMiddleware wraps an http.Handler with duration and active-request metrics.
func HTTPMiddleware(next http.Handler) http.Handler {
	return http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		start := time.Now()
		next.ServeHTTP(w, r)
		HTTPRequestDuration.WithLabelValues(r.Method, r.URL.Path, "200").Observe(time.Since(start).Seconds())
	})
}
