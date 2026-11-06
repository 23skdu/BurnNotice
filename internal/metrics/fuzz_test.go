package metrics

import (
	"net/http"
	"net/http/httptest"
	"testing"
)

func FuzzAdaptiveSampler(f *testing.F) {
	f.Add(uint64(0))
	f.Add(uint64(1))
	f.Add(uint64(10))
	f.Add(uint64(100))
	f.Add(uint64(1000))
	f.Add(uint64(1<<63 - 1))

	f.Fuzz(func(t *testing.T, n uint64) {
		s := NewAdaptiveSampler(n)

		weight := s.Weight()
		if weight < 1 {
			t.Errorf("weight = %f, want >= 1", weight)
		}
		if n == 0 && weight != 1.0 {
			t.Errorf("weight for n=0 = %f, want 1.0", weight)
		}
		if n == 1 && weight != 1.0 {
			t.Errorf("weight for n=1 = %f, want 1.0", weight)
		}

		shouldSample := s.ShouldSample()
		_ = shouldSample
	})
}

func FuzzAdaptiveSamplerSetRate(f *testing.F) {
	f.Add(uint64(0), uint64(0))
	f.Add(uint64(1), uint64(5))
	f.Add(uint64(10), uint64(3))
	f.Add(uint64(100), uint64(50))

	f.Fuzz(func(t *testing.T, initial, newN uint64) {
		s := NewAdaptiveSampler(initial)
		s.SetSampleRate(newN)

		weight := s.Weight()
		if weight < 1 {
			t.Errorf("after SetSampleRate(%d), weight = %f, want >= 1", newN, weight)
		}
		if newN == 0 && weight != 1.0 {
			t.Errorf("after SetSampleRate(0), weight = %f, want 1.0", weight)
		}

		s.ShouldSample()
	})
}

func FuzzDefaultSampler(f *testing.F) {
	f.Add(uint64(50))
	f.Add(uint64(200))

	f.Fuzz(func(t *testing.T, rate uint64) {
		if rate == 0 {
			rate = 1
		}
		if rate > 10000 {
			rate = 10000
		}
		s := NewAdaptiveSampler(rate)

		samples := 0
		attempts := rate * 3
		if attempts == 0 || attempts < rate {
			attempts = 3
		}
		for range attempts {
			if s.ShouldSample() {
				samples++
			}
		}

		if rate <= 1000 && samples == 0 {
			t.Errorf("expected at least one sample in %d attempts", attempts)
		}
	})
}

func FuzzMetricsHandler(f *testing.F) {
	f.Add("/metrics")
	f.Add("/")
	f.Add("/health")
	f.Add("/nonexistent")

	f.Fuzz(func(t *testing.T, path string) {
		handler := Handler()
		req, err := http.NewRequest(http.MethodGet, path, nil)
		if err != nil {
			t.Skip()
		}
		w := httptest.NewRecorder()

		handler.ServeHTTP(w, req)

		if w.Code == 0 {
			t.Error("handler returned zero status code")
		}
		if w.Code < 100 || w.Code > 599 {
			t.Errorf("handler returned invalid status code: %d", w.Code)
		}
	})
}

func FuzzOTelConfigFromEnv(f *testing.F) {
	f.Add("http://localhost:4318", "true", "burnnotice-test")
	f.Add("https://otel.example.com:4318", "false", "my-service")
	f.Add("", "false", "")
	f.Add("http://127.0.0.1:4318/v1/metrics", "true", "burnnotice")

	f.Fuzz(func(t *testing.T, endpoint, insecure, serviceName string) {
		if containsNUL(endpoint, insecure, serviceName) {
			t.Skip()
		}
		t.Setenv("BURNNOTICE_OTEL_ENDPOINT", endpoint)
		t.Setenv("BURNNOTICE_OTEL_INSECURE", insecure)
		t.Setenv("OTEL_SERVICE_NAME", serviceName)

		cfg := OTelConfigFromEnv()
		if cfg == nil {
			t.Fatal("OTelConfigFromEnv returned nil")
		}
		if cfg.ServiceName == "" {
			t.Error("ServiceName should not be empty")
		}
		_ = cfg.Endpoint
		_ = cfg.Insecure
	})
}

func containsNUL(values ...string) bool {
	for _, v := range values {
		for i := 0; i < len(v); i++ {
			if v[i] == 0 {
				return true
			}
		}
	}
	return false
}
