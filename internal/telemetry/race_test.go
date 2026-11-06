package telemetry

import (
	"net/http"
	"net/http/httptest"
	"sync"
	"testing"
)

func TestReporter_ConcurrentSubmitAndAggregate(t *testing.T) {
	srv := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.WriteHeader(http.StatusOK)
	}))
	defer srv.Close()

	rep := NewReporter(srv.URL, t.TempDir())
	var wg sync.WaitGroup
	for i := 0; i < 8; i++ {
		wg.Add(1)
		go func(id int) {
			defer wg.Done()
			for j := 0; j < 10; j++ {
				r := Report{
					ModelArchitecture:   "llama",
					ModelSize:           "8B",
					SteeringMethod:      "mpoa",
					RefusalRate:         0.05,
					KLDivergence:        0.02,
					CapabilityRetention: 0.99,
				}
				if err := rep.Submit(&r); err != nil {
					t.Errorf("concurrent Submit failed: %v", err)
				}
				_ = Aggregate([]Report{r})
			}
		}(i)
	}
	wg.Wait()
}
