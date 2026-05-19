package longbow

import (
	"context"
	"testing"
)

func TestNewClient_Defaults(t *testing.T) {
	c := NewClient(Config{Addr: "localhost:3000", Dim: 4096})

	if c.addr != "localhost:3000" {
		t.Errorf("addr = %q, want localhost:3000", c.addr)
	}
	if c.dim != 4096 {
		t.Errorf("dim = %d, want 4096", c.dim)
	}
	if c.metric != "cosine" {
		t.Errorf("metric = %q, want cosine", c.metric)
	}
	if c.m != 16 {
		t.Errorf("m = %d, want 16", c.m)
	}
	if c.efConstr != 100 {
		t.Errorf("efConstr = %d, want 100", c.efConstr)
	}
}

func TestNewClient_CustomConfig(t *testing.T) {
	c := NewClient(Config{
		Addr:     "remote:5000",
		Dim:      768,
		Metric:   "euclidean",
		M:        32,
		EFConstr: 200,
	})

	if c.metric != "euclidean" {
		t.Errorf("metric = %q, want euclidean", c.metric)
	}
	if c.m != 32 {
		t.Errorf("m = %d, want 32", c.m)
	}
	if c.efConstr != 200 {
		t.Errorf("efConstr = %d, want 200", c.efConstr)
	}
}

func TestIngest_LengthMismatch(t *testing.T) {
	c := NewClient(Config{Dim: 128})
	err := c.Ingest(context.Background(), [][]float32{{1, 2}}, []uint64{1, 2})
	if err == nil {
		t.Error("expected error for length mismatch")
	}
}

func TestIngest_Empty(t *testing.T) {
	c := NewClient(Config{Dim: 128})
	err := c.Ingest(context.Background(), nil, nil)
	if err != nil {
		t.Errorf("Ingest(empty) error: %v", err)
	}
}

func TestQuery_Empty(t *testing.T) {
	c := NewClient(Config{Dim: 128})
	result, err := c.Query(context.Background(), nil)
	if err != nil {
		t.Errorf("Query(empty) error: %v", err)
	}
	if result != nil {
		t.Errorf("Query(empty) = %v, want nil", result)
	}
}

func TestSearch_DimensionMismatch(t *testing.T) {
	c := NewClient(Config{Dim: 128})
	_, _, err := c.Search(context.Background(), []float32{1, 2, 3}, 10)
	if err == nil {
		t.Error("expected error for dimension mismatch")
	}
}

func TestClose_NoError(t *testing.T) {
	c := NewClient(Config{Dim: 128})
	err := c.Close()
	if err != nil {
		t.Errorf("Close error: %v", err)
	}
}
