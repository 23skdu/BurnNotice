package longbow

import (
	"context"
	"math"
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

func TestClient_LocalFallbackFlow(t *testing.T) {
	c := NewClient(Config{
		Addr:   "localhost:0",
		Dim:    3,
		Metric: "cosine",
	})

	ctx := context.Background()
	acts := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}
	ids := []uint64{101, 102, 103}

	// Should trigger automatic fallback on localhost:0
	err := c.Ingest(ctx, acts, ids)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}

	if !c.IsFallbackActive() {
		t.Error("expected fallback mode to be active")
	}

	// Test Query
	found, err := c.Query(ctx, []uint64{102, 103})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(found) != 2 {
		t.Errorf("got %d vectors, want 2", len(found))
	}

	// Test Search
	matchedIDs, dists, err := c.Search(ctx, []float32{0.0, 0.99, 0.01}, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matchedIDs) != 2 {
		t.Fatalf("got %d matches, want 2", len(matchedIDs))
	}
	if dists[0] >= dists[1] {
		t.Errorf("expected closer match first: %f >= %f", dists[0], dists[1])
	}
	if matchedIDs[0] != 102 {
		t.Errorf("got closest ID %d, want 102", matchedIDs[0])
	}
}

func TestNewClient_SQLiteBackend(t *testing.T) {
	c := NewClient(Config{Dim: 128, Backend: "sqlite"})
	if c.db == nil {
		t.Fatal("expected sqlite backend to be initialized")
	}
	if c.IsFallbackActive() {
		t.Error("expected sqlite backend without fallback")
	}
}

func TestIngest_Query_Search_SQLite(t *testing.T) {
	c := NewClient(Config{Dim: 3, Backend: "sqlite"})
	defer c.Close()

	ctx := context.Background()
	acts := [][]float32{
		{1.0, 0.0, 0.0},
		{0.0, 1.0, 0.0},
		{0.0, 0.0, 1.0},
	}
	ids := []uint64{10, 20, 30}

	err := c.Ingest(ctx, acts, ids)
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}

	// Query by IDs
	found, err := c.Query(ctx, []uint64{20, 10})
	if err != nil {
		t.Fatalf("Query failed: %v", err)
	}
	if len(found) != 2 {
		t.Fatalf("got %d vectors, want 2", len(found))
	}
	if found[0][0] != 0 || found[0][1] != 1 || found[0][2] != 0 {
		t.Errorf("unexpected first vector: %v", found[0])
	}
	if found[1][0] != 1 || found[1][1] != 0 || found[1][2] != 0 {
		t.Errorf("unexpected second vector: %v", found[1])
	}

	// Search nearest to (0, 0.99, 0.01) → closest is (0,1,0) id=20
	matchedIDs, dists, err := c.Search(ctx, []float32{0.0, 0.99, 0.01}, 2)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matchedIDs) != 2 {
		t.Fatalf("got %d matches, want 2", len(matchedIDs))
	}
	if dists[0] >= dists[1] {
		t.Errorf("expected closer match first: %f >= %f", dists[0], dists[1])
	}
	if matchedIDs[0] != 20 {
		t.Errorf("got closest ID %d, want 20", matchedIDs[0])
	}
}

func TestClient_ZeroDowntimeHotSwap(t *testing.T) {
	c := NewClient(Config{
		Addr:   "localhost:9099",
		Dim:    2,
		Metric: "euclidean",
	})

	ctx := context.Background()
	c.SetFallbackMode(true) // force fallback

	err := c.Ingest(ctx, [][]float32{{1.0, 1.0}}, []uint64{50})
	if err != nil {
		t.Fatalf("Ingest failed: %v", err)
	}

	matched, dists, err := c.Search(ctx, []float32{1.0, 2.0}, 1)
	if err != nil {
		t.Fatalf("Search failed: %v", err)
	}
	if len(matched) != 1 || matched[0] != 50 {
		t.Errorf("unexpected matched results: %v", matched)
	}
	if math.Abs(float64(dists[0])-1.0) > 1e-5 {
		t.Errorf("unexpected euclidean distance: %f", dists[0])
	}
}

func TestClient_AdditionalEdgeCases(t *testing.T) {
	ctx := context.Background()

	// 1. Invalid SQLite path should trigger fallback
	cBadSqlite := NewClient(Config{
		Addr:    "/nonexistent_dir/vectors.db",
		Dim:     3,
		Backend: "sqlite",
	})
	if !cBadSqlite.IsFallbackActive() {
		t.Error("expected fallback mode to be active for invalid SQLite path")
	}

	// 2. SetFallbackMode(false)
	cBadSqlite.SetFallbackMode(false)
	if cBadSqlite.IsFallbackActive() {
		t.Error("expected fallback mode to be inactive after SetFallbackMode(false)")
	}

	// 3. Manual mock of Client with fallbackMode=0 and sc=nil
	cMock := &Client{
		dim:          3,
		fallbackMode: 0,
		sc:           nil,
	}
	err := cMock.Ingest(ctx, [][]float32{{1.0, 2.0, 3.0}}, []uint64{10})
	if err != nil {
		t.Fatalf("Ingest on mock client failed: %v", err)
	}
	if !cMock.IsFallbackActive() {
		t.Error("expected Ingest to activate fallback mode when sc is nil")
	}

	// 4. Closed SQLite database errors
	cSqlite := NewClient(Config{Dim: 3, Backend: "sqlite", Addr: "file::memory:?cache=shared"})
	cSqlite.Close() // Close immediately

	// Ingest on closed SQLite db should fail
	err = cSqlite.Ingest(ctx, [][]float32{{1, 2, 3}}, []uint64{10})
	if err == nil {
		t.Error("expected error for Ingest on closed SQLite database")
	}

	// Query on closed SQLite db should fail
	_, err = cSqlite.Query(ctx, []uint64{10})
	if err == nil {
		t.Error("expected error for Query on closed SQLite database")
	}

	// Search on closed SQLite db should fail
	_, _, err = cSqlite.Search(ctx, []float32{1, 2, 3}, 5)
	if err == nil {
		t.Error("expected error for Search on closed SQLite database")
	}

	// 5. placeholders(0)
	ph := placeholders(0)
	if ph != "" {
		t.Errorf("expected empty string for placeholders(0), got %q", ph)
	}

	// 6. cosineSimilarity32 with zero-norm vectors
	sim := cosineSimilarity32([]float32{0, 0, 0}, []float32{1, 2, 3})
	if sim != 0 {
		t.Errorf("expected 0 similarity for zero vector, got %f", sim)
	}

	// 7. Test Query and Search on mock uninitialized client switches to fallback
	cFlightMock := &Client{
		dim:          3,
		fallbackMode: 0,
		sc:           nil,
	}
	// Query with sc=nil switches to fallback
	qRes, err := cFlightMock.Query(ctx, []uint64{10})
	if err != nil {
		t.Fatalf("Query on mock client failed: %v", err)
	}
	if len(qRes) != 0 {
		t.Errorf("expected empty results on fresh fallback, got %v", qRes)
	}
	if !cFlightMock.IsFallbackActive() {
		t.Error("expected Query to activate fallback mode when sc is nil")
	}

	// Search with sc=nil switches to fallback
	sIDs, sDists, err := cFlightMock.Search(ctx, []float32{1, 2, 3}, 5)
	if err != nil {
		t.Fatalf("Search on mock client failed: %v", err)
	}
	if len(sIDs) != 0 || len(sDists) != 0 {
		t.Errorf("expected empty search results on fresh fallback, got %v, %v", sIDs, sDists)
	}
}
