package longbow

import (
	"context"
	"fmt"
)

// Client wraps Longbow's Arrow Flight client for activation vector storage and retrieval.
type Client struct {
	addr       string
	dim        int
	metric     string
	m          int
	efConstr   int
}

// Config holds Longbow client configuration.
type Config struct {
	Addr       string
	Dim        int
	Metric     string
	M          int
	EFConstr   int
}

// NewClient creates a new Longbow client for activation storage.
func NewClient(cfg Config) *Client {
	if cfg.Metric == "" {
		cfg.Metric = "cosine"
	}
	if cfg.M == 0 {
		cfg.M = 16
	}
	if cfg.EFConstr == 0 {
		cfg.EFConstr = 100
	}
	return &Client{
		addr:     cfg.Addr,
		dim:      cfg.Dim,
		metric:   cfg.Metric,
		m:        cfg.M,
		efConstr: cfg.EFConstr,
	}
}

// Ingest streams activation vectors to Longbow via Arrow Flight DoPut.
// Each activation is a flat float32 slice of length equal to the model's hidden_dim.
func (c *Client) Ingest(ctx context.Context, activations [][]float32, ids []uint64) error {
	if len(activations) != len(ids) {
		return fmt.Errorf("activations (%d) and ids (%d) length mismatch", len(activations), len(ids))
	}
	if len(activations) == 0 {
		return nil
	}

	// TODO: Convert activations to Arrow RecordBatch and stream via DoPut.
	// This will use Longbow's Go SDK to:
	// 1. Create Arrow schema with vector (FixedSizeList[float32]) and id (uint64) columns
	// 2. Build RecordBatch from activations
	// 3. Open DoPut stream to Longbow server
	// 4. Write RecordBatch and close stream

	return nil
}

// Query retrieves activation vectors from Longbow via Arrow Flight DoGet.
func (c *Client) Query(ctx context.Context, ids []uint64) ([][]float32, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	// TODO: Query Longbow via DoGet:
	// 1. Build Arrow Flight Ticket with ID filter
	// 2. Execute DoGet and read RecordBatch stream
	// 3. Deserialize vectors into [][]float32

	return nil, nil
}

// Search performs nearest-neighbor search on stored activations using Longbow's HNSW index.
func (c *Client) Search(ctx context.Context, query []float32, k int) ([]uint64, []float32, error) {
	if len(query) != c.dim {
		return nil, nil, fmt.Errorf("query dimension %d != store dimension %d", len(query), c.dim)
	}

	// TODO: Execute HNSW search via Longbow:
	// 1. Build search request with query vector, k, ef_search
	// 2. Execute search on Longbow server (uses SIMD-optimized HNSW traversal)
	// 3. Return matched IDs and distances

	return nil, nil, nil
}

// Close releases the Longbow client connection.
func (c *Client) Close() error {
	// TODO: Close Arrow Flight gRPC connection
	return nil
}
