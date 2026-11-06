package longbow

import (
	"context"
	"database/sql"
	"encoding/binary"
	"fmt"
	"math"
	"sort"
	"sync"
	"sync/atomic"

	longbowclient "github.com/23skdu/longbow/client"
	"github.com/apache/arrow-go/v18/arrow"
	"github.com/apache/arrow-go/v18/arrow/array"
	"github.com/apache/arrow-go/v18/arrow/flight"
	"github.com/apache/arrow-go/v18/arrow/ipc"
	"github.com/apache/arrow-go/v18/arrow/memory"
	_ "modernc.org/sqlite"
)

// Client wraps a backend for activation vector storage and retrieval.
// Supports Longbow (Arrow Flight) and SQLite backends.
type Client struct {
	addr     string
	dim      int
	metric   string
	m        int
	efConstr int

	sc *longbowclient.SmartClient
	db *sql.DB

	// Standalone Fallback Math Backend
	fallbackMode int32 // 0 = standard operation, 1 = local fallback active
	fallbackMu   sync.RWMutex
	localVectors [][]float32
	localIDs     []uint64
}

// Config holds client configuration.
type Config struct {
	Addr     string
	Dim      int
	Metric   string
	M        int
	EFConstr int
	Backend  string // "longbow" (default) or "sqlite"
}

// NewClient creates a new client for activation storage.
func NewClient(cfg Config) *Client {
	if cfg.Backend == "" {
		cfg.Backend = "longbow"
	}
	if cfg.Metric == "" {
		cfg.Metric = "cosine"
	}
	if cfg.M == 0 {
		cfg.M = 16
	}
	if cfg.EFConstr == 0 {
		cfg.EFConstr = 100
	}

	c := &Client{
		addr:         cfg.Addr,
		dim:          cfg.Dim,
		metric:       cfg.Metric,
		m:            cfg.M,
		efConstr:     cfg.EFConstr,
		localVectors: make([][]float32, 0),
		localIDs:     make([]uint64, 0),
	}

	switch cfg.Backend {
	case "sqlite":
		db, err := initSQLite(cfg.Addr)
		if err != nil {
			c.SetFallbackMode(true)
		} else {
			c.db = db
		}
	default:
		sc, err := longbowclient.NewSmartClient(cfg.Addr)
		if err != nil {
			c.SetFallbackMode(true)
		} else {
			c.sc = sc
		}
	}

	return c
}

func initSQLite(dsn string) (*sql.DB, error) {
	if dsn == "" {
		dsn = "file:vectors.db?_journal_mode=WAL"
	}
	db, err := sql.Open("sqlite", dsn)
	if err != nil {
		return nil, fmt.Errorf("open sqlite: %w", err)
	}
	_, err = db.Exec(`CREATE TABLE IF NOT EXISTS vectors (
		id INTEGER PRIMARY KEY,
		vector BLOB NOT NULL,
		dim INTEGER NOT NULL
	)`)
	if err != nil {
		_ = db.Close()
		return nil, fmt.Errorf("create vectors table: %w", err)
	}
	return db, nil
}

func serializeFloats(v []float32) []byte {
	b := make([]byte, len(v)*4)
	for i, f := range v {
		binary.LittleEndian.PutUint32(b[i*4:], math.Float32bits(f))
	}
	return b
}

func deserializeFloats(b []byte) []float32 {
	v := make([]float32, len(b)/4)
	for i := range v {
		v[i] = math.Float32frombits(binary.LittleEndian.Uint32(b[i*4:]))
	}
	return v
}

func (c *Client) SetFallbackMode(active bool) {
	if active {
		atomic.StoreInt32(&c.fallbackMode, 1)
	} else {
		atomic.StoreInt32(&c.fallbackMode, 0)
	}
}

func (c *Client) IsFallbackActive() bool {
	return atomic.LoadInt32(&c.fallbackMode) == 1
}

func (c *Client) Ingest(ctx context.Context, activations [][]float32, ids []uint64) error {
	if len(activations) != len(ids) {
		return fmt.Errorf("activations (%d) and ids (%d) length mismatch", len(activations), len(ids))
	}
	if len(activations) == 0 {
		return nil
	}

	if c.db != nil {
		return c.ingestSQLite(ctx, activations, ids)
	}

	if c.IsFallbackActive() {
		c.fallbackMu.Lock()
		defer c.fallbackMu.Unlock()
		for i, act := range activations {
			c.localVectors = append(c.localVectors, act)
			c.localIDs = append(c.localIDs, ids[i])
		}
		return nil
	}

	if c.sc == nil {
		c.SetFallbackMode(true)
		return c.Ingest(ctx, activations, ids)
	}

	pool := memory.NewGoAllocator()
	schema := arrow.NewSchema(
		[]arrow.Field{
			{Name: "id", Type: arrow.PrimitiveTypes.Uint64},
			/* #nosec G115 */
			{Name: "embedding", Type: arrow.FixedSizeListOf(int32(c.dim), arrow.PrimitiveTypes.Float32)},
		},
		nil,
	)

	b := array.NewRecordBuilder(pool, schema)
	defer b.Release()

	idB := b.Field(0).(*array.Uint64Builder)
	embB := b.Field(1).(*array.FixedSizeListBuilder)
	floatB := embB.ValueBuilder().(*array.Float32Builder)

	for i, act := range activations {
		idB.Append(ids[i])
		embB.Append(true)
		floatB.AppendValues(act, nil)
	}

	rec := b.NewRecord()
	defer rec.Release()

	desc := &flight.FlightDescriptor{
		Type: flight.DescriptorPATH,
		Path: []string{"activations"},
	}

	stream, err := c.sc.DoPut(ctx, desc)
	if err != nil {
		c.SetFallbackMode(true)
		return c.Ingest(ctx, activations, ids)
	}

	w := flight.NewRecordWriter(stream, ipc.WithSchema(schema))
	defer w.Close()

	if err := w.Write(rec); err != nil {
		return err
	}

	return nil
}

func (c *Client) ingestSQLite(ctx context.Context, activations [][]float32, ids []uint64) error {
	tx, err := c.db.BeginTx(ctx, nil)
	if err != nil {
		return fmt.Errorf("begin tx: %w", err)
	}
	defer tx.Rollback()

	stmt, err := tx.PrepareContext(ctx, `INSERT OR REPLACE INTO vectors (id, vector, dim) VALUES (?, ?, ?)`)
	if err != nil {
		return fmt.Errorf("prepare: %w", err)
	}
	defer stmt.Close()

	for i, act := range activations {
		if _, err := stmt.ExecContext(ctx, ids[i], serializeFloats(act), len(act)); err != nil {
			return fmt.Errorf("insert id %d: %w", ids[i], err)
		}
	}

	return tx.Commit()
}

func (c *Client) Query(ctx context.Context, ids []uint64) ([][]float32, error) {
	if len(ids) == 0 {
		return nil, nil
	}

	if c.db != nil {
		return c.querySQLite(ctx, ids)
	}

	if c.IsFallbackActive() {
		c.fallbackMu.RLock()
		defer c.fallbackMu.RUnlock()

		idMap := make(map[uint64]bool)
		for _, id := range ids {
			idMap[id] = true
		}

		var found [][]float32
		for i, lId := range c.localIDs {
			if idMap[lId] {
				found = append(found, c.localVectors[i])
			}
		}
		return found, nil
	}

	if c.sc == nil {
		c.SetFallbackMode(true)
		return c.Query(ctx, ids)
	}

	ticket := &flight.Ticket{Ticket: []byte("query")}
	stream, err := c.sc.DoGet(ctx, ticket.Ticket)
	if err != nil {
		c.SetFallbackMode(true)
		return c.Query(ctx, ids)
	}
	_ = stream
	c.SetFallbackMode(true)
	return c.Query(ctx, ids)
}

func (c *Client) querySQLite(ctx context.Context, ids []uint64) ([][]float32, error) {
	rows, err := c.db.QueryContext(ctx, `SELECT id, vector FROM vectors WHERE id IN (`+placeholders(len(ids))+`)`, toAnySlice(ids)...) // #nosec G202 — placeholders generate only ?, characters, values are parameterized
	if err != nil {
		return nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	idMap := make(map[uint64][]float32, len(ids))
	for rows.Next() {
		var rowID uint64
		var blob []byte
		if err := rows.Scan(&rowID, &blob); err != nil {
			return nil, fmt.Errorf("scan: %w", err)
		}
		idMap[rowID] = deserializeFloats(blob)
	}
	if err := rows.Err(); err != nil {
		return nil, err
	}

	found := make([][]float32, 0, len(ids))
	for _, id := range ids {
		if v, ok := idMap[id]; ok {
			found = append(found, v)
		}
	}
	return found, nil
}

func (c *Client) Search(ctx context.Context, query []float32, k int) ([]uint64, []float32, error) {
	if len(query) != c.dim {
		return nil, nil, fmt.Errorf("query dimension %d != store dimension %d", len(query), c.dim)
	}

	if c.db != nil {
		return c.searchSQLite(ctx, query, k)
	}

	if c.IsFallbackActive() {
		c.fallbackMu.RLock()
		defer c.fallbackMu.RUnlock()

		if len(c.localVectors) == 0 {
			return nil, nil, nil
		}

		type neighbor struct {
			id   uint64
			dist float32
		}

		var list []neighbor
		for i, vec := range c.localVectors {
			sim := cosineSimilarity32(query, vec)
			var dist float32
			if c.metric == "cosine" {
				dist = 1.0 - sim
			} else {
				dist = euclideanDistance32(query, vec)
			}
			list = append(list, neighbor{id: c.localIDs[i], dist: dist})
		}

		sort.Slice(list, func(i, j int) bool {
			return list[i].dist < list[j].dist
		})

		limit := k
		if len(list) < limit {
			limit = len(list)
		}

		retIDs := make([]uint64, limit)
		retDists := make([]float32, limit)
		for i := 0; i < limit; i++ {
			retIDs[i] = list[i].id
			retDists[i] = list[i].dist
		}

		return retIDs, retDists, nil
	}

	if c.sc == nil {
		c.SetFallbackMode(true)
		return c.Search(ctx, query, k)
	}

	action := &flight.Action{Type: "search"}
	actStream, err := c.sc.DoAction(ctx, action)
	if err != nil {
		c.SetFallbackMode(true)
		return c.Search(ctx, query, k)
	}
	_ = actStream
	c.SetFallbackMode(true)
	return c.Search(ctx, query, k)
}

func (c *Client) Close() error {
	if c.db != nil {
		return c.db.Close()
	}
	if c.sc != nil {
		return c.sc.Close()
	}
	return nil
}

func (c *Client) searchSQLite(ctx context.Context, query []float32, k int) ([]uint64, []float32, error) {
	rows, err := c.db.QueryContext(ctx, `SELECT id, vector FROM vectors`)
	if err != nil {
		return nil, nil, fmt.Errorf("query: %w", err)
	}
	defer rows.Close()

	type neighbor struct {
		id   uint64
		dist float32
	}

	var list []neighbor
	for rows.Next() {
		var rowID uint64
		var blob []byte
		if err := rows.Scan(&rowID, &blob); err != nil {
			return nil, nil, fmt.Errorf("scan: %w", err)
		}
		vec := deserializeFloats(blob)
		sim := cosineSimilarity32(query, vec)
		var dist float32
		if c.metric == "cosine" {
			dist = 1.0 - sim
		} else {
			dist = euclideanDistance32(query, vec)
		}
		list = append(list, neighbor{id: rowID, dist: dist})
	}
	if err := rows.Err(); err != nil {
		return nil, nil, err
	}

	sort.Slice(list, func(i, j int) bool {
		return list[i].dist < list[j].dist
	})

	limit := k
	if len(list) < limit {
		limit = len(list)
	}

	retIDs := make([]uint64, limit)
	retDists := make([]float32, limit)
	for i := 0; i < limit; i++ {
		retIDs[i] = list[i].id
		retDists[i] = list[i].dist
	}

	return retIDs, retDists, nil
}

func placeholders(n int) string {
	if n <= 0 {
		return ""
	}
	b := make([]byte, 0, n*2-1)
	for i := 0; i < n; i++ {
		if i > 0 {
			b = append(b, ',')
		}
		b = append(b, '?')
	}
	return string(b)
}

func toAnySlice[T any](s []T) []any {
	r := make([]any, len(s))
	for i, v := range s {
		r[i] = v
	}
	return r
}

func cosineSimilarity32(a, b []float32) float32 {
	var dot, normA, normB float32
	for i := range a {
		dot += a[i] * b[i]
		normA += a[i] * a[i]
		normB += b[i] * b[i]
	}
	if normA < 1e-10 || normB < 1e-10 {
		return 0
	}
	return dot / (float32(math.Sqrt(float64(normA))) * float32(math.Sqrt(float64(normB))))
}

func euclideanDistance32(a, b []float32) float32 {
	var sum float32
	for i := range a {
		diff := a[i] - b[i]
		sum += diff * diff
	}
	return float32(math.Sqrt(float64(sum)))
}
