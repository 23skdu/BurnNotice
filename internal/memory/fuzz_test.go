package memory

import (
	"testing"
)

func FuzzPoolGetPut(f *testing.F) {
	f.Add(1024, 64, 128)
	f.Add(100, 10, 20)
	f.Add(0, 16, 32)

	f.Fuzz(func(t *testing.T, maxUsage, size1, size2 int) {
		if size1 < 0 || size1 > 8192 || size2 < 0 || size2 > 8192 {
			t.Skip()
		}
		p := NewPool(maxUsage)
		b1 := p.Get(size1)
		if len(b1) != size1 {
			t.Fatalf("len(b1) = %d, want %d", len(b1), size1)
		}
		p.Put(b1)

		b2 := p.Get(size1)
		if len(b2) != size1 {
			t.Fatalf("len(b2) = %d, want %d", len(b2), size1)
		}
		b3 := p.Get(size2)
		if len(b3) != size2 {
			t.Fatalf("len(b3) = %d, want %d", len(b3), size2)
		}
		p.Put(b2)
		p.Put(b3)

		_ = p.ShouldGC()
		stats := p.Stats()
		if stats.BufferCount != 2 {
			t.Errorf("BufferCount = %d, want 2", stats.BufferCount)
		}
		p.Clear()
	})
}
