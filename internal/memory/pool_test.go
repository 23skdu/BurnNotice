package memory

import (
	"testing"
)

func TestBufferReuse(t *testing.T) {
	p := NewPool(10000)

	buf1 := p.Get(100)
	addr1 := &buf1[0]
	p.Put(buf1)

	buf2 := p.Get(100)
	addr2 := &buf2[0]

	if addr1 != addr2 {
		t.Fatal("expected same buffer address after Put/Get")
	}
}

func TestPoolStats(t *testing.T) {
	p := NewPool(10000)

	buf1 := p.Get(100)
	_ = p.Get(200)
	p.Put(buf1)

	statsAfterPut := p.Stats()
	if statsAfterPut.BufferCount != 1 {
		t.Errorf("BufferCount after Put = %d, want 1", statsAfterPut.BufferCount)
	}

	buf3 := p.Get(100)

	statsAfterGet := p.Stats()
	if statsAfterGet.TotalAlloc != 2 {
		t.Errorf("TotalAlloc = %d, want 2", statsAfterGet.TotalAlloc)
	}
	if statsAfterGet.TotalReuse != 1 {
		t.Errorf("TotalReuse = %d, want 1", statsAfterGet.TotalReuse)
	}
	if &buf1[0] != &buf3[0] {
		t.Fatal("buf1 and buf3 should be the same buffer")
	}
}

func TestShouldGC(t *testing.T) {
	p := NewPool(1000)

	for i := 0; i < 9; i++ {
		buf := p.Get(100 + i)
		p.Put(buf)
	}

	stats := p.Stats()
	if stats.BufferCount != 9 {
		t.Fatalf("BufferCount = %d, want 9", stats.BufferCount)
	}

	if !p.ShouldGC() {
		t.Fatal("ShouldGC should return true at >80% pressure")
	}
}

func TestClear(t *testing.T) {
	p := NewPool(10000)

	_ = p.Get(100)
	_ = p.Get(200)

	p.Clear()

	stats := p.Stats()
	if stats.BufferCount != 0 {
		t.Errorf("BufferCount after Clear = %d, want 0", stats.BufferCount)
	}
	if stats.CurrentUsage != 0 {
		t.Errorf("CurrentUsage after Clear = %d, want 0", stats.CurrentUsage)
	}
}

func TestDifferentSizes(t *testing.T) {
	p := NewPool(10000)

	buf1 := p.Get(100)
	buf2 := p.Get(200)

	p.Put(buf1)
	p.Put(buf2)

	buf3 := p.Get(100)
	buf4 := p.Get(200)

	if &buf3[0] == &buf4[0] {
		t.Fatal("different size buffers should not share memory")
	}
}

func TestShouldGC_ZeroMaxUsage(t *testing.T) {
	p := NewPool(0)
	buf := p.Get(100)
	p.Put(buf)
	if p.ShouldGC() {
		t.Error("ShouldGC should return false when maxUsage <= 0")
	}
}

func BenchmarkPool_GetPut(b *testing.B) {
	p := NewPool(1 << 20)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := p.Get(4096)
		p.Put(buf)
	}
}

func BenchmarkPool_ConcurrentGetPut(b *testing.B) {
	p := NewPool(1 << 22)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := p.Get(1024)
			p.Put(buf)
		}
	})
}
