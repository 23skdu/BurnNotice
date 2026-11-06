package memory

import "testing"

func BenchmarkPool_GetPutLarge(b *testing.B) {
	pool := NewPool(1024 * 1024)
	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		buf := pool.Get(4096)
		pool.Put(buf)
	}
}

func BenchmarkPool_ParallelGetPut(b *testing.B) {
	pool := NewPool(16 * 1024 * 1024)
	b.ReportAllocs()
	b.ResetTimer()
	b.RunParallel(func(pb *testing.PB) {
		for pb.Next() {
			buf := pool.Get(4096)
			pool.Put(buf)
		}
	})
}
