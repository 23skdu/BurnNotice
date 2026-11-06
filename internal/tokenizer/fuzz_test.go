package tokenizer

import (
	"testing"
)

func FuzzWordHashTokenizer(f *testing.F) {
	f.Add("hello world", 16)
	f.Add("   multiple\t\nspaces\rhere  ", 8)
	f.Add("", 4)
	f.Add("singleword", 1)

	f.Fuzz(func(t *testing.T, text string, bufLen int) {
		if len(text) > 4096 || bufLen <= 0 || bufLen > 512 {
			return
		}

		tok := NewWordHashTokenizer()
		out1 := make([]int64, bufLen)
		out2 := make([]int64, bufLen)

		n1 := tok.Tokenize(text, out1)
		n2 := tok.Tokenize(text, out2)

		if n1 != n2 {
			t.Fatalf("non-deterministic token count: %d vs %d", n1, n2)
		}
		if n1 < 0 || n1 > bufLen {
			t.Fatalf("token count %d out of bounds [0, %d]", n1, bufLen)
		}
		for i := 0; i < n1; i++ {
			if out1[i] != out2[i] {
				t.Fatalf("non-deterministic token at %d: %d vs %d", i, out1[i], out2[i])
			}
			if out1[i] < 0 {
				t.Errorf("negative token ID at %d: %d", i, out1[i])
			}
		}
	})
}

func FuzzByteTokenizer(f *testing.F) {
	f.Add("burnnotice", 16)
	f.Add("", 4)
	f.Add("truncateme", 3)

	f.Fuzz(func(t *testing.T, text string, bufLen int) {
		if len(text) > 4096 || bufLen <= 0 || bufLen > 512 {
			return
		}

		tok := NewByteTokenizer()
		out := make([]int64, bufLen)
		n := tok.Tokenize(text, out)

		want := len(text)
		if bufLen < want {
			want = bufLen
		}
		if n != want {
			t.Fatalf("ByteTokenizer count = %d, want %d", n, want)
		}
		for i := 0; i < n; i++ {
			if out[i] != int64(text[i]) {
				t.Errorf("out[%d] = %d, want %d", i, out[i], text[i])
			}
		}
	})
}
