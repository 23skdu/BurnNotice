package tokenizer

import "testing"

func TestWordHashTokenizer_Tokenize(t *testing.T) {
	tok := NewWordHashTokenizer()
	out := make([]int64, 100)

	// Single word
	n := tok.Tokenize("hello", out)
	if n != 1 {
		t.Fatalf("expected 1 token, got %d", n)
	}

	// Multiple words
	n = tok.Tokenize("hello world", out)
	if n != 2 {
		t.Fatalf("expected 2 tokens, got %d", n)
	}

	// Trailing spaces
	n = tok.Tokenize("hello world   ", out)
	if n != 2 {
		t.Fatalf("expected 2 tokens, got %d", n)
	}

	// Empty string
	n = tok.Tokenize("", out)
	if n != 0 {
		t.Fatalf("expected 0 tokens, got %d", n)
	}

	// Spaces only
	n = tok.Tokenize("   ", out)
	if n != 0 {
		t.Fatalf("expected 0 tokens, got %d", n)
	}

	// Output buffer too small
	smallOut := make([]int64, 1)
	n = tok.Tokenize("hello world", smallOut)
	if n != 1 {
		t.Fatalf("expected 1 token (truncated), got %d", n)
	}
}

func TestByteTokenizer_Tokenize(t *testing.T) {
	tok := NewByteTokenizer()
	out := make([]int64, 100)

	n := tok.Tokenize("abc", out)
	if n != 3 {
		t.Fatalf("expected 3 tokens, got %d", n)
	}
	if out[0] != 'a' || out[1] != 'b' || out[2] != 'c' {
		t.Errorf("unexpected output: %v", out[:3])
	}

	smallOut := make([]int64, 2)
	n = tok.Tokenize("abcd", smallOut)
	if n != 2 {
		t.Fatalf("expected 2 tokens, got %d", n)
	}
}
