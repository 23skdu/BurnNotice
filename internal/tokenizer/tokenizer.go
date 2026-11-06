package tokenizer

// Tokenizer defines a zero-allocation tokenization interface.
type Tokenizer interface {
	// Tokenize extracts tokens from the input text into the pre-allocated out slice.
	// It returns the number of tokens written.
	Tokenize(text string, out []int64) int
}

// ByteTokenizer implements a simple byte-level tokenizer.
// It avoids any heap allocations by reading bytes directly from the string.
type ByteTokenizer struct{}

func NewByteTokenizer() *ByteTokenizer {
	return &ByteTokenizer{}
}

func (t *ByteTokenizer) Tokenize(text string, out []int64) int {
	n := len(text)
	if len(out) < n {
		n = len(out)
	}
	for i := 0; i < n; i++ {
		out[i] = int64(text[i])
	}
	return n
}

// WordHashTokenizer implements a simple zero-alloc whitespace-separated tokenizer
// that maps words to integer IDs using FNV-1a hash.
type WordHashTokenizer struct{}

func NewWordHashTokenizer() *WordHashTokenizer {
	return &WordHashTokenizer{}
}

func (t *WordHashTokenizer) Tokenize(text string, out []int64) int {
	var count int
	var inWord bool
	var currentHash uint64

	const fnvPrime = 1099511628211
	const fnvOffset = 14695981039346656037

	emit := func() {
		if inWord && count < len(out) {
			// Convert the hash to a positive int64
			out[count] = int64(currentHash >> 1)
			count++
		}
		inWord = false
	}

	for i := 0; i < len(text); i++ {
		c := text[i]
		if c == ' ' || c == '\n' || c == '\t' || c == '\r' {
			emit()
		} else {
			if !inWord {
				inWord = true
				currentHash = fnvOffset
			}
			currentHash ^= uint64(c)
			currentHash *= fnvPrime
		}
		if count >= len(out) {
			return count
		}
	}
	emit()

	return count
}
