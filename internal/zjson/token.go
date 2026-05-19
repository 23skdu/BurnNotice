package zjson

import (
	"strconv"
	"unsafe"
)

// JsonState holds the result of parsing a single JSON value.
// Modeled after sonic's JsonState for efficient tokenization.
type JsonState struct {
	// Type is the JSON value type.
	Type ValueType

	// StrStart is the start offset of a string value (relative to input).
	StrStart int

	// StrEnd is the end offset of a string value (relative to input).
	// If no escapes were found, the string is input[StrStart:StrEnd].
	StrEnd int

	// EscapePos is the position of the first escape sequence within the string,
	// or -1 if no escapes were found (fast path).
	EscapePos int

	// NumStart is the start offset of a number value.
	NumStart int

	// NumEnd is the end offset of a number value.
	NumEnd int

	// IntVal holds the parsed integer value (for small integers).
	IntVal int64

	// FloatVal holds the parsed float value.
	FloatVal float64

	// BoolVal holds the parsed boolean value.
	BoolVal bool

	// ElemStart is the start offset of the first element in an object/array.
	ElemStart int

	// ElemCount is the number of elements in an object/array.
	ElemCount int
}

// Tokenizer parses JSON input incrementally.
type Tokenizer struct {
	// src is the input JSON bytes.
	src []byte

	// pos is the current position in the input.
	pos int

	// state holds the result of the last parsed value.
	state JsonState

	// err holds any parse error.
	err error
}

// NewTokenizer creates a new Tokenizer for the given input.
func NewTokenizer(src []byte) *Tokenizer {
	return &Tokenizer{src: src, pos: 0, state: JsonState{EscapePos: -1}}
}

// Next parses the next JSON value and returns its state.
func (t *Tokenizer) Next() (JsonState, error) {
	t.skipWhitespace()
	if t.pos >= len(t.src) {
		return JsonState{}, nil
	}

	t.state = JsonState{EscapePos: -1}

	switch t.src[t.pos] {
	case '"':
		t.parseString()
	case '{':
		t.parseObject()
	case '[':
		t.parseArray()
	case 't', 'f':
		t.parseBool()
	case 'n':
		t.parseNull()
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		t.parseNumber()
	default:
		t.err = &SyntaxError{Offset: t.pos, Msg: "unexpected character"}
	}

	return t.state, t.err
}

// skipWhitespace advances past JSON whitespace.
func (t *Tokenizer) skipWhitespace() {
	for t.pos < len(t.src) {
		switch t.src[t.pos] {
		case ' ', '\t', '\n', '\r':
			t.pos++
		default:
			return
		}
	}
}

// parseString parses a JSON string value.
// Uses SIMD-accelerated quote/escape detection.
func (t *Tokenizer) parseString() {
	t.pos++ // skip opening quote
	start := t.pos

	// Fast path: find next quote or escape using SIMD
	for t.pos < len(t.src) {
		remaining := t.src[t.pos:]
		idx := findQuoteOrEscape(remaining)

		if t.pos+idx >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unterminated string"}
			return
		}

		t.pos += idx

		if t.src[t.pos] == '"' {
			// Found closing quote
			t.state.Type = VString
			t.state.StrStart = start
			t.state.StrEnd = t.pos
			// Don't reset EscapePos - it was set when we found the backslash
			t.pos++ // skip closing quote
			return
		}

		// Found backslash - slow path for escape handling
		t.state.EscapePos = t.pos - start
		t.pos++ // skip backslash

		if t.pos >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unexpected end of string"}
			return
		}

		// Skip escaped character
		switch t.src[t.pos] {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			t.pos++
		case 'u':
			t.pos++
			if t.pos+4 > len(t.src) {
				t.err = &SyntaxError{Offset: t.pos, Msg: "invalid unicode escape"}
				return
			}
			t.pos += 4
		default:
			t.err = &SyntaxError{Offset: t.pos - 1, Msg: "invalid escape sequence"}
			return
		}
	}

	t.err = &SyntaxError{Offset: start, Msg: "unterminated string"}
}

// parseObject parses a JSON object.
func (t *Tokenizer) parseObject() {
	t.pos++ // skip '{'
	t.skipWhitespace()

	t.state.Type = VObject
	t.state.ElemStart = t.pos
	t.state.ElemCount = 0

	if t.pos < len(t.src) && t.src[t.pos] == '}' {
		t.pos++
		return
	}

	for {
		t.skipWhitespace()
		if t.pos >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unterminated object"}
			return
		}
		if t.src[t.pos] == '}' {
			t.pos++
			return
		}

		// Save container state before parsing key
		containerType := t.state.Type
		containerStart := t.state.ElemStart
		containerCount := t.state.ElemCount

		// Parse key
		if t.src[t.pos] != '"' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "expected string key"}
			return
		}
		t.parseString()
		if t.err != nil {
			return
		}

		t.skipWhitespace()
		if t.pos >= len(t.src) || t.src[t.pos] != ':' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "expected ':'"}
			return
		}
		t.pos++

		// Parse value (inline, not recursive)
		t.skipWhitespace()
		if err := t.parseValue(); err != nil {
			t.err = err
			return
		}

		// Restore container state
		t.state.Type = containerType
		t.state.ElemStart = containerStart
		t.state.ElemCount = containerCount + 1

		t.skipWhitespace()
		if t.pos >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unterminated object"}
			return
		}

		if t.src[t.pos] == '}' {
			t.pos++
			return
		}
		if t.src[t.pos] != ',' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "expected ',' or '}'"}
			return
		}
		t.pos++
	}
}

// parseArray parses a JSON array.
func (t *Tokenizer) parseArray() {
	t.pos++ // skip '['
	t.skipWhitespace()

	t.state.Type = VArray
	t.state.ElemStart = t.pos
	t.state.ElemCount = 0

	if t.pos < len(t.src) && t.src[t.pos] == ']' {
		t.pos++
		return
	}

	for {
		t.skipWhitespace()
		if t.pos >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unterminated array"}
			return
		}
		if t.src[t.pos] == ']' {
			t.pos++
			return
		}

		// Parse value (inline, not recursive)
		// Save container state before parsing value
		containerType := t.state.Type
		containerStart := t.state.ElemStart
		containerCount := t.state.ElemCount

		if err := t.parseValue(); err != nil {
			t.err = err
			return
		}

		// Restore container state
		t.state.Type = containerType
		t.state.ElemStart = containerStart
		t.state.ElemCount = containerCount + 1

		t.skipWhitespace()
		if t.pos >= len(t.src) {
			t.err = &SyntaxError{Offset: t.pos, Msg: "unterminated array"}
			return
		}

		if t.src[t.pos] == ']' {
			t.pos++
			return
		}
		if t.src[t.pos] != ',' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "expected ',' or ']"}
			return
		}
		t.pos++
	}
}

// parseValue parses a single JSON value without resetting state.
func (t *Tokenizer) parseValue() error {
	t.skipWhitespace()
	if t.pos >= len(t.src) {
		return &SyntaxError{Offset: t.pos, Msg: "unexpected end of input"}
	}

	switch t.src[t.pos] {
	case '"':
		t.parseString()
		return t.err
	case '{':
		t.parseObject()
		return t.err
	case '[':
		t.parseArray()
		return t.err
	case 't', 'f':
		t.parseBool()
		return t.err
	case 'n':
		t.parseNull()
		return t.err
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		t.parseNumber()
		return t.err
	default:
		return &SyntaxError{Offset: t.pos, Msg: "unexpected character"}
	}
}

// parseBool parses a JSON boolean value.
func (t *Tokenizer) parseBool() {
	if t.pos+4 <= len(t.src) && string(t.src[t.pos:t.pos+4]) == "true" {
		t.state.Type = VBool
		t.state.BoolVal = true
		t.pos += 4
		return
	}
	if t.pos+5 <= len(t.src) && string(t.src[t.pos:t.pos+5]) == "false" {
		t.state.Type = VBool
		t.state.BoolVal = false
		t.pos += 5
		return
	}
	t.err = &SyntaxError{Offset: t.pos, Msg: "invalid boolean"}
}

// parseNull parses a JSON null value.
func (t *Tokenizer) parseNull() {
	if t.pos+4 <= len(t.src) && string(t.src[t.pos:t.pos+4]) == "null" {
		t.state.Type = VNull
		t.pos += 4
		return
	}
	t.err = &SyntaxError{Offset: t.pos, Msg: "invalid null"}
}

// parseNumber parses a JSON number value.
func (t *Tokenizer) parseNumber() {
	start := t.pos

	// Optional minus sign
	if t.pos < len(t.src) && t.src[t.pos] == '-' {
		t.pos++
	}

	// Integer part
	if t.pos >= len(t.src) || t.src[t.pos] < '0' || t.src[t.pos] > '9' {
		t.err = &SyntaxError{Offset: t.pos, Msg: "invalid number"}
		return
	}
	if t.src[t.pos] == '0' {
		t.pos++
	} else {
		for t.pos < len(t.src) && t.src[t.pos] >= '0' && t.src[t.pos] <= '9' {
			t.pos++
		}
	}

	// Fractional part
	if t.pos < len(t.src) && t.src[t.pos] == '.' {
		t.pos++
		if t.pos >= len(t.src) || t.src[t.pos] < '0' || t.src[t.pos] > '9' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "invalid number"}
			return
		}
		for t.pos < len(t.src) && t.src[t.pos] >= '0' && t.src[t.pos] <= '9' {
			t.pos++
		}
	}

	// Exponent part
	if t.pos < len(t.src) && (t.src[t.pos] == 'e' || t.src[t.pos] == 'E') {
		t.pos++
		if t.pos < len(t.src) && (t.src[t.pos] == '+' || t.src[t.pos] == '-') {
			t.pos++
		}
		if t.pos >= len(t.src) || t.src[t.pos] < '0' || t.src[t.pos] > '9' {
			t.err = &SyntaxError{Offset: t.pos, Msg: "invalid number"}
			return
		}
		for t.pos < len(t.src) && t.src[t.pos] >= '0' && t.src[t.pos] <= '9' {
			t.pos++
		}
	}

	t.state.Type = VNumber
	t.state.NumStart = start
	t.state.NumEnd = t.pos

	// Try to parse as int64 first
	numStr := Mem2Str(t.src[start:t.pos])
	if i, err := strconv.ParseInt(numStr, 10, 64); err == nil {
		t.state.IntVal = i
		t.state.FloatVal = float64(i)
	} else {
		t.state.FloatVal, _ = strconv.ParseFloat(numStr, 64)
	}
}

// GetBytes returns the raw bytes for a string value.
// If escapes were found, this returns the raw (escaped) bytes.
func (t *Tokenizer) GetBytes(state JsonState) []byte {
	if state.StrEnd > state.StrStart {
		return t.src[state.StrStart:state.StrEnd]
	}
	return nil
}

// GetString returns the string value.
// Zero-copy if no escapes were found and CopyString is false.
func (t *Tokenizer) GetString(state JsonState, copyString bool) string {
	if state.StrEnd <= state.StrStart {
		return ""
	}

	raw := t.src[state.StrStart:state.StrEnd]

	if state.EscapePos >= 0 || copyString {
		// Slow path: unescape
		return unescapeString(raw)
	}

	// Fast path: zero-copy reference to input
	return Mem2Str(raw)
}

// SyntaxError represents a JSON parsing error.
type SyntaxError struct {
	Offset int
	Msg    string
}

func (e *SyntaxError) Error() string {
	return "zjson: " + e.Msg + " at offset " + strconv.Itoa(e.Offset)
}

// unescapeString handles the slow path for strings with escape sequences.
func unescapeString(raw []byte) string {
	buf := GetBuf()
	defer PutBuf(buf, DefaultConfig.MaxBufferSize)

	out := (*buf)[:0]
	i := 0

	for i < len(raw) {
		if raw[i] == '\\' {
			i++
			if i >= len(raw) {
				break
			}
			switch raw[i] {
			case '"':
				out = append(out, '"')
			case '\\':
				out = append(out, '\\')
			case '/':
				out = append(out, '/')
			case 'b':
				out = append(out, '\b')
			case 'f':
				out = append(out, '\f')
			case 'n':
				out = append(out, '\n')
			case 'r':
				out = append(out, '\r')
			case 't':
				out = append(out, '\t')
			case 'u':
				if i+4 < len(raw) {
					code := parseHex4(raw[i+1 : i+5])
					i += 4
					// Simple UTF-8 encoding
					if code < 0x80 {
						out = append(out, byte(code))
					} else if code < 0x800 {
						out = append(out, byte(0xC0|code>>6), byte(0x80|code&0x3F))
					} else {
						out = append(out, byte(0xE0|code>>12), byte(0x80|code>>6&0x3F), byte(0x80|code&0x3F))
					}
				}
			}
			i++
		} else {
			out = append(out, raw[i])
			i++
		}
	}

	result := make([]byte, len(out))
	copy(result, out)
	return Mem2Str(result)
}

// parseHex4 parses a 4-character hex string into a rune.
func parseHex4(b []byte) rune {
	var r rune
	for _, c := range b {
		r <<= 4
		switch {
		case c >= '0' && c <= '9':
			r |= rune(c - '0')
		case c >= 'a' && c <= 'f':
			r |= rune(c - 'a' + 10)
		case c >= 'A' && c <= 'F':
			r |= rune(c - 'A' + 10)
		}
	}
	return r
}

// Unescape writes the unescaped version of raw into dst.
// Returns the number of bytes written.
func Unescape(dst, raw []byte) int {
	out := dst[:0]
	i := 0

	for i < len(raw) {
		if raw[i] == '\\' && i+1 < len(raw) {
			i++
			switch raw[i] {
			case '"':
				out = append(out, '"')
			case '\\':
				out = append(out, '\\')
			case '/':
				out = append(out, '/')
			case 'b':
				out = append(out, '\b')
			case 'f':
				out = append(out, '\f')
			case 'n':
				out = append(out, '\n')
			case 'r':
				out = append(out, '\r')
			case 't':
				out = append(out, '\t')
			case 'u':
				if i+4 < len(raw) {
					code := parseHex4(raw[i+1 : i+5])
					i += 4
					if code < 0x80 {
						out = append(out, byte(code))
					} else if code < 0x800 {
						out = append(out, byte(0xC0|code>>6), byte(0x80|code&0x3F))
					} else {
						out = append(out, byte(0xE0|code>>12), byte(0x80|code>>6&0x3F), byte(0x80|code&0x3F))
					}
				}
			}
			i++
		} else {
			out = append(out, raw[i])
			i++
		}
	}

	return len(out)
}

// UnsafeString creates a string from bytes without copying.
// Only safe if the backing array is not modified.
func UnsafeString(b []byte) string {
	return *(*string)(unsafe.Pointer(&b))
}
