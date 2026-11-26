package zjson

import (
	"testing"
)

func TestDecodeArrayBranches(t *testing.T) {
	// Empty array into nil slice: pre-alloc branch.
	var s []int
	if err := Unmarshal([]byte(`[]`), &s); err != nil || s == nil {
		t.Fatalf("empty slice: %v %v", err, s)
	}
	if len(s) != 0 {
		t.Fatalf("len = %d", len(s))
	}

	// Empty array into fixed array: early return.
	var a [3]int
	if err := Unmarshal([]byte(`[]`), &a); err != nil {
		t.Fatalf("empty fixed array: %v", err)
	}

	// Fixed array with fewer elements than capacity: early ].
	var a2 [5]int
	if err := Unmarshal([]byte(`[1,2]`), &a2); err != nil {
		t.Fatalf("short fixed array: %v", err)
	}
	if a2[0] != 1 || a2[1] != 2 {
		t.Fatalf("a2 = %v", a2)
	}

	// Fixed array with trailing elements: skipValue path.
	var a3 [2]int
	if err := Unmarshal([]byte(`[1,2,3,4]`), &a3); err != nil {
		t.Fatalf("trailing fixed array: %v", err)
	}

	// Unterminated array.
	var s2 []int
	if err := Unmarshal([]byte(`[1,2`), &s2); err == nil {
		t.Error("expected unterminated array error")
	}

	// Missing comma.
	var s3 []int
	if err := Unmarshal([]byte(`[1 2]`), &s3); err == nil {
		t.Error("expected missing comma error")
	}

	// []any via interface.
	var v any
	if err := Unmarshal([]byte(`[1, "x", true, null]`), &v); err != nil {
		t.Fatalf("interface array: %v", err)
	}
	arr := v.([]any)
	if len(arr) != 4 {
		t.Fatalf("len = %d", len(arr))
	}

	// Empty []any.
	var v2 any
	if err := Unmarshal([]byte(`[]`), &v2); err != nil {
		t.Fatalf("empty interface array: %v", err)
	}
}

func TestSkipValueBranches(t *testing.T) {
	// Unknown fields trigger skipValue for various value types.
	type rec struct {
		Name string `json:"name"`
	}
	var r rec
	cases := []string{
		`{"name":"x","skip":123}`,
		`{"name":"x","skip":"str"}`,
		`{"name":"x","skip":true}`,
		`{"name":"x","skip":null}`,
		`{"name":"x","skip":[1,2,3]}`,
		`{"name":"x","skip":{"a":1,"b":2}}`,
		`{"name":"x","skip":3.14}`,
	}
	for _, c := range cases {
		var r2 rec
		if err := Unmarshal([]byte(c), &r2); err != nil {
			t.Errorf("Unmarshal(%s) error: %v", c, err)
			continue
		}
		if r2.Name != "x" {
			t.Errorf("Name = %q, want x", r2.Name)
		}
		_ = r
	}

	// Unterminated nested value triggers skipValue error.
	var r3 rec
	if err := Unmarshal([]byte(`{"name":"x","skip":[1,2`), &r3); err == nil {
		t.Error("expected error for unterminated skipped array")
	}
}

func TestGetBytesExtra(t *testing.T) {
	// Unterminated input.
	var target []byte
	if err := GetBytes([]byte(""), &target); err == nil {
		t.Error("expected error for empty input")
	}

	// Unterminated string.
	if err := GetBytes([]byte(`"abc`), &target); err == nil {
		t.Error("expected error for unterminated string")
	}

	// Base64-decodable string.
	if err := GetBytes([]byte(`"aGVsbG8="`), &target); err != nil || string(target) != "hello" {
		t.Errorf("base64 GetBytes = %q, %v", string(target), err)
	}

	// Escaped string (non-base64) falls back to raw bytes.
	if err := GetBytes([]byte(`"a\\nb"`), &target); err != nil {
		t.Errorf("escaped GetBytes error: %v", err)
	}
}

func TestMarshalToBufAndAppendJSON(t *testing.T) {
	v := map[string]int{"a": 1}
	buf := make([]byte, 0, 64)
	n, err := MarshalToBuf(v, buf)
	if err != nil {
		t.Fatalf("MarshalToBuf error: %v", err)
	}
	if n <= 0 {
		t.Errorf("n = %d", n)
	}

	// AppendJSON path: append to existing buffer.
	out := []byte("prefix:")
	out, err = AppendJSON(out, v)
	if err != nil {
		t.Fatalf("AppendJSON error: %v", err)
	}
	if len(out) <= len("prefix:") {
		t.Errorf("AppendJSON did not append: %s", string(out))
	}
}
func TestSkipValueMoreBranches(t *testing.T) {
	type rec struct {
		Name string `json:"name"`
	}
	cases := []string{
		`{"name":"x","skip":"with\\escape and \u0041"}`,
		`{"name":"x","skip":[{"a":1},{"b":2}]}`,
		`{"name":"x","skip":{"nested":{"deep":true}}}`,
		`{"name":"x","skip":999999999999999999999}`,
	}
	for _, c := range cases {
		var r rec
		if err := Unmarshal([]byte(c), &r); err != nil {
			t.Errorf("Unmarshal(%s) error: %v", c, err)
		}
	}
}

func TestMarshalToBufReuse(t *testing.T) {
	v := map[string]any{"a": 1, "b": "two", "c": true}
	buf := make([]byte, 0, 4) // small capacity -> growth path
	n, err := MarshalToBuf(v, buf)
	if err != nil || n <= 0 {
		t.Fatalf("MarshalToBuf error: %v n=%d", err, n)
	}

	// Large pre-allocated buffer (reuse path).
	big := make([]byte, 0, 256)
	n2, err := MarshalToBuf(v, big)
	if err != nil || n2 <= 0 {
		t.Fatalf("MarshalToBuf big error: %v n=%d", err, n2)
	}
}

func TestAppendJSONGrowth(t *testing.T) {
	v := []int{1, 2, 3}
	out := []byte("x")
	out, err := AppendJSON(out, v)
	if err != nil {
		t.Fatalf("AppendJSON error: %v", err)
	}
	if string(out) != "x[1,2,3]" {
		t.Errorf("got %s", string(out))
	}
}

func TestTokenizerEscapes(t *testing.T) {
	in := `["a\\b\/c\bd\fe\ng\thi", "\u0041\u00e9", 42, true, false, null, {"k":"v"}]`
	tok := NewTokenizer([]byte(in))
	for tok.pos < len(tok.src) {
		st, err := tok.Next()
		if err != nil {
			break
		}
		if st.Type == VString {
			_ = tok.GetBytes(st)
		}
		_ = st
	}
}

func TestDecodeArrayToArrayTrailing(t *testing.T) {
	var a [1]int
	if err := Unmarshal([]byte(`[5, 6, 7]`), &a); err != nil {
		t.Fatalf("err: %v", err)
	}
	if a[0] != 5 {
		t.Errorf("a[0] = %d", a[0])
	}
}

func TestDecodeArrayToSliceAnyEmpty(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`[]`), &v); err != nil {
		t.Fatalf("err: %v", err)
	}
	arr := v.([]any)
	if len(arr) != 0 {
		t.Errorf("len = %d", len(arr))
	}
}
