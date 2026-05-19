package zjson

import (
	"encoding/json"
	"math"
	"reflect"
	"testing"
)

func TestMem2Str(t *testing.T) {
	b := []byte("hello")
	s := Mem2Str(b)
	if s != "hello" {
		t.Fatalf("expected 'hello', got %q", s)
	}
}

func TestStr2Mem(t *testing.T) {
	s := "world"
	b := Str2Mem(s)
	if string(b) != "world" {
		t.Fatalf("expected 'world', got %q", string(b))
	}
}

func TestCanSizeReuse(t *testing.T) {
	if !CanSizeReuse(1024, 2048) {
		t.Fatal("expected true")
	}
	if CanSizeReuse(4096, 2048) {
		t.Fatal("expected false")
	}
}

func TestGrowSlice(t *testing.T) {
	b := make([]byte, 0, 10)
	b = GrowSlice(b, 20)
	if cap(b) < 20 {
		t.Fatalf("expected cap >= 20, got %d", cap(b))
	}
	if len(b) != 20 {
		t.Fatalf("expected len 20, got %d", len(b))
	}

	// No grow needed
	b = make([]byte, 0, 30)
	b2 := GrowSlice(b, 20)
	if cap(b2) != 30 {
		t.Fatalf("expected cap 30, got %d", cap(b2))
	}
}

func TestGetPutBuf(t *testing.T) {
	b := GetBuf()
	if cap(*b) != DefaultBufSize {
		t.Fatalf("expected cap %d, got %d", DefaultBufSize, cap(*b))
	}
	*b = append(*b, make([]byte, 100)...)
	PutBuf(b, DefaultConfig.MaxBufferSize)

	// Should be reusable
	b2 := GetBuf()
	if len(*b2) != 0 {
		t.Fatal("expected empty buffer")
	}
	PutBuf(b2, DefaultConfig.MaxBufferSize)
}

func TestGetPutDigitBuf(t *testing.T) {
	b := GetDigitBuf()
	if cap(*b) != DefaultDigitBufSize {
		t.Fatalf("expected cap %d, got %d", DefaultDigitBufSize, cap(*b))
	}
	PutDigitBuf(b)
}

func TestBufferPool(t *testing.T) {
	bp := NewBufferPool(1024, 4096)
	b := bp.Get()
	if cap(*b) != 1024 {
		t.Fatalf("expected cap 1024, got %d", cap(*b))
	}
	bp.Put(b)
}

func TestGetBytesBuffer(t *testing.T) {
	buf := GetBytesBuffer()
	buf.WriteString("test")
	if buf.String() != "test" {
		t.Fatal("expected 'test'")
	}
	PutBytesBuffer(buf)
}

func TestUnmarshalString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`"hello"`, "hello"},
		{`""`, ""},
		{`"with\ttab"`, "with\ttab"},
		{`"with\nnewline"`, "with\nnewline"},
		{`"with\"quote"`, "with\"quote"},
		{`"with\\backslash"`, "with\\backslash"},
		{`"unicode \u0041"`, "unicode A"},
	}

	for _, tt := range tests {
		var s string
		if err := Unmarshal([]byte(tt.input), &s); err != nil {
			t.Errorf("Unmarshal(%q) error: %v", tt.input, err)
			continue
		}
		if s != tt.expected {
			t.Errorf("Unmarshal(%q) = %q, want %q", tt.input, s, tt.expected)
		}
	}
}

func TestUnmarshalBool(t *testing.T) {
	var b bool
	if err := Unmarshal([]byte("true"), &b); err != nil {
		t.Fatal(err)
	}
	if !b {
		t.Fatal("expected true")
	}

	if err := Unmarshal([]byte("false"), &b); err != nil {
		t.Fatal(err)
	}
	if b {
		t.Fatal("expected false")
	}
}

func TestUnmarshalNull(t *testing.T) {
	var s *string
	if err := Unmarshal([]byte("null"), &s); err != nil {
		t.Fatal(err)
	}
	if s != nil {
		t.Fatal("expected nil")
	}
}

func TestUnmarshalInt(t *testing.T) {
	tests := []struct {
		input    string
		expected int64
	}{
		{"0", 0},
		{"42", 42},
		{"-17", -17},
		{"9223372036854775807", math.MaxInt64},
	}

	for _, tt := range tests {
		var i int64
		if err := Unmarshal([]byte(tt.input), &i); err != nil {
			t.Errorf("Unmarshal(%q) error: %v", tt.input, err)
			continue
		}
		if i != tt.expected {
			t.Errorf("Unmarshal(%q) = %d, want %d", tt.input, i, tt.expected)
		}
	}
}

func TestUnmarshalFloat(t *testing.T) {
	tests := []struct {
		input    string
		expected float64
	}{
		{"0.0", 0.0},
		{"3.14159", 3.14159},
		{"-2.5e10", -2.5e10},
		{"1.23e-4", 1.23e-4},
	}

	for _, tt := range tests {
		var f float64
		if err := Unmarshal([]byte(tt.input), &f); err != nil {
			t.Errorf("Unmarshal(%q) error: %v", tt.input, err)
			continue
		}
		if math.Abs(f-tt.expected) > 1e-9 {
			t.Errorf("Unmarshal(%q) = %f, want %f", tt.input, f, tt.expected)
		}
	}
}

func TestUnmarshalSlice(t *testing.T) {
	var s []int
	if err := Unmarshal([]byte("[1,2,3]"), &s); err != nil {
		t.Fatal(err)
	}
	if len(s) != 3 || s[0] != 1 || s[1] != 2 || s[2] != 3 {
		t.Fatalf("expected [1,2,3], got %v", s)
	}

	// Empty array
	var s2 []int
	if err := Unmarshal([]byte("[]"), &s2); err != nil {
		t.Fatal(err)
	}
	if len(s2) != 0 {
		t.Fatalf("expected empty slice, got %v", s2)
	}

	// Null array
	var s3 []int
	if err := Unmarshal([]byte("null"), &s3); err != nil {
		t.Fatal(err)
	}
	if s3 != nil {
		t.Fatal("expected nil slice")
	}
}

func TestUnmarshalMap(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte(`{"a":1,"b":2}`), &m); err != nil {
		t.Fatal(err)
	}
	if m["a"] != 1 || m["b"] != 2 {
		t.Fatalf("expected {a:1,b:2}, got %v", m)
	}

	// Empty object
	var m2 map[string]int
	if err := Unmarshal([]byte("{}"), &m2); err != nil {
		t.Fatal(err)
	}
	if len(m2) != 0 {
		t.Fatalf("expected empty map, got %v", m2)
	}
}

func TestUnmarshalStruct(t *testing.T) {
	type TestStruct struct {
		Name  string `json:"name"`
		Age   int    `json:"age"`
		Skip  string `json:"-"`
		Extra string `json:"extra,omitempty"`
	}

	var s TestStruct
	if err := Unmarshal([]byte(`{"name":"Alice","age":30}`), &s); err != nil {
		t.Fatal(err)
	}
	if s.Name != "Alice" || s.Age != 30 {
		t.Fatalf("expected {Alice,30}, got %+v", s)
	}

	// With omitempty
	var s2 TestStruct
	if err := Unmarshal([]byte(`{"name":"Bob","age":25,"extra":"data"}`), &s2); err != nil {
		t.Fatal(err)
	}
	if s2.Name != "Bob" || s2.Age != 25 || s2.Extra != "data" {
		t.Fatalf("expected {Bob,25,data}, got %+v", s2)
	}
}

func TestUnmarshalNestedStruct(t *testing.T) {
	type Address struct {
		City string `json:"city"`
	}
	type Person struct {
		Name    string  `json:"name"`
		Address Address `json:"address"`
	}

	var p Person
	if err := Unmarshal([]byte(`{"name":"Alice","address":{"city":"NYC"}}`), &p); err != nil {
		t.Fatal(err)
	}
	if p.Name != "Alice" || p.Address.City != "NYC" {
		t.Fatalf("expected {Alice,{NYC}}, got %+v", p)
	}
}

func TestUnmarshalAny(t *testing.T) {
	result, err := UnmarshalAny([]byte(`{"name":"test","values":[1,2,3]}`))
	if err != nil {
		t.Fatal(err)
	}

	m, ok := result.(map[string]any)
	if !ok {
		t.Fatal("expected map[string]any")
	}
	if m["name"] != "test" {
		t.Fatalf("expected name=test, got %v", m["name"])
	}

	values, ok := m["values"].([]any)
	if !ok {
		t.Fatal("expected []any for values")
	}
	if len(values) != 3 {
		t.Fatalf("expected 3 values, got %d", len(values))
	}
}

func TestUnmarshalInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`{"key":"value"}`), &v); err != nil {
		t.Fatal(err)
	}
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatal("expected map[string]any")
	}
	if m["key"] != "value" {
		t.Fatalf("expected value, got %v", m["key"])
	}
}

func TestUnmarshalArray(t *testing.T) {
	var arr [3]int
	if err := Unmarshal([]byte("[10,20,30]"), &arr); err != nil {
		t.Fatal(err)
	}
	if arr[0] != 10 || arr[1] != 20 || arr[2] != 30 {
		t.Fatalf("expected [10,20,30], got %v", arr)
	}
}

func TestUnmarshalMapStringAny(t *testing.T) {
	var m map[string]any
	if err := Unmarshal([]byte(`{"a":1,"b":"two","c":true}`), &m); err != nil {
		t.Fatal(err)
	}
	if m["a"].(int64) != 1 {
		t.Fatal("expected a=1")
	}
	if m["b"] != "two" {
		t.Fatal("expected b=two")
	}
	if m["c"] != true {
		t.Fatal("expected c=true")
	}
}

func TestUnmarshalNestedArray(t *testing.T) {
	var s [][]int
	if err := Unmarshal([]byte(`[[1,2],[3,4]]`), &s); err != nil {
		t.Fatal(err)
	}
	if len(s) != 2 || len(s[0]) != 2 || s[0][0] != 1 {
		t.Fatalf("expected [[1,2],[3,4]], got %v", s)
	}
}

func TestUnmarshalStringToBytes(t *testing.T) {
	var b []byte
	if err := Unmarshal([]byte(`"hello"`), &b); err != nil {
		t.Fatal(err)
	}
	if string(b) != "hello" {
		t.Fatalf("expected 'hello', got %q", string(b))
	}
}

func TestUnmarshalErrors(t *testing.T) {
	tests := []struct {
		input string
		v     any
	}{
		{"", new(string)},
		{"{", new(map[string]int)},
		{"[", new([]int)},
		{`{"key"}`, new(map[string]int)},
		{`{"key":}`, new(map[string]int)},
		{`"unterminated`, new(string)},
		{"tru", new(bool)},
		{"nul", new(*string)},
		{"abc", new(int)},
	}

	for _, tt := range tests {
		err := Unmarshal([]byte(tt.input), tt.v)
		if err == nil {
			t.Errorf("Unmarshal(%q) expected error, got nil", tt.input)
		}
	}
}

func TestUnmarshalNil(t *testing.T) {
	err := Unmarshal([]byte(`{"a":1}`), nil)
	if err == nil {
		t.Fatal("expected error for nil target")
	}
}

func TestUnmarshalNonPointer(t *testing.T) {
	var s string
	err := Unmarshal([]byte(`"test"`), s)
	if err == nil {
		t.Fatal("expected error for non-pointer target")
	}
}

func TestMarshalString(t *testing.T) {
	s := "hello"
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `"hello"` {
		t.Fatalf("expected \"hello\", got %q", string(data))
	}
}

func TestMarshalEscapedString(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{"hello", `"hello"`},
		{"with\ttab", `"with\ttab"`},
		{"with\nnewline", `"with\nnewline"`},
		{`with"quote`, `"with\"quote"`},
		{`with\backslash`, `"with\\backslash"`},
	}

	for _, tt := range tests {
		data, err := Marshal(tt.input)
		if err != nil {
			t.Errorf("Marshal(%q) error: %v", tt.input, err)
			continue
		}
		if string(data) != tt.expected {
			t.Errorf("Marshal(%q) = %q, want %q", tt.input, string(data), tt.expected)
		}
	}
}

func TestMarshalBool(t *testing.T) {
	data, err := Marshal(true)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "true" {
		t.Fatalf("expected true, got %q", string(data))
	}

	data, err = Marshal(false)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "false" {
		t.Fatalf("expected false, got %q", string(data))
	}
}

func TestMarshalInt(t *testing.T) {
	data, err := Marshal(int64(42))
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "42" {
		t.Fatalf("expected 42, got %q", string(data))
	}
}

func TestMarshalFloat(t *testing.T) {
	data, err := Marshal(3.14)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "3.14" {
		t.Fatalf("expected 3.14, got %q", string(data))
	}

	// NaN/Inf
	data, err = Marshal(math.NaN())
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "null" {
		t.Fatalf("expected null for NaN, got %q", string(data))
	}
}

func TestMarshalSlice(t *testing.T) {
	s := []int{1, 2, 3}
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "[1,2,3]" {
		t.Fatalf("expected [1,2,3], got %q", string(data))
	}

	// Nil slice
	var nilSlice []int
	data, err = Marshal(nilSlice)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "null" {
		t.Fatalf("expected null for nil slice, got %q", string(data))
	}
}

func TestMarshalMap(t *testing.T) {
	m := map[string]int{"a": 1, "b": 2}
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	// Order may vary
	if len(data) != 13 { // {"a":1,"b":2} or {"b":2,"a":1}
		t.Fatalf("expected 13 bytes, got %d: %q", len(data), string(data))
	}
}

func TestMarshalStruct(t *testing.T) {
	type TestStruct struct {
		Name  string `json:"name"`
		Age   int    `json:"age"`
		Skip  string `json:"-"`
		Extra string `json:"extra,omitempty"`
	}

	s := TestStruct{Name: "Alice", Age: 30}
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"name":"Alice","age":30}`
	if string(data) != expected {
		t.Fatalf("expected %q, got %q", expected, string(data))
	}

	// With omitempty
	s2 := TestStruct{Name: "Bob", Age: 25, Extra: "data"}
	data, err = Marshal(s2)
	if err != nil {
		t.Fatal(err)
	}
	expected2 := `{"name":"Bob","age":25,"extra":"data"}`
	if string(data) != expected2 {
		t.Fatalf("expected %q, got %q", expected2, string(data))
	}
}

func TestMarshalNestedStruct(t *testing.T) {
	type Address struct {
		City string `json:"city"`
	}
	type Person struct {
		Name    string  `json:"name"`
		Address Address `json:"address"`
	}

	p := Person{Name: "Alice", Address: Address{City: "NYC"}}
	data, err := Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"name":"Alice","address":{"city":"NYC"}}`
	if string(data) != expected {
		t.Fatalf("expected %q, got %q", expected, string(data))
	}
}

func TestMarshalNilPointer(t *testing.T) {
	var p *int
	data, err := Marshal(p)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "null" {
		t.Fatalf("expected null, got %q", string(data))
	}
}

func TestMarshalBytes(t *testing.T) {
	b := []byte("hello")
	data, err := Marshal(b)
	if err != nil {
		t.Fatal(err)
	}
	// Should be base64 encoded
	expected := `"aGVsbG8="`
	if string(data) != expected {
		t.Fatalf("expected %q, got %q", expected, string(data))
	}
}

func TestMarshalArray(t *testing.T) {
	arr := [3]int{10, 20, 30}
	data, err := Marshal(arr)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "[10,20,30]" {
		t.Fatalf("expected [10,20,30], got %q", string(data))
	}
}

func TestMarshalEmptyStruct(t *testing.T) {
	type Empty struct{}
	data, err := Marshal(Empty{})
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "{}" {
		t.Fatalf("expected {}, got %q", string(data))
	}
}

func TestMarshalOmitEmpty(t *testing.T) {
	type TestStruct struct {
		Name  string `json:"name"`
		Extra string `json:"extra,omitempty"`
	}

	s := TestStruct{Name: "test"}
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"name":"test"}` {
		t.Fatalf("expected {\"name\":\"test\"}, got %q", string(data))
	}
}

func TestAppendJSON(t *testing.T) {
	buf := []byte("prefix:")
	result, err := AppendJSON(buf, 42)
	if err != nil {
		t.Fatal(err)
	}
	if string(result) != "prefix:42" {
		t.Fatalf("expected 'prefix:42', got %q", string(result))
	}
}

func TestEncodeToWriter(t *testing.T) {
	data, err := EncodeToWriter(map[string]int{"a": 1})
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"a":1}` {
		t.Fatalf("expected {\"a\":1}, got %q", string(data))
	}
}

func TestMarshalToBuf(t *testing.T) {
	buf := make([]byte, 0, 64)
	n, err := MarshalToBuf(42, buf)
	if err != nil {
		t.Fatal(err)
	}
	if n != 2 {
		t.Fatalf("expected 2 bytes, got %d", n)
	}
	if string(buf[:n]) != "42" {
		t.Fatalf("expected '42', got %q", string(buf[:n]))
	}
}

func TestTokenizerString(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello"`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VString {
		t.Fatalf("expected VString, got %v", state.Type)
	}
	s := tok.GetString(state, false)
	if s != "hello" {
		t.Fatalf("expected 'hello', got %q", s)
	}
}

func TestTokenizerEscapedString(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello\tworld"`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	s := tok.GetString(state, false)
	if s != "hello\tworld" {
		t.Fatalf("expected 'hello\\tworld', got %q", s)
	}
}

func TestTokenizerNumber(t *testing.T) {
	tok := NewTokenizer([]byte("42"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNumber {
		t.Fatalf("expected VNumber, got %v", state.Type)
	}
	if state.IntVal != 42 {
		t.Fatalf("expected 42, got %d", state.IntVal)
	}
}

func TestTokenizerFloat(t *testing.T) {
	tok := NewTokenizer([]byte("3.14"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNumber {
		t.Fatalf("expected VNumber, got %v", state.Type)
	}
	if math.Abs(state.FloatVal-3.14) > 1e-9 {
		t.Fatalf("expected 3.14, got %f", state.FloatVal)
	}
}

func TestTokenizerBool(t *testing.T) {
	tok := NewTokenizer([]byte("true"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VBool {
		t.Fatalf("expected VBool, got %v", state.Type)
	}
	if !state.BoolVal {
		t.Fatal("expected true")
	}
}

func TestTokenizerNull(t *testing.T) {
	tok := NewTokenizer([]byte("null"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNull {
		t.Fatalf("expected VNull, got %v", state.Type)
	}
}

func TestTokenizerObject(t *testing.T) {
	tok := NewTokenizer([]byte(`{"a":1,"b":2}`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VObject {
		t.Fatalf("expected VObject, got %v", state.Type)
	}
	if state.ElemCount != 2 {
		t.Fatalf("expected 2 elements, got %d", state.ElemCount)
	}
}

func TestTokenizerArray(t *testing.T) {
	tok := NewTokenizer([]byte("[1,2,3]"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VArray {
		t.Fatalf("expected VArray, got %v", state.Type)
	}
	if state.ElemCount != 3 {
		t.Fatalf("expected 3 elements, got %d", state.ElemCount)
	}
}

func TestTokenizerEmptyObject(t *testing.T) {
	tok := NewTokenizer([]byte("{}"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VObject {
		t.Fatalf("expected VObject, got %v", state.Type)
	}
	if state.ElemCount != 0 {
		t.Fatalf("expected 0 elements, got %d", state.ElemCount)
	}
}

func TestTokenizerEmptyArray(t *testing.T) {
	tok := NewTokenizer([]byte("[]"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VArray {
		t.Fatalf("expected VArray, got %v", state.Type)
	}
	if state.ElemCount != 0 {
		t.Fatalf("expected 0 elements, got %d", state.ElemCount)
	}
}

func TestTokenizerMultipleValues(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello" 42 true`))

	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VString {
		t.Fatalf("expected VString, got %v", state.Type)
	}

	state, err = tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNumber {
		t.Fatalf("expected VNumber, got %v", state.Type)
	}

	state, err = tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VBool {
		t.Fatalf("expected VBool, got %v", state.Type)
	}
}

func TestTokenizerUnterminatedString(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for unterminated string")
	}
}

func TestTokenizerInvalidBool(t *testing.T) {
	tok := NewTokenizer([]byte("tru"))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for invalid bool")
	}
}

func TestTokenizerInvalidNull(t *testing.T) {
	tok := NewTokenizer([]byte("nul"))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for invalid null")
	}
}

func TestTokenizerInvalidNumber(t *testing.T) {
	tok := NewTokenizer([]byte("-"))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for invalid number")
	}
}

func TestTokenizerUnexpectedChar(t *testing.T) {
	tok := NewTokenizer([]byte("abc"))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for unexpected char")
	}
}

func TestTokenizerGetObject(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello"`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	b := tok.GetBytes(state)
	if string(b) != "hello" {
		t.Fatalf("expected 'hello', got %q", string(b))
	}
}

func TestTokenizerCopyString(t *testing.T) {
	tok := NewTokenizer([]byte(`"hello"`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	s := tok.GetString(state, true)
	if s != "hello" {
		t.Fatalf("expected 'hello', got %q", s)
	}
}

func TestUnescape(t *testing.T) {
	tests := []struct {
		input    string
		expected string
	}{
		{`hello`, "hello"},
		{`hello\tworld`, "hello\tworld"},
		{`hello\nworld`, "hello\nworld"},
		{`hello\"world`, "hello\"world"},
		{`hello\\world`, "hello\\world"},
		{`hello\u0041world`, "helloAworld"},
	}

	for _, tt := range tests {
		buf := make([]byte, 0, 64)
		n := Unescape(buf, []byte(tt.input))
		result := string(buf[:n])
		if result != tt.expected {
			t.Errorf("Unescape(%q) = %q, want %q", tt.input, result, tt.expected)
		}
	}
}

func TestUnsafeString(t *testing.T) {
	b := []byte("test")
	s := UnsafeString(b)
	if s != "test" {
		t.Fatalf("expected 'test', got %q", s)
	}
}

func TestParseHex4(t *testing.T) {
	tests := []struct {
		input    string
		expected rune
	}{
		{"0041", 'A'},
		{"0061", 'a'},
		{"0030", '0'},
		{"FFFF", 0xFFFF},
	}

	for _, tt := range tests {
		r := parseHex4([]byte(tt.input))
		if r != tt.expected {
			t.Errorf("parseHex4(%q) = %d, want %d", tt.input, r, tt.expected)
		}
	}
}

func TestConfigValid(t *testing.T) {
	c := Config{MaxBufferSize: -1}
	valid := c.Valid()
	if valid.MaxBufferSize <= 0 {
		t.Fatal("expected positive MaxBufferSize")
	}
}

func TestNodeType(t *testing.T) {
	tests := []struct {
		input    string
		expected ValueType
	}{
		{`"hello"`, VString},
		{"42", VNumber},
		{"true", VBool},
		{"null", VNull},
		{"{}", VObject},
		{"[]", VArray},
	}

	for _, tt := range tests {
		n := Parse([]byte(tt.input))
		if n.Type() != tt.expected {
			t.Errorf("Parse(%q).Type() = %v, want %v", tt.input, n.Type(), tt.expected)
		}
	}
}

func TestNodeString(t *testing.T) {
	n := Parse([]byte(`"hello"`))
	s, err := n.String()
	if err != nil {
		t.Fatal(err)
	}
	if s != "hello" {
		t.Fatalf("expected 'hello', got %q", s)
	}
}

func TestNodeEscapedString(t *testing.T) {
	n := Parse([]byte(`"hello\tworld"`))
	s, err := n.String()
	if err != nil {
		t.Fatal(err)
	}
	if s != "hello\tworld" {
		t.Fatalf("expected 'hello\\tworld', got %q", s)
	}
}

func TestNodeInt(t *testing.T) {
	n := Parse([]byte("42"))
	i, err := n.Int()
	if err != nil {
		t.Fatal(err)
	}
	if i != 42 {
		t.Fatalf("expected 42, got %d", i)
	}
}

func TestNodeFloat(t *testing.T) {
	n := Parse([]byte("3.14"))
	f, err := n.Float()
	if err != nil {
		t.Fatal(err)
	}
	if math.Abs(f-3.14) > 1e-9 {
		t.Fatalf("expected 3.14, got %f", f)
	}
}

func TestNodeBool(t *testing.T) {
	n := Parse([]byte("true"))
	b, err := n.Bool()
	if err != nil {
		t.Fatal(err)
	}
	if !b {
		t.Fatal("expected true")
	}
}

func TestNodeIsNull(t *testing.T) {
	n := Parse([]byte("null"))
	if !n.IsNull() {
		t.Fatal("expected null")
	}
}

func TestNodeArrayLen(t *testing.T) {
	n := Parse([]byte("[1,2,3]"))
	if n.Len() != 3 {
		t.Fatalf("expected 3, got %d", n.Len())
	}
}

func TestNodeObjectLen(t *testing.T) {
	n := Parse([]byte(`{"a":1,"b":2}`))
	if n.Len() != 2 {
		t.Fatalf("expected 2, got %d", n.Len())
	}
}

func TestNodeArrayAt(t *testing.T) {
	n := Parse([]byte("[10,20,30]"))
	v := n.At(1)
	if v == nil {
		t.Fatal("expected node")
	}
	i, _ := v.Int()
	if i != 20 {
		t.Fatalf("expected 20, got %d", i)
	}
}

func TestNodeObjectGet(t *testing.T) {
	n := Parse([]byte(`{"name":"Alice","age":30}`))
	nameNode := n.Get("name")
	if nameNode == nil {
		t.Fatal("expected name node")
	}
	name, _ := nameNode.String()
	if name != "Alice" {
		t.Fatalf("expected 'Alice', got %q", name)
	}
}

func TestNodeKeys(t *testing.T) {
	n := Parse([]byte(`{"a":1,"b":2}`))
	keys := n.Keys()
	if len(keys) != 2 {
		t.Fatalf("expected 2 keys, got %d", len(keys))
	}
}

func TestNodeEach(t *testing.T) {
	n := Parse([]byte(`["a","b","c"]`))
	count := 0
	n.Each(func(key string, value *Node) bool {
		count++
		return true
	})
	if count != 3 {
		t.Fatalf("expected 3 iterations, got %d", count)
	}
}

func TestNodeEachObject(t *testing.T) {
	n := Parse([]byte(`{"a":1,"b":2}`))
	count := 0
	n.Each(func(key string, value *Node) bool {
		count++
		return true
	})
	if count != 2 {
		t.Fatalf("expected 2 iterations, got %d", count)
	}
}

func TestNodeRaw(t *testing.T) {
	n := Parse([]byte("42"))
	raw := n.Raw()
	if string(raw) != "42" {
		t.Fatalf("expected '42', got %q", string(raw))
	}
}

func TestNodeExists(t *testing.T) {
	n := Parse([]byte(`{"a":{"b":{"c":1}}}`))
	if !n.Exists("a", "b", "c") {
		t.Fatal("expected path to exist")
	}
	if n.Exists("a", "x") {
		t.Fatal("expected path to not exist")
	}
}

func TestNodeGetString(t *testing.T) {
	n := Parse([]byte(`{"name":"Alice"}`))
	s, ok := n.GetString("name")
	if !ok || s != "Alice" {
		t.Fatalf("expected 'Alice', got %q (ok=%v)", s, ok)
	}
}

func TestNodeGetInt(t *testing.T) {
	n := Parse([]byte(`{"age":30}`))
	i, ok := n.GetInt("age")
	if !ok || i != 30 {
		t.Fatalf("expected 30, got %d (ok=%v)", i, ok)
	}
}

func TestNodeGetFloat(t *testing.T) {
	n := Parse([]byte(`{"pi":3.14}`))
	f, ok := n.GetFloat("pi")
	if !ok || math.Abs(f-3.14) > 1e-9 {
		t.Fatalf("expected 3.14, got %f (ok=%v)", f, ok)
	}
}

func TestNodeGetBool(t *testing.T) {
	n := Parse([]byte(`{"active":true}`))
	b, ok := n.GetBool("active")
	if !ok || !b {
		t.Fatalf("expected true, got %v (ok=%v)", b, ok)
	}
}

func TestNodeNestedArray(t *testing.T) {
	n := Parse([]byte(`[[1,2],[3,4]]`))
	first := n.At(0)
	if first == nil {
		t.Fatal("expected first element")
	}
	if first.Len() != 2 {
		t.Fatalf("expected 2 elements in first array, got %d", first.Len())
	}
}

func TestNodeNestedObject(t *testing.T) {
	n := Parse([]byte(`{"outer":{"inner":"value"}}`))
	inner := n.Get("outer")
	if inner == nil {
		t.Fatal("expected outer node")
	}
	v, ok := inner.GetString("inner")
	if !ok || v != "value" {
		t.Fatalf("expected 'value', got %q (ok=%v)", v, ok)
	}
}

func TestNodeArrayIndexOutOfBounds(t *testing.T) {
	n := Parse([]byte("[1,2,3]"))
	v := n.At(10)
	if v != nil {
		t.Fatal("expected nil for out of bounds")
	}
}

func TestNodeGetNonExistentKey(t *testing.T) {
	n := Parse([]byte(`{"a":1}`))
	v := n.Get("b")
	if v != nil {
		t.Fatal("expected nil for non-existent key")
	}
}

func TestNodeTypeMismatch(t *testing.T) {
	n := Parse([]byte("42"))
	_, err := n.String()
	if err == nil {
		t.Fatal("expected error for type mismatch")
	}
}

func TestNodeEmptyParse(t *testing.T) {
	n := Parse([]byte{})
	if n.Type() != VNull {
		t.Fatalf("expected VNull for empty input, got %v", n.Type())
	}
}

func TestNodeWhitespaceOnly(t *testing.T) {
	n := Parse([]byte("   \n\t  "))
	if n.Type() != VNull {
		t.Fatalf("expected VNull for whitespace-only, got %v", n.Type())
	}
}

func TestDecodeStream(t *testing.T) {
	data := []byte(`{"a":1}{"b":2}`)
	v1, pos1, err := DecodeStream(data, 0)
	if err != nil {
		t.Fatal(err)
	}
	m1 := v1.(map[string]any)
	if m1["a"].(int64) != 1 {
		t.Fatal("expected a=1")
	}

	v2, _, err := DecodeStream(data, pos1)
	if err != nil {
		t.Fatal(err)
	}
	m2 := v2.(map[string]any)
	if m2["b"].(int64) != 2 {
		t.Fatal("expected b=2")
	}
}

func TestGetBytes(t *testing.T) {
	var target []byte
	if err := GetBytes([]byte(`"hello"`), &target); err != nil {
		t.Fatal(err)
	}
	if string(target) != "hello" {
		t.Fatalf("expected 'hello', got %q", string(target))
	}
}

func TestGetBytesBase64(t *testing.T) {
	var target []byte
	if err := GetBytes([]byte(`"aGVsbG8="`), &target); err != nil {
		t.Fatal(err)
	}
	if string(target) != "hello" {
		t.Fatalf("expected 'hello', got %q", string(target))
	}
}

func TestUnsafeDecode(t *testing.T) {
	var m map[string]int
	if err := UnsafeDecode([]byte(`{"a":1}`), &m); err != nil {
		t.Fatal(err)
	}
	if m["a"] != 1 {
		t.Fatalf("expected a=1, got %d", m["a"])
	}
}

func TestRoundTrip(t *testing.T) {
	type TestStruct struct {
		Name    string   `json:"name"`
		Age     int      `json:"age"`
		Scores  []float64 `json:"scores"`
		Active  bool     `json:"active"`
		Meta    map[string]string `json:"meta"`
	}

	original := TestStruct{
		Name:   "Alice",
		Age:    30,
		Scores: []float64{95.5, 87.3, 92.1},
		Active: true,
		Meta:   map[string]string{"role": "admin", "level": "5"},
	}

	data, err := Marshal(original)
	if err != nil {
		t.Fatal(err)
	}

	var decoded TestStruct
	if err := Unmarshal(data, &decoded); err != nil {
		t.Fatal(err)
	}

	if decoded.Name != original.Name {
		t.Fatalf("name mismatch: expected %q, got %q", original.Name, decoded.Name)
	}
	if decoded.Age != original.Age {
		t.Fatalf("age mismatch: expected %d, got %d", original.Age, decoded.Age)
	}
	if len(decoded.Scores) != len(original.Scores) {
		t.Fatalf("scores length mismatch: expected %d, got %d", len(original.Scores), len(decoded.Scores))
	}
	if decoded.Active != original.Active {
		t.Fatalf("active mismatch: expected %v, got %v", original.Active, decoded.Active)
	}
	if len(decoded.Meta) != len(original.Meta) {
		t.Fatalf("meta length mismatch: expected %d, got %d", len(original.Meta), len(decoded.Meta))
	}
}

func TestCompareWithStdlib(t *testing.T) {
	testCases := []string{
		`{"name":"test","values":[1,2,3],"nested":{"a":true,"b":null}}`,
		`[{"id":1,"tags":["a","b"]},{"id":2,"tags":["c"]}]`,
		`{"unicode":"\u0041\u0042\u0043","escaped":"line1\nline2\ttab"}`,
		`{"empty_obj":{},"empty_arr":[],"empty_str":""}`,
		`{"int_max":9223372036854775807,"float_pi":3.14159265358979}`,
	}

	for _, tc := range testCases {
		data := []byte(tc)

		// Stdlib unmarshal
		var stdlib any
		if err := json.Unmarshal(data, &stdlib); err != nil {
			t.Fatalf("stdlib Unmarshal error: %v", err)
		}

		// zjson unmarshal
		var zjson any
		if err := Unmarshal(data, &zjson); err != nil {
			t.Fatalf("zjson Unmarshal error: %v", err)
		}

		// Compare
		if !valuesEqual(stdlib, zjson) {
			t.Errorf("mismatch for %q\nstdlib: %#v\nzjson:  %#v", tc, stdlib, zjson)
		}
	}
}

func TestMarshalCompareWithStdlib(t *testing.T) {
	testCases := []any{
		map[string]any{"name": "test", "values": []int{1, 2, 3}},
		map[string]any{"nested": map[string]any{"a": true, "b": nil}},
		[]any{1, "two", true, nil},
		map[string]any{"empty_obj": map[string]any{}, "empty_arr": []any{}},
	}

	for _, tc := range testCases {
		stdlibData, err := json.Marshal(tc)
		if err != nil {
			t.Fatalf("stdlib Marshal error: %v", err)
		}

		zjsonData, err := Marshal(tc)
		if err != nil {
			t.Fatalf("zjson Marshal error: %v", err)
		}

		// Compare by unmarshaling both
		var stdlibResult, zjsonResult any
		if err := json.Unmarshal(stdlibData, &stdlibResult); err != nil {
			t.Fatalf("stdlib Unmarshal error: %v", err)
		}
		if err := Unmarshal(zjsonData, &zjsonResult); err != nil {
			t.Fatalf("zjson Unmarshal error: %v", err)
		}

		if !valuesEqual(stdlibResult, zjsonResult) {
			t.Errorf("marshal mismatch for %#v\nstdlib: %#v\nzjson:  %#v", tc, stdlibResult, zjsonResult)
		}
	}
}

func TestUnmarshalNegativeInt(t *testing.T) {
	var i int64
	if err := Unmarshal([]byte("-123"), &i); err != nil {
		t.Fatal(err)
	}
	if i != -123 {
		t.Fatalf("expected -123, got %d", i)
	}
}

func TestUnmarshalUint(t *testing.T) {
	var u uint64
	if err := Unmarshal([]byte("123"), &u); err != nil {
		t.Fatal(err)
	}
	if u != 123 {
		t.Fatalf("expected 123, got %d", u)
	}
}

func TestUnmarshalFloat32(t *testing.T) {
	var f float32
	if err := Unmarshal([]byte("3.14"), &f); err != nil {
		t.Fatal(err)
	}
	if math.Abs(float64(f)-3.14) > 1e-5 {
		t.Fatalf("expected 3.14, got %f", f)
	}
}

func TestUnmarshalInt8(t *testing.T) {
	var i int8
	if err := Unmarshal([]byte("42"), &i); err != nil {
		t.Fatal(err)
	}
	if i != 42 {
		t.Fatalf("expected 42, got %d", i)
	}
}

func TestUnmarshalUint8(t *testing.T) {
	var u uint8
	if err := Unmarshal([]byte("255"), &u); err != nil {
		t.Fatal(err)
	}
	if u != 255 {
		t.Fatalf("expected 255, got %d", u)
	}
}

func TestUnmarshalNestedMap(t *testing.T) {
	var m map[string]map[string]int
	if err := Unmarshal([]byte(`{"a":{"x":1},"b":{"y":2}}`), &m); err != nil {
		t.Fatal(err)
	}
	if m["a"]["x"] != 1 || m["b"]["y"] != 2 {
		t.Fatalf("expected nested map values, got %v", m)
	}
}

func TestUnmarshalSliceOfStructs(t *testing.T) {
	type Item struct {
		ID   int    `json:"id"`
		Name string `json:"name"`
	}

	var items []Item
	if err := Unmarshal([]byte(`[{"id":1,"name":"Alice"},{"id":2,"name":"Bob"}]`), &items); err != nil {
		t.Fatal(err)
	}
	if len(items) != 2 || items[0].Name != "Alice" || items[1].Name != "Bob" {
		t.Fatalf("expected [Alice,Bob], got %v", items)
	}
}

func TestUnmarshalMapWithEscapedKey(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte(`{"key\"with\"quotes":1}`), &m); err != nil {
		t.Fatal(err)
	}
	if m[`key"with"quotes`] != 1 {
		t.Fatalf("expected 1, got %v", m)
	}
}

func TestMarshalEmptyMap(t *testing.T) {
	m := make(map[string]int)
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "{}" {
		t.Fatalf("expected {}, got %q", string(data))
	}
}

func TestMarshalNilMap(t *testing.T) {
	var m map[string]int
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "null" {
		t.Fatalf("expected null, got %q", string(data))
	}
}

func TestMarshalEmptySlice(t *testing.T) {
	s := make([]int, 0)
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "[]" {
		t.Fatalf("expected [], got %q", string(data))
	}
}

func TestMarshalNestedSlices(t *testing.T) {
	s := [][]int{{1, 2}, {3, 4}}
	data, err := Marshal(s)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != "[[1,2],[3,4]]" {
		t.Fatalf("expected [[1,2],[3,4]], got %q", string(data))
	}
}

func TestMarshalNestedMaps(t *testing.T) {
	m := map[string]map[string]int{
		"outer": {"inner": 42},
	}
	data, err := Marshal(m)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"outer":{"inner":42}}` {
		t.Fatalf("expected {\"outer\":{\"inner\":42}}, got %q", string(data))
	}
}

func TestMarshalStructWithPointer(t *testing.T) {
	type Inner struct {
		Value int `json:"value"`
	}
	type Outer struct {
		Inner *Inner `json:"inner"`
	}

	val := 42
	o := Outer{Inner: &Inner{Value: val}}
	data, err := Marshal(o)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"inner":{"value":42}}` {
		t.Fatalf("expected {\"inner\":{\"value\":42}}, got %q", string(data))
	}

	// Nil pointer
	o2 := Outer{Inner: nil}
	data, err = Marshal(o2)
	if err != nil {
		t.Fatal(err)
	}
	if string(data) != `{"inner":null}` {
		t.Fatalf("expected {\"inner\":null}, got %q", string(data))
	}
}

func TestUnmarshalStructWithPointer(t *testing.T) {
	type Inner struct {
		Value int `json:"value"`
	}
	type Outer struct {
		Inner *Inner `json:"inner"`
	}

	var o Outer
	if err := Unmarshal([]byte(`{"inner":{"value":42}}`), &o); err != nil {
		t.Fatal(err)
	}
	if o.Inner == nil || o.Inner.Value != 42 {
		t.Fatalf("expected inner.value=42, got %+v", o)
	}
}

func TestTokenizerUnicodeEscape(t *testing.T) {
	tok := NewTokenizer([]byte(`"\u0048\u0065\u006c\u006c\u006f"`))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	s := tok.GetString(state, false)
	if s != "Hello" {
		t.Fatalf("expected 'Hello', got %q", s)
	}
}

func TestTokenizerInvalidUnicodeEscape(t *testing.T) {
	tok := NewTokenizer([]byte(`"\u00"`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for invalid unicode escape")
	}
}

func TestTokenizerInvalidEscapeSequence(t *testing.T) {
	tok := NewTokenizer([]byte(`"\q"`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for invalid escape sequence")
	}
}

func TestTokenizerObjectMissingColon(t *testing.T) {
	tok := NewTokenizer([]byte(`{"key" "value"}`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for missing colon")
	}
}

func TestTokenizerObjectMissingComma(t *testing.T) {
	tok := NewTokenizer([]byte(`{"a":1 "b":2}`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for missing comma")
	}
}

func TestTokenizerArrayMissingComma(t *testing.T) {
	tok := NewTokenizer([]byte(`[1 2]`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for missing comma")
	}
}

func TestTokenizerObjectInvalidKey(t *testing.T) {
	tok := NewTokenizer([]byte(`{123:"value"}`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for non-string key")
	}
}

func TestTokenizerUnterminatedObject(t *testing.T) {
	tok := NewTokenizer([]byte(`{"a":1`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for unterminated object")
	}
}

func TestTokenizerUnterminatedArray(t *testing.T) {
	tok := NewTokenizer([]byte(`[1,2`))
	_, err := tok.Next()
	if err == nil {
		t.Fatal("expected error for unterminated array")
	}
}

func TestTokenizerNumberWithExponent(t *testing.T) {
	tok := NewTokenizer([]byte("1.23e10"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNumber {
		t.Fatalf("expected VNumber, got %v", state.Type)
	}
}

func TestTokenizerNegativeNumber(t *testing.T) {
	tok := NewTokenizer([]byte("-42.5"))
	state, err := tok.Next()
	if err != nil {
		t.Fatal(err)
	}
	if state.Type != VNumber {
		t.Fatalf("expected VNumber, got %v", state.Type)
	}
	if math.Abs(state.FloatVal-(-42.5)) > 1e-9 {
		t.Fatalf("expected -42.5, got %f", state.FloatVal)
	}
}

func TestSyntaxError(t *testing.T) {
	err := &SyntaxError{Offset: 10, Msg: "test error"}
	expected := "zjson: test error at offset 10"
	if err.Error() != expected {
		t.Fatalf("expected %q, got %q", expected, err.Error())
	}
}

func TestUnmarshalMapWithAnyValues(t *testing.T) {
	var m map[string]any
	if err := Unmarshal([]byte(`{"str":"hello","num":42,"bool":true,"nested":{"a":1}}`), &m); err != nil {
		t.Fatal(err)
	}
	if m["str"] != "hello" {
		t.Fatalf("expected str=hello, got %v", m["str"])
	}
	if m["num"].(int64) != 42 {
		t.Fatalf("expected num=42, got %v", m["num"])
	}
	if m["bool"] != true {
		t.Fatalf("expected bool=true, got %v", m["bool"])
	}
}

func TestUnmarshalSliceOfAny(t *testing.T) {
	var s []any
	if err := Unmarshal([]byte(`[1,"two",true,null]`), &s); err != nil {
		t.Fatal(err)
	}
	if len(s) != 4 {
		t.Fatalf("expected 4 elements, got %d", len(s))
	}
	if s[0].(int64) != 1 {
		t.Fatalf("expected 1, got %v", s[0])
	}
	if s[1] != "two" {
		t.Fatalf("expected 'two', got %v", s[1])
	}
	if s[2] != true {
		t.Fatalf("expected true, got %v", s[2])
	}
	if s[3] != nil {
		t.Fatalf("expected nil, got %v", s[3])
	}
}

func TestUnmarshalEmptyString(t *testing.T) {
	err := Unmarshal([]byte(""), new(string))
	if err == nil {
		t.Fatal("expected error for empty input")
	}
}

func TestUnmarshalInvalidType(t *testing.T) {
	var ch chan int
	err := Unmarshal([]byte("42"), &ch)
	if err == nil {
		t.Fatal("expected error for invalid type")
	}
}

func TestUnmarshalInvalidNumber(t *testing.T) {
	var i int64
	err := Unmarshal([]byte("abc"), &i)
	if err == nil {
		t.Fatal("expected error for invalid number")
	}
}

func TestUnmarshalInvalidBool(t *testing.T) {
	var b bool
	err := Unmarshal([]byte("tru"), &b)
	if err == nil {
		t.Fatal("expected error for invalid bool")
	}
}

func TestUnmarshalInvalidNull(t *testing.T) {
	var s *string
	err := Unmarshal([]byte("nul"), &s)
	if err == nil {
		t.Fatal("expected error for invalid null")
	}
}

func TestUnmarshalNumberIntoString(t *testing.T) {
	var s string
	err := Unmarshal([]byte("42"), &s)
	if err == nil {
		t.Fatal("expected error for number into string")
	}
}

func TestUnmarshalStringIntoInt(t *testing.T) {
	var i int64
	err := Unmarshal([]byte(`"42"`), &i)
	if err == nil {
		t.Fatal("expected error for string into int")
	}
}

func TestUnmarshalObjectIntoSlice(t *testing.T) {
	var s []int
	err := Unmarshal([]byte(`{"a":1}`), &s)
	if err == nil {
		t.Fatal("expected error for object into slice")
	}
}

func TestUnmarshalArrayIntoMap(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte(`[1,2,3]`), &m)
	if err == nil {
		t.Fatal("expected error for array into map")
	}
}

func TestNodeStringTypeMismatch(t *testing.T) {
	n := Parse([]byte("42"))
	_, err := n.String()
	if err == nil {
		t.Fatal("expected error for string type mismatch")
	}
}

func TestNodeTypeMismatchInt(t *testing.T) {
	n := Parse([]byte(`"hello"`))
	_, err := n.Int()
	if err == nil {
		t.Fatal("expected error for int type mismatch")
	}
}

func TestNodeTypeMismatchFloat(t *testing.T) {
	n := Parse([]byte(`"hello"`))
	_, err := n.Float()
	if err == nil {
		t.Fatal("expected error for float type mismatch")
	}
}

func TestNodeTypeMismatchBool(t *testing.T) {
	n := Parse([]byte(`"hello"`))
	_, err := n.Bool()
	if err == nil {
		t.Fatal("expected error for bool type mismatch")
	}
}

func TestNodeGetInvalidPath(t *testing.T) {
	n := Parse([]byte(`{"a":1}`))
	v := n.Get("123") // invalid index for object
	if v != nil {
		t.Fatal("expected nil for invalid path")
	}
}

func TestNodeEachEarlyExit(t *testing.T) {
	n := Parse([]byte(`[1,2,3,4,5]`))
	count := 0
	n.Each(func(key string, value *Node) bool {
		count++
		return count < 3
	})
	if count != 3 {
		t.Fatalf("expected 3 iterations, got %d", count)
	}
}

func TestNodeEachObjectEarlyExit(t *testing.T) {
	n := Parse([]byte(`{"a":1,"b":2,"c":3}`))
	count := 0
	n.Each(func(key string, value *Node) bool {
		count++
		return count < 2
	})
	if count != 2 {
		t.Fatalf("expected 2 iterations, got %d", count)
	}
}

func TestMarshalUnsupportedType(t *testing.T) {
	ch := make(chan int)
	_, err := Marshal(ch)
	if err == nil {
		t.Fatal("expected error for unsupported type")
	}
}

func TestEncodeValueInvalid(t *testing.T) {
	var zero reflect.Value
	e := &encoder{buf: make([]byte, 0, 64)}
	err := e.encodeValue(zero)
	if err != nil {
		t.Fatal(err)
	}
	if string(e.buf) != "null" {
		t.Fatalf("expected null, got %q", string(e.buf))
	}
}

func TestDecodeObjectToInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`{"a":1,"b":"two"}`), &v); err != nil {
		t.Fatal(err)
	}
	m := v.(map[string]any)
	if m["a"].(int64) != 1 || m["b"] != "two" {
		t.Fatalf("expected {a:1,b:two}, got %v", m)
	}
}

func TestDecodeArrayToInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`[1,"two",true]`), &v); err != nil {
		t.Fatal(err)
	}
	s := v.([]any)
	if len(s) != 3 {
		t.Fatalf("expected 3 elements, got %d", len(s))
	}
}

func TestDecodeObjectToMapWithEscapedKey(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte(`{"key\"name":42}`), &m); err != nil {
		t.Fatal(err)
	}
	if m[`key"name`] != 42 {
		t.Fatalf("expected 42, got %v", m)
	}
}

func TestDecodeStructWithUnknownField(t *testing.T) {
	type TestStruct struct {
		Name string `json:"name"`
	}

	var s TestStruct
	if err := Unmarshal([]byte(`{"name":"Alice","unknown":123}`), &s); err != nil {
		t.Fatal(err)
	}
	if s.Name != "Alice" {
		t.Fatalf("expected Alice, got %q", s.Name)
	}
}

func TestDecodeStructWithEscapedKey(t *testing.T) {
	type TestStruct struct {
		Name string `json:"na\"me"`
	}

	var s TestStruct
	if err := Unmarshal([]byte(`{"na\"me":"Alice"}`), &s); err != nil {
		t.Fatal(err)
	}
	if s.Name != "Alice" {
		t.Fatalf("expected Alice, got %q", s.Name)
	}
}

func TestDecodeArrayToArrayWithExtra(t *testing.T) {
	var arr [2]int
	if err := Unmarshal([]byte(`[1,2,3,4]`), &arr); err != nil {
		t.Fatal(err)
	}
	if arr[0] != 1 || arr[1] != 2 {
		t.Fatalf("expected [1,2], got %v", arr)
	}
}

func TestDecodeArrayToArrayWithFewer(t *testing.T) {
	var arr [3]int
	if err := Unmarshal([]byte(`[1,2]`), &arr); err != nil {
		t.Fatal(err)
	}
	if arr[0] != 1 || arr[1] != 2 || arr[2] != 0 {
		t.Fatalf("expected [1,2,0], got %v", arr)
	}
}

func TestDecodeNullToSlice(t *testing.T) {
	var s []int
	if err := Unmarshal([]byte("null"), &s); err != nil {
		t.Fatal(err)
	}
	if s != nil {
		t.Fatal("expected nil slice")
	}
}

func TestDecodeNullToMap(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte("null"), &m); err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatal("expected nil map")
	}
}

func TestDecodeNullToInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("null"), &v); err != nil {
		t.Fatal(err)
	}
	if v != nil {
		t.Fatal("expected nil interface")
	}
}

func TestDecodeNumberIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("42"), &v); err != nil {
		t.Fatal(err)
	}
	if v.(int64) != 42 {
		t.Fatalf("expected 42, got %v", v)
	}
}

func TestDecodeFloatIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("3.14"), &v); err != nil {
		t.Fatal(err)
	}
	if v.(float64) != 3.14 {
		t.Fatalf("expected 3.14, got %v", v)
	}
}

func TestDecodeStringIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`"hello"`), &v); err != nil {
		t.Fatal(err)
	}
	if v != "hello" {
		t.Fatalf("expected hello, got %v", v)
	}
}

func TestDecodeBoolIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("true"), &v); err != nil {
		t.Fatal(err)
	}
	if v != true {
		t.Fatalf("expected true, got %v", v)
	}
}

func TestDecodeNullToNonNullable(t *testing.T) {
	var i int64
	if err := Unmarshal([]byte("null"), &i); err != nil {
		t.Fatal(err)
	}
	if i != 0 {
		t.Fatalf("expected 0, got %d", i)
	}
}

func TestDecodeStringWithEscapesIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`"hello\tworld"`), &v); err != nil {
		t.Fatal(err)
	}
	if v != "hello\tworld" {
		t.Fatalf("expected hello\tworld, got %v", v)
	}
}

func TestDecodeObjectWithEscapedKeyIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte(`{"key\"name":"value"}`), &v); err != nil {
		t.Fatal(err)
	}
	m := v.(map[string]any)
	if m[`key"name`] != "value" {
		t.Fatalf("expected value, got %v", m[`key"name`])
	}
}

func TestDecodeArrayWithEscapedString(t *testing.T) {
	var s []string
	if err := Unmarshal([]byte(`["hello\tworld","foo\nbar"]`), &s); err != nil {
		t.Fatal(err)
	}
	if len(s) != 2 || s[0] != "hello\tworld" || s[1] != "foo\nbar" {
		t.Fatalf("expected [hello\\tworld,foo\\nbar], got %v", s)
	}
}

func TestDecodeNestedObjectWithEscapedStrings(t *testing.T) {
	type Item struct {
		Name string `json:"name"`
	}
	type Container struct {
		Items []Item `json:"items"`
	}

	var c Container
	if err := Unmarshal([]byte(`{"items":[{"name":"hello\tworld"}]}`), &c); err != nil {
		t.Fatal(err)
	}
	if len(c.Items) != 1 || c.Items[0].Name != "hello\tworld" {
		t.Fatalf("expected hello\\tworld, got %q", c.Items[0].Name)
	}
}

func TestDecodeMapWithComplexValues(t *testing.T) {
	var m map[string]any
	if err := Unmarshal([]byte(`{"arr":[1,2],"obj":{"a":true}}`), &m); err != nil {
		t.Fatal(err)
	}
	arr := m["arr"].([]any)
	if len(arr) != 2 {
		t.Fatalf("expected 2 elements, got %d", len(arr))
	}
	obj := m["obj"].(map[string]any)
	if obj["a"] != true {
		t.Fatalf("expected true, got %v", obj["a"])
	}
}

func TestDecodeEmptyObjectIntoStruct(t *testing.T) {
	type TestStruct struct {
		Name string `json:"name"`
	}

	var s TestStruct
	if err := Unmarshal([]byte("{}"), &s); err != nil {
		t.Fatal(err)
	}
	if s.Name != "" {
		t.Fatalf("expected empty name, got %q", s.Name)
	}
}

func TestDecodeEmptyArrayIntoSlice(t *testing.T) {
	var s []int
	if err := Unmarshal([]byte("[]"), &s); err != nil {
		t.Fatal(err)
	}
	if len(s) != 0 {
		t.Fatalf("expected empty slice, got %v", s)
	}
}

func TestDecodeWhitespaceOnly(t *testing.T) {
	var s string
	err := Unmarshal([]byte("   \n\t  "), &s)
	if err == nil {
		t.Fatal("expected error for whitespace-only input")
	}
}

func TestDecodeUnexpectedEndInObject(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte(`{"a":`), &m)
	if err == nil {
		t.Fatal("expected error for unexpected end in object")
	}
}

func TestDecodeUnexpectedEndInArray(t *testing.T) {
	var s []int
	err := Unmarshal([]byte(`[1,`), &s)
	if err == nil {
		t.Fatal("expected error for unexpected end in array")
	}
}

func TestDecodeInvalidCharacter(t *testing.T) {
	var s string
	err := Unmarshal([]byte("@"), &s)
	if err == nil {
		t.Fatal("expected error for invalid character")
	}
}

func TestDecodeInvalidNumberFormat(t *testing.T) {
	var i int64
	err := Unmarshal([]byte(".5"), &i)
	if err == nil {
		t.Fatal("expected error for invalid number format")
	}
}

func TestDecodeInvalidExponent(t *testing.T) {
	var f float64
	err := Unmarshal([]byte("1e"), &f)
	if err == nil {
		t.Fatal("expected error for invalid exponent")
	}
}

func TestDecodeStringIntoBool(t *testing.T) {
	var b bool
	err := Unmarshal([]byte(`"true"`), &b)
	if err == nil {
		t.Fatal("expected error for string into bool")
	}
}

func TestDecodeNumberIntoBool(t *testing.T) {
	var b bool
	err := Unmarshal([]byte("1"), &b)
	if err == nil {
		t.Fatal("expected error for number into bool")
	}
}

func TestDecodeObjectIntoBool(t *testing.T) {
	var b bool
	err := Unmarshal([]byte("{}"), &b)
	if err == nil {
		t.Fatal("expected error for object into bool")
	}
}

func TestDecodeArrayIntoBool(t *testing.T) {
	var b bool
	err := Unmarshal([]byte("[]"), &b)
	if err == nil {
		t.Fatal("expected error for array into bool")
	}
}

func TestDecodeNullIntoBool(t *testing.T) {
	var b bool
	if err := Unmarshal([]byte("null"), &b); err != nil {
		t.Fatal(err)
	}
	if b != false {
		t.Fatalf("expected false, got %v", b)
	}
}

func TestDecodeStringIntoInt(t *testing.T) {
	var i int64
	err := Unmarshal([]byte(`"42"`), &i)
	if err == nil {
		t.Fatal("expected error for string into int")
	}
}

func TestDecodeObjectIntoInt(t *testing.T) {
	var i int64
	err := Unmarshal([]byte("{}"), &i)
	if err == nil {
		t.Fatal("expected error for object into int")
	}
}

func TestDecodeArrayIntoInt(t *testing.T) {
	var i int64
	err := Unmarshal([]byte("[]"), &i)
	if err == nil {
		t.Fatal("expected error for array into int")
	}
}

func TestDecodeBoolIntoInt(t *testing.T) {
	var i int64
	err := Unmarshal([]byte("true"), &i)
	if err == nil {
		t.Fatal("expected error for bool into int")
	}
}

func TestDecodeNullIntoInt(t *testing.T) {
	var i int64
	if err := Unmarshal([]byte("null"), &i); err != nil {
		t.Fatal(err)
	}
	if i != 0 {
		t.Fatalf("expected 0, got %d", i)
	}
}

func TestDecodeStringIntoFloat(t *testing.T) {
	var f float64
	err := Unmarshal([]byte(`"3.14"`), &f)
	if err == nil {
		t.Fatal("expected error for string into float")
	}
}

func TestDecodeObjectIntoFloat(t *testing.T) {
	var f float64
	err := Unmarshal([]byte("{}"), &f)
	if err == nil {
		t.Fatal("expected error for object into float")
	}
}

func TestDecodeArrayIntoFloat(t *testing.T) {
	var f float64
	err := Unmarshal([]byte("[]"), &f)
	if err == nil {
		t.Fatal("expected error for array into float")
	}
}

func TestDecodeBoolIntoFloat(t *testing.T) {
	var f float64
	err := Unmarshal([]byte("true"), &f)
	if err == nil {
		t.Fatal("expected error for bool into float")
	}
}

func TestDecodeNullIntoFloat(t *testing.T) {
	var f float64
	if err := Unmarshal([]byte("null"), &f); err != nil {
		t.Fatal(err)
	}
	if f != 0 {
		t.Fatalf("expected 0, got %f", f)
	}
}

func TestDecodeObjectIntoString(t *testing.T) {
	var s string
	err := Unmarshal([]byte("{}"), &s)
	if err == nil {
		t.Fatal("expected error for object into string")
	}
}

func TestDecodeArrayIntoString(t *testing.T) {
	var s string
	err := Unmarshal([]byte("[]"), &s)
	if err == nil {
		t.Fatal("expected error for array into string")
	}
}

func TestDecodeBoolIntoString(t *testing.T) {
	var s string
	err := Unmarshal([]byte("true"), &s)
	if err == nil {
		t.Fatal("expected error for bool into string")
	}
}

func TestDecodeNumberIntoString(t *testing.T) {
	var s string
	err := Unmarshal([]byte("42"), &s)
	if err == nil {
		t.Fatal("expected error for number into string")
	}
}

func TestDecodeNullIntoString(t *testing.T) {
	var s string
	if err := Unmarshal([]byte("null"), &s); err != nil {
		t.Fatal(err)
	}
	if s != "" {
		t.Fatalf("expected empty string, got %q", s)
	}
}

func TestDecodeSliceIntoMap(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte("[1,2,3]"), &m)
	if err == nil {
		t.Fatal("expected error for slice into map")
	}
}

func TestDecodeMapIntoSlice(t *testing.T) {
	var s []int
	err := Unmarshal([]byte(`{"a":1}`), &s)
	if err == nil {
		t.Fatal("expected error for map into slice")
	}
}

func TestDecodeStringIntoMap(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte(`"hello"`), &m)
	if err == nil {
		t.Fatal("expected error for string into map")
	}
}

func TestDecodeStringIntoSlice(t *testing.T) {
	var s []int
	err := Unmarshal([]byte(`"hello"`), &s)
	if err == nil {
		t.Fatal("expected error for string into slice")
	}
}

func TestDecodeNumberIntoMap(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte("42"), &m)
	if err == nil {
		t.Fatal("expected error for number into map")
	}
}

func TestDecodeNumberIntoSlice(t *testing.T) {
	var s []int
	err := Unmarshal([]byte("42"), &s)
	if err == nil {
		t.Fatal("expected error for number into slice")
	}
}

func TestDecodeBoolIntoMap(t *testing.T) {
	var m map[string]int
	err := Unmarshal([]byte("true"), &m)
	if err == nil {
		t.Fatal("expected error for bool into map")
	}
}

func TestDecodeBoolIntoSlice(t *testing.T) {
	var s []int
	err := Unmarshal([]byte("true"), &s)
	if err == nil {
		t.Fatal("expected error for bool into slice")
	}
}

func TestDecodeNullIntoMap(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte("null"), &m); err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatal("expected nil map")
	}
}

func TestDecodeNullIntoSlice(t *testing.T) {
	var s []int
	if err := Unmarshal([]byte("null"), &s); err != nil {
		t.Fatal(err)
	}
	if s != nil {
		t.Fatal("expected nil slice")
	}
}

func TestDecodeNullIntoInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("null"), &v); err != nil {
		t.Fatal(err)
	}
	if v != nil {
		t.Fatal("expected nil interface")
	}
}

func TestDecodeNullIntoPointer(t *testing.T) {
	var p *int
	if err := Unmarshal([]byte("null"), &p); err != nil {
		t.Fatal(err)
	}
	if p != nil {
		t.Fatal("expected nil pointer")
	}
}

func TestDecodeNullIntoNilSlice(t *testing.T) {
	var s []int
	if err := Unmarshal([]byte("null"), &s); err != nil {
		t.Fatal(err)
	}
	if s != nil {
		t.Fatal("expected nil slice")
	}
}

func TestDecodeNullIntoNilMap(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte("null"), &m); err != nil {
		t.Fatal(err)
	}
	if m != nil {
		t.Fatal("expected nil map")
	}
}

func TestDecodeNullIntoNilInterface(t *testing.T) {
	var v any
	if err := Unmarshal([]byte("null"), &v); err != nil {
		t.Fatal(err)
	}
	if v != nil {
		t.Fatal("expected nil interface")
	}
}

func TestDecodeNullIntoNilPointer(t *testing.T) {
	var p *int
	if err := Unmarshal([]byte("null"), &p); err != nil {
		t.Fatal(err)
	}
	if p != nil {
		t.Fatal("expected nil pointer")
	}
}
