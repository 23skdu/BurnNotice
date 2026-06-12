package zjson

import (
	"testing"
)

func TestZJSON_EncodeUint(t *testing.T) {
	type S struct {
		U uint64 `json:"u"`
	}
	res, err := Marshal(S{U: 18446744073709551615})
	if err != nil {
		t.Fatal(err)
	}
	expected := `{"u":18446744073709551615}`
	if string(res) != expected {
		t.Errorf("got %s, want %s", string(res), expected)
	}
}

func TestZJSON_HexDigit_StringEscapes(t *testing.T) {
	// \x01 is < 10, should append \u0001
	// \x0b is >= 10 (value 11), should append \u000b
	res, err := Marshal("\x01\x0b")
	if err != nil {
		t.Fatal(err)
	}
	expected := `"\u0001\u000b"`
	if string(res) != expected {
		t.Errorf("got %s, want %s", string(res), expected)
	}
}

func TestZJSON_HasOmitEmpty(t *testing.T) {
	if !hasOmitEmpty("name,omitempty") {
		t.Error("expected true for name,omitempty")
	}
	if hasOmitEmpty("name") {
		t.Error("expected false for name")
	}
	if hasOmitEmpty("") {
		t.Error("expected false for empty tag")
	}
}

func TestZJSON_IsEmptyValue(t *testing.T) {
	type OmitStruct struct {
		A []int    `json:"a,omitempty"`
		B bool     `json:"b,omitempty"`
		C int      `json:"c,omitempty"`
		D uint     `json:"d,omitempty"`
		E float64  `json:"e,omitempty"`
		F *int     `json:"f,omitempty"`
	}
	// All empty values should be omitted
	res, err := Marshal(OmitStruct{})
	if err != nil {
		t.Fatal(err)
	}
	if string(res) != "{}" {
		t.Errorf("expected empty JSON {}, got %s", string(res))
	}
}

func TestZJSON_HeadersReflect(t *testing.T) {
	b := []byte("hello")
	sh := SliceHeader(&b)
	if sh.Len != 5 {
		t.Errorf("SliceHeader len = %d, want 5", sh.Len)
	}

	s := "world"
	sth := StringHeader(&s)
	if sth.Len != 5 {
		t.Errorf("StringHeader len = %d, want 5", sth.Len)
	}
}

func TestZJSON_DecoderErrors(t *testing.T) {
	var v any
	// Invalid token/syntax errors
	if err := Unmarshal([]byte("{invalid"), &v); err == nil {
		t.Error("expected error for invalid object")
	}
	if err := Unmarshal([]byte("[invalid"), &v); err == nil {
		t.Error("expected error for invalid array")
	}
	// Unexpected characters
	if err := Unmarshal([]byte("?"), &v); err == nil {
		t.Error("expected error for unexpected character")
	}
	// invalid bool
	if err := Unmarshal([]byte("trxx"), &v); err == nil {
		t.Error("expected error for invalid bool")
	}
	// invalid null
	if err := Unmarshal([]byte("nuxx"), &v); err == nil {
		t.Error("expected error for invalid null")
	}
	// invalid number
	if err := Unmarshal([]byte("-x"), &v); err == nil {
		t.Error("expected error for invalid number")
	}
	// invalid number fractional
	tokFraction := NewTokenizer([]byte("1.x"))
	if _, err := tokFraction.Next(); err == nil {
		t.Error("expected error for invalid number fraction in Tokenizer")
	}
	// invalid number exponent
	tokExponent := NewTokenizer([]byte("1e-x"))
	if _, err := tokExponent.Next(); err == nil {
		t.Error("expected error for invalid number exponent in Tokenizer")
	}
	// unterminated string
	if err := Unmarshal([]byte(`"unterminated`), &v); err == nil {
		t.Error("expected error for unterminated string")
	}
	// invalid escape sequence
	if err := Unmarshal([]byte(`"bad \x"`), &v); err == nil {
		t.Error("expected error for bad escape sequence")
	}
	// invalid unicode escape
	if err := Unmarshal([]byte(`"bad \u12"`), &v); err == nil {
		t.Error("expected error for bad unicode escape")
	}
}

func TestZJSON_Token_ParseValue(t *testing.T) {
	var v any
	input := []byte(`{"a": [true, false, null, 12.34, "str", {}]}`)
	if err := Unmarshal(input, &v); err != nil {
		t.Fatalf("failed to parse complex input: %v", err)
	}
}
