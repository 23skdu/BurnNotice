package zjson

import (
	"strings"
	"testing"
)

func TestNodeBoolPaths(t *testing.T) {
	cases := []struct {
		raw     string
		want    bool
		wantErr bool
	}{
		{"true", true, false},
		{"false", false, false},
		{" \t\n\rtrue", true, false},
		{" \tfalse", false, false},
		{"truX", false, true},
		{"falsX", false, true},
		{"t", false, true},
		{"f", false, true},
	}
	for _, tc := range cases {
		got, err := Parse([]byte(tc.raw)).Bool()
		if tc.wantErr {
			if err == nil {
				t.Errorf("Bool(%q) expected error", tc.raw)
			}
			continue
		}
		if err != nil || got != tc.want {
			t.Errorf("Bool(%q) = %v, %v; want %v, nil", tc.raw, got, err, tc.want)
		}
	}

	if _, err := Parse([]byte("123")).Bool(); err == nil {
		t.Error("Bool on number should fail with type error")
	}
}

func TestNodeIntFloatErrors(t *testing.T) {
	if _, err := Parse([]byte(`"x"`)).Int(); err == nil {
		t.Error("Int on string should fail")
	}
	if _, err := Parse([]byte(`"x"`)).Float(); err == nil {
		t.Error("Float on string should fail")
	}
	if _, err := Parse([]byte("1.5")).Int(); err == nil {
		t.Error("Int on 1.5 should fail ParseInt")
	}
	if _, err := Parse([]byte("1e")).Float(); err == nil {
		t.Error("Float on 1e should fail ParseFloat")
	}
	if v, err := Parse([]byte(" \t-42")).Int(); err != nil || v != -42 {
		t.Errorf("Int whitespace = %d, %v; want -42", v, err)
	}
	if v, err := Parse([]byte(" \n2.5e1")).Float(); err != nil || v != 25 {
		t.Errorf("Float whitespace = %f, %v; want 25", v, err)
	}
}

func TestNodeStringPaths(t *testing.T) {
	if _, err := Parse([]byte("42")).String(); err == nil {
		t.Error("String on number should fail")
	}
	if v, err := Parse([]byte(`"plain"`)).String(); err != nil || v != "plain" {
		t.Errorf("String plain = %q, %v", v, err)
	}
	if v, err := Parse([]byte(`"a\nb\tc\"d\\e\/f\bg\f"`)).String(); err != nil || v != "a\nb\tc\"d\\e/f\bg\f" {
		t.Errorf("String escapes = %q, %v", v, err)
	}
	if v, err := Parse([]byte(`"\u0041\u00e9"`)).String(); err != nil || v != "Aé" {
		t.Errorf("String unicode = %q, %v", v, err)
	}
	if _, err := Parse([]byte(`"unterminated`)).String(); err == nil {
		t.Error("unterminated string should fail")
	}
}

func TestUnmarshalEscapedMapKeys(t *testing.T) {
	var m map[string]int
	if err := Unmarshal([]byte(`{"a\u0062c": 1, "plain": 2}`), &m); err != nil {
		t.Fatalf("escaped map key error: %v", err)
	}
	if m["abc"] != 1 || m["plain"] != 2 {
		t.Errorf("map = %v, want abc:1 plain:2", m)
	}
}

func TestUnmarshalEscapedIntMapKeys(t *testing.T) {
	var m map[int]int
	if err := Unmarshal([]byte(`{"\u0031": 5}`), &m); err != nil {
		t.Fatalf("escaped int key error: %v", err)
	}
	if m[1] != 5 {
		t.Errorf("map = %v, want 1:5", m)
	}

	var bad map[int]int
	if err := Unmarshal([]byte(`{"\u0061": 5}`), &bad); err == nil {
		t.Error("non-numeric escaped int key should fail")
	}
}

func TestUnmarshalEscapedUintMapKeys(t *testing.T) {
	var m map[uint8]int
	if err := Unmarshal([]byte(`{"\u0032": 7}`), &m); err != nil {
		t.Fatalf("escaped uint key error: %v", err)
	}
	if m[2] != 7 {
		t.Errorf("map = %v, want 2:7", m)
	}

	var bad map[uint8]int
	if err := Unmarshal([]byte(`{"\u0061": 7}`), &bad); err == nil {
		t.Error("non-numeric escaped uint key should fail")
	}
}

func TestUnmarshalEscapedStructKeys(t *testing.T) {
	type rec struct {
		Name string `json:"name"`
		Age  int    `json:"age"`
	}
	var r rec
	if err := Unmarshal([]byte(`{"n\u0061me": "x", "\u0061ge": 3}`), &r); err != nil {
		t.Fatalf("escaped struct key error: %v", err)
	}
	if r.Name != "x" || r.Age != 3 {
		t.Errorf("rec = %+v, want {x 3}", r)
	}

	// Unknown escaped key is skipped.
	var r2 rec
	if err := Unmarshal([]byte(`{"\u0075nknown": {"nested": [1,2]}, "name": "y"}`), &r2); err != nil {
		t.Fatalf("unknown escaped key error: %v", err)
	}
	if r2.Name != "y" {
		t.Errorf("Name = %q, want y", r2.Name)
	}
}

func TestUnmarshalAnyEscapedKeys(t *testing.T) {
	v, err := UnmarshalAny([]byte(`{"k\u0065y": true, "n": 1}`))
	if err != nil {
		t.Fatalf("UnmarshalAny escaped key error: %v", err)
	}
	m, ok := v.(map[string]any)
	if !ok {
		t.Fatalf("type = %T, want map[string]any", v)
	}
	if m["key"] != true {
		t.Errorf("m = %v, want key:true", m)
	}
}

func TestEscapedKeySyntaxErrors(t *testing.T) {
	cases := []string{
		`{"a\u0062c" 1}`,         // missing ':'
		`{"a\u0062c": 1`,         // unterminated object
		`{"a\u0062c": 1 "b": 2}`, // missing ','
	}
	for _, src := range cases {
		var m map[string]int
		if err := Unmarshal([]byte(src), &m); err == nil {
			t.Errorf("Unmarshal(%s) expected syntax error", src)
		}
	}

	type rec struct {
		Name string `json:"name"`
	}
	structCases := []string{
		`{"n\u0061me" "x"}`,
		`{"n\u0061me": "x"`,
		`{"n\u0061me": "x" "age": 1}`,
	}
	for _, src := range structCases {
		var r rec
		if err := Unmarshal([]byte(src), &r); err == nil {
			t.Errorf("Unmarshal(%s) expected syntax error", src)
		}
	}
}

func TestUnsupportedMapKeyType(t *testing.T) {
	var m map[float64]int
	err := Unmarshal([]byte(`{"1.5": 2}`), &m)
	if err == nil || !strings.Contains(err.Error(), "map key") {
		t.Errorf("expected unsupported map key error, got %v", err)
	}
}
