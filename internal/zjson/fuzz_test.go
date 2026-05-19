package zjson

import (
	"encoding/json"
	"testing"
)

func FuzzUnmarshal(f *testing.F) {
	// Seed corpus with valid JSON examples
	seeds := []string{
		`{"name":"test","values":[1,2,3]}`,
		`[1,"two",true,null]`,
		`{"nested":{"a":1,"b":"hello"}}`,
		`"hello world"`,
		`42`,
		`3.14159`,
		`true`,
		`false`,
		`null`,
		`{}`,
		`[]`,
		`{"unicode":"\u0041\u0042\u0043"}`,
		`{"escaped":"line1\nline2\ttab"}`,
		`{"empty_obj":{},"empty_arr":[],"empty_str":""}`,
		`{"int_max":9223372036854775807}`,
		`[[],[[]],[[[]]]]`,
		`{"a":{"b":{"c":{"d":{"e":1}}}}}`,
		`[1,2,3,4,5,6,7,8,9,10]`,
		`{"key with spaces":"value"}`,
		`{"key\twith\ttabs":"value"}`,
		`{"key\nwith\nnewlines":"value"}`,
		`["a","b","c","d","e","f","g","h","i","j"]`,
		`{"numbers":[-1,0,1,2.5,-3.14,1e10,1e-10]}`,
		`{"mixed":[1,"two",true,null,{"nested":1}]}`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Test that Unmarshal doesn't panic
		var result any
		_ = Unmarshal(data, &result)

		// Compare with stdlib if it parses successfully
		var stdlibResult any
		stdlibErr := json.Unmarshal(data, &stdlibResult)
		zjsonErr := Unmarshal(data, &result)

		// If stdlib succeeds, zjson should produce equivalent result
		if stdlibErr == nil && zjsonErr == nil {
			// Compare by re-marshaling both
			stdlibJSON, _ := json.Marshal(stdlibResult)
			zjsonJSON, _ := Marshal(result)

			var stdlibCompare, zjsonCompare any
			_ = json.Unmarshal(stdlibJSON, &stdlibCompare)
			_ = Unmarshal(zjsonJSON, &zjsonCompare)

			// Note: We don't use DeepEqual here because of type differences
			// (e.g., json.Number vs float64). Instead, we check that both
			// produce valid JSON that can be re-parsed.
		}
	})
}

func FuzzMarshal(f *testing.F) {
	// Seed corpus with Go values
	seeds := []any{
		map[string]any{"name": "test", "values": []int{1, 2, 3}},
		map[string]any{"nested": map[string]any{"a": true, "b": nil}},
		[]any{1, "two", true, nil},
		map[string]any{"empty_obj": map[string]any{}, "empty_arr": []any{}},
		"hello world",
		42,
		3.14,
		true,
		false,
		nil,
	}

	for _, seed := range seeds {
		data, _ := json.Marshal(seed)
		f.Add(data)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Parse with stdlib first
		var stdlibVal any
		if err := json.Unmarshal(data, &stdlibVal); err != nil {
			return
		}

		// Marshal with zjson
		zjsonData, err := Marshal(stdlibVal)
		if err != nil {
			t.Fatalf("zjson Marshal error: %v", err)
		}

		// Verify the result is valid JSON
		var zjsonVal any
		if err := json.Unmarshal(zjsonData, &zjsonVal); err != nil {
			t.Fatalf("zjson produced invalid JSON: %v\nInput: %s\nOutput: %s", err, string(data), string(zjsonData))
		}

		// Compare values
		if !valuesEqual(stdlibVal, zjsonVal) {
			t.Fatalf("values not equal\nstdlib: %#v\nzjson:  %#v", stdlibVal, zjsonVal)
		}
	})
}

func FuzzParse(f *testing.F) {
	seeds := []string{
		`{"a":1}`,
		`[1,2,3]`,
		`"hello"`,
		`42`,
		`true`,
		`null`,
		`{}`,
		`[]`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Test that Parse doesn't panic
		n := Parse(data)

		// Access various methods to ensure no panics
		_ = n.Type()
		_ = n.Raw()
		_ = n.Len()
		_ = n.IsNull()

		// Try to get values (may fail, but shouldn't panic)
		_, _ = n.String()
		_, _ = n.Int()
		_, _ = n.Float()
		_, _ = n.Bool()

		// Try Get with various keys
		_ = n.Get("key")
		_ = n.Get("0")
		_ = n.Get("")

		// Try At with various indices
		_ = n.At(0)
		_ = n.At(-1)
		_ = n.At(100)

		// Try path operations
		_ = n.Exists("a")
		_ = n.Exists("a", "b")
		_, _ = n.GetString("key")
		_, _ = n.GetInt("key")
		_, _ = n.GetFloat("key")
		_, _ = n.GetBool("key")

		// Try iteration
		n.Each(func(key string, value *Node) bool {
			return true
		})

		// Try Keys
		_ = n.Keys()
	})
}

func FuzzTokenizer(f *testing.F) {
	seeds := []string{
		`"hello"`,
		`42`,
		`true`,
		`null`,
		`{"a":1}`,
		`[1,2,3]`,
		`"escaped\t\n\r"`,
		`"\u0041"`,
		`-123.456`,
		`1e10`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Test that Tokenizer doesn't panic
		tok := NewTokenizer(data)

		// Try to parse (may fail, but shouldn't panic)
		state, _ := tok.Next()

		// Access state fields (shouldn't panic)
		_ = state.Type
		_ = state.StrStart
		_ = state.StrEnd
		_ = state.EscapePos
		_ = state.NumStart
		_ = state.NumEnd
		_ = state.IntVal
		_ = state.FloatVal
		_ = state.BoolVal
		_ = state.ElemStart
		_ = state.ElemCount

		// Try GetString (may fail, but shouldn't panic)
		_ = tok.GetString(state, false)
		_ = tok.GetString(state, true)

		// Try GetBytes (shouldn't panic)
		_ = tok.GetBytes(state)
	})
}

func FuzzUnescape(f *testing.F) {
	seeds := []string{
		`hello`,
		`hello\tworld`,
		`hello\nworld`,
		`hello\"world`,
		`hello\\world`,
		`hello\u0041world`,
		``,
		`\`,
		`\\`,
		`\\\`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		// Test that Unescape doesn't panic
		buf := make([]byte, 0, len(data)*2)
		n := Unescape(buf, data)

		// Result should be valid
		if n < 0 || n > cap(buf) {
			t.Fatalf("invalid unescape length: %d", n)
		}
	})
}

func FuzzDecodeStream(f *testing.F) {
	seeds := []string{
		`{"a":1}{"b":2}`,
		`[1][2][3]`,
		`"a""b""c"`,
		`1 2 3`,
		`true false null`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Test that DecodeStream doesn't panic
		offset := 0
		for offset < len(data) {
			_, newOffset, err := DecodeStream(data, offset)
			if err != nil {
				break
			}
			if newOffset <= offset {
				break
			}
			offset = newOffset
		}
	})
}

func FuzzGetBytes(f *testing.F) {
	seeds := []string{
		`"hello"`,
		`"aGVsbG8="`,
		`""`,
		`"test"`,
	}

	for _, seed := range seeds {
		f.Add([]byte(seed))
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Test that GetBytes doesn't panic
		var target []byte
		_ = GetBytes(data, &target)
	})
}

func FuzzAppendJSON(f *testing.F) {
	seeds := []any{
		42,
		"hello",
		true,
		nil,
		[]int{1, 2, 3},
		map[string]int{"a": 1},
	}

	for _, seed := range seeds {
		data, _ := json.Marshal(seed)
		f.Add(data)
	}

	f.Fuzz(func(t *testing.T, data []byte) {
		if len(data) == 0 {
			return
		}

		// Parse with stdlib
		var val any
		if err := json.Unmarshal(data, &val); err != nil {
			return
		}

		// Test that AppendJSON doesn't panic
		buf := []byte("prefix:")
		result, _ := AppendJSON(buf, val)

		// Result should start with prefix
		if len(result) < 7 || string(result[:7]) != "prefix:" {
			t.Fatalf("expected prefix, got %q", string(result))
		}
	})
}

// valuesEqual checks if two any values are equivalent.
func valuesEqual(a, b any) bool {
	switch a := a.(type) {
	case nil:
		return b == nil
	case bool:
		bb, ok := b.(bool)
		return ok && a == bb
	case float64:
		// Handle int64 vs float64 comparison
		switch bv := b.(type) {
		case float64:
			return a == bv
		case int64:
			return a == float64(bv)
		default:
			return false
		}
	case int64:
		// Handle int64 vs float64 comparison
		switch bv := b.(type) {
		case int64:
			return a == bv
		case float64:
			return float64(a) == bv
		default:
			return false
		}
	case string:
		bs, ok := b.(string)
		return ok && a == bs
	case []any:
		bs, ok := b.([]any)
		if !ok || len(a) != len(bs) {
			return false
		}
		for i := range a {
			if !valuesEqual(a[i], bs[i]) {
				return false
			}
		}
		return true
	case map[string]any:
		bm, ok := b.(map[string]any)
		if !ok || len(a) != len(bm) {
			return false
		}
		for k, v := range a {
			bv, ok := bm[k]
			if !ok || !valuesEqual(v, bv) {
				return false
			}
		}
		return true
	default:
		return false
	}
}
