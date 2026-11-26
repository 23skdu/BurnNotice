package zjson

import "testing"

func TestTokenizerCoverage(t *testing.T) {
	inputs := []string{
		`{"name":"Alice","age":30,"active":true,"score":3.14,"tags":["a","b"],"meta":null,"nested":{"x":1}}`,
		`[1,2,3]`,
		`"hello world"`,
		`"with \"escape\" and \u0041"`,
		`true`,
		`false`,
		`null`,
		`42`,
		`-1.5e3`,
		`[]`,
		`{}`,
	}
	for _, in := range inputs {
		tok := NewTokenizer([]byte(in))
		for tok.pos < len(tok.src) {
			st, err := tok.Next()
			if err != nil {
				break
			}
			_ = st
			if st.Type == VString {
				_ = tok.GetBytes(st)
			}
		}
	}
}
