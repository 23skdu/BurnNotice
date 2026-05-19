package zjson

import (
	"encoding/base64"
	"errors"
	"fmt"
	"reflect"
	"strconv"
)

// Unmarshal parses JSON data and stores the result in the value pointed to by v.
// Uses zero-allocation techniques where possible.
func Unmarshal(data []byte, v any) error {
	if v == nil {
		return errors.New("zjson: Unmarshal(nil)")
	}

	rv := reflect.ValueOf(v)
	if rv.Kind() != reflect.Ptr || rv.IsNil() {
		return errors.New("zjson: Unmarshal(non-pointer)")
	}

	d := &decoder{src: data, pos: 0}
	return d.decodeValue(rv.Elem())
}

// decoder holds state during recursive descent decoding.
type decoder struct {
	src []byte
	pos int
}

// decodeValue decodes a JSON value into the given reflect.Value.
func (d *decoder) decodeValue(rv reflect.Value) error {
	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unexpected end of input"}
	}

	// Check for null before dereferencing pointers
	if d.src[d.pos] == 'n' {
		if d.pos+4 <= len(d.src) && string(d.src[d.pos:d.pos+4]) == "null" {
			d.pos += 4
			switch rv.Kind() {
			case reflect.Ptr, reflect.Interface, reflect.Slice, reflect.Map:
				rv.Set(reflect.Zero(rv.Type()))
			default:
				// For non-nullable types, just leave as zero value
			}
			return nil
		}
	}

	// Handle pointer types
	for rv.Kind() == reflect.Ptr {
		if rv.IsNil() {
			rv.Set(reflect.New(rv.Type().Elem()))
		}
		rv = rv.Elem()
	}

	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unexpected end of input"}
	}

	switch d.src[d.pos] {
	case '"':
		return d.decodeString(rv)
	case '{':
		return d.decodeObject(rv)
	case '[':
		return d.decodeArray(rv)
	case 't', 'f':
		return d.decodeBool(rv)
	case 'n':
		return d.decodeNull(rv)
	case '-', '0', '1', '2', '3', '4', '5', '6', '7', '8', '9':
		return d.decodeNumber(rv)
	default:
		return &SyntaxError{Offset: d.pos, Msg: "unexpected character"}
	}
}

// skipWhitespace advances past JSON whitespace.
func (d *decoder) skipWhitespace() {
	for d.pos < len(d.src) {
		switch d.src[d.pos] {
		case ' ', '\t', '\n', '\r':
			d.pos++
		default:
			return
		}
	}
}

// decodeString decodes a JSON string into the given reflect.Value.
func (d *decoder) decodeString(rv reflect.Value) error {
	if d.src[d.pos] != '"' {
		return &SyntaxError{Offset: d.pos, Msg: "expected '\"'"}
	}
	d.pos++
	start := d.pos

	// Check for escapes
	hasEscape := false
	for d.pos < len(d.src) {
		remaining := d.src[d.pos:]
		idx := findQuoteOrEscape(remaining)

		if d.pos+idx >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated string"}
		}

		d.pos += idx

		if d.src[d.pos] == '"' {
			raw := d.src[start:d.pos]
			d.pos++ // skip closing quote

			if hasEscape {
				// Slow path: unescape
				buf := GetBuf()
				defer PutBuf(buf, DefaultConfig.MaxBufferSize)
				n := Unescape(*buf, raw)
				unescaped := (*buf)[:n]

				switch rv.Kind() {
				case reflect.String:
					rv.SetString(string(unescaped))
				case reflect.Slice:
					if rv.Type().Elem().Kind() == reflect.Uint8 {
						rv.SetBytes([]byte(unescaped))
					} else {
						return fmt.Errorf("zjson: cannot unmarshal string into %s", rv.Type())
					}
				case reflect.Interface:
					rv.Set(reflect.ValueOf(string(unescaped)))
				default:
					return fmt.Errorf("zjson: cannot unmarshal string into %s", rv.Type())
				}
				return nil
			}

			// Fast path: zero-copy string
			str := Mem2Str(raw)
			switch rv.Kind() {
			case reflect.String:
				rv.SetString(str)
			case reflect.Slice:
				if rv.Type().Elem().Kind() == reflect.Uint8 {
					rv.SetBytes(raw)
				} else {
					return fmt.Errorf("zjson: cannot unmarshal string into %s", rv.Type())
				}
			case reflect.Interface:
				rv.Set(reflect.ValueOf(str))
			default:
				return fmt.Errorf("zjson: cannot unmarshal string into %s", rv.Type())
			}
			return nil
		}

		// Found backslash
		hasEscape = true
		d.pos++ // skip backslash
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unexpected end of string"}
		}
		// Skip escaped character
		switch d.src[d.pos] {
		case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
			d.pos++
		case 'u':
			d.pos++
			if d.pos+4 > len(d.src) {
				return &SyntaxError{Offset: d.pos, Msg: "invalid unicode escape"}
			}
			d.pos += 4
		default:
			return &SyntaxError{Offset: d.pos - 1, Msg: "invalid escape sequence"}
		}
	}

	return &SyntaxError{Offset: start, Msg: "unterminated string"}
}

// decodeObject decodes a JSON object into the given reflect.Value.
func (d *decoder) decodeObject(rv reflect.Value) error {
	if d.src[d.pos] != '{' {
		return &SyntaxError{Offset: d.pos, Msg: "expected '{'"}
	}
	d.pos++
	d.skipWhitespace()

	switch rv.Kind() {
	case reflect.Map:
		return d.decodeObjectToMap(rv)
	case reflect.Struct:
		return d.decodeObjectToStruct(rv)
	case reflect.Interface:
		return d.decodeObjectToInterface(rv)
	default:
		return fmt.Errorf("zjson: cannot unmarshal object into %s", rv.Type())
	}
}

// indexOfCommaDecode finds the first comma in a string.
func indexOfCommaDecode(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			return i
		}
	}
	return -1
}

// decodeObjectToMap decodes a JSON object into a map.
func (d *decoder) decodeObjectToMap(rv reflect.Value) error {
	if rv.IsNil() {
		rv.Set(reflect.MakeMap(rv.Type()))
	}

	keyType := rv.Type().Key()
	elemType := rv.Type().Elem()

	if d.pos < len(d.src) && d.src[d.pos] == '}' {
		d.pos++
		return nil
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}

		// Parse key
		keyStart := d.pos
		if d.src[d.pos] != '"' {
			return &SyntaxError{Offset: d.pos, Msg: "expected string key"}
		}
		d.pos++
		keyRawStart := d.pos

		// Fast key parsing (no escapes expected in keys)
		for d.pos < len(d.src) {
			if d.src[d.pos] == '"' {
				break
			}
			if d.src[d.pos] == '\\' {
				// Slow path for escaped keys
				return d.decodeEscapedKey(rv, keyStart, keyType, elemType)
			}
			d.pos++
		}

		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated string key"}
		}

		keyStr := Mem2Str(d.src[keyRawStart:d.pos])
		d.pos++ // skip closing quote

		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] != ':' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
		}
		d.pos++

		// Parse value
		keyVal := reflect.New(keyType).Elem()
		keyVal.SetString(keyStr)

		elemVal := reflect.New(elemType).Elem()
		if err := d.decodeValue(elemVal); err != nil {
			return err
		}

		rv.SetMapIndex(keyVal, elemVal)

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
		}
		d.pos++
	}
}

// decodeEscapedKey handles the slow path for keys with escape sequences.
func (d *decoder) decodeEscapedKey(rv reflect.Value, keyStart int, keyType, elemType reflect.Type) error {
	// Reset and re-parse with escape handling
	d.pos = keyStart
	var keyStr string
	if err := d.decodeString(reflect.ValueOf(&keyStr).Elem()); err != nil {
		return err
	}

	d.skipWhitespace()
	if d.pos >= len(d.src) || d.src[d.pos] != ':' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
	}
	d.pos++

	keyVal := reflect.New(keyType).Elem()
	keyVal.SetString(keyStr)

	elemVal := reflect.New(elemType).Elem()
	if err := d.decodeValue(elemVal); err != nil {
		return err
	}

	rv.SetMapIndex(keyVal, elemVal)

	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
	}
	if d.src[d.pos] == '}' {
		d.pos++
		return nil
	}
	if d.src[d.pos] != ',' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
	}
	d.pos++

	return nil
}

// decodeObjectToStruct decodes a JSON object into a struct.
func (d *decoder) decodeObjectToStruct(rv reflect.Value) error {
	// Build field index for fast lookup
	typ := rv.Type()
	fieldIndex := make(map[string]int, typ.NumField())
	for i := 0; i < typ.NumField(); i++ {
		f := typ.Field(i)
		if f.PkgPath != "" {
			continue // skip unexported fields
		}
		tag := f.Tag.Get("json")
		if tag == "-" {
			continue
		}
		// Parse tag: "name" or "name,omitempty"
		name := tag
		if idx := indexOfCommaDecode(tag); idx >= 0 {
			name = tag[:idx]
		}
		if name == "" {
			name = f.Name
		}
		fieldIndex[name] = i
	}

	if d.pos < len(d.src) && d.src[d.pos] == '}' {
		d.pos++
		return nil
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}

		// Parse key
		if d.src[d.pos] != '"' {
			return &SyntaxError{Offset: d.pos, Msg: "expected string key"}
		}
		d.pos++
		keyRawStart := d.pos

		for d.pos < len(d.src) {
			if d.src[d.pos] == '"' {
				break
			}
			if d.src[d.pos] == '\\' {
				// Slow path for escaped keys
				return d.decodeEscapedStructKey(rv, fieldIndex)
			}
			d.pos++
		}

		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated string key"}
		}

		keyStr := Mem2Str(d.src[keyRawStart:d.pos])
		d.pos++ // skip closing quote

		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] != ':' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
		}
		d.pos++

		// Find field
		if idx, ok := fieldIndex[keyStr]; ok {
			fieldVal := rv.Field(idx)
			if err := d.decodeValue(fieldVal); err != nil {
				return err
			}
		} else {
			// Skip unknown field
			if err := d.skipValue(); err != nil {
				return err
			}
		}

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
		}
		d.pos++
	}
}

// decodeEscapedStructKey handles escaped keys in struct decoding.
func (d *decoder) decodeEscapedStructKey(rv reflect.Value, fieldIndex map[string]int) error {
	// Reset and re-parse
	start := d.pos
	for d.pos > 0 && d.src[d.pos-1] != '"' {
		d.pos--
	}
	d.pos-- // back to opening quote

	var keyStr string
	if err := d.decodeString(reflect.ValueOf(&keyStr).Elem()); err != nil {
		return err
	}

	d.skipWhitespace()
	if d.pos >= len(d.src) || d.src[d.pos] != ':' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
	}
	d.pos++

	if idx, ok := fieldIndex[keyStr]; ok {
		fieldVal := rv.Field(idx)
		if err := d.decodeValue(fieldVal); err != nil {
			return err
		}
	} else {
		if err := d.skipValue(); err != nil {
			return err
		}
	}

	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
	}
	if d.src[d.pos] == '}' {
		d.pos++
		return nil
	}
	if d.src[d.pos] != ',' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
	}
	d.pos++
	_ = start
	return nil
}

// decodeObjectToInterface decodes a JSON object into map[string]any.
func (d *decoder) decodeObjectToInterface(rv reflect.Value) error {
	result := make(map[string]any)
	if err := d.decodeObjectToMapAny(result); err != nil {
		return err
	}
	rv.Set(reflect.ValueOf(result))
	return nil
}

// decodeObjectToMapAny decodes into map[string]any.
func (d *decoder) decodeObjectToMapAny(result map[string]any) error {
	if d.pos < len(d.src) && d.src[d.pos] == '}' {
		d.pos++
		return nil
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}

		// Parse key
		if d.src[d.pos] != '"' {
			return &SyntaxError{Offset: d.pos, Msg: "expected string key"}
		}
		d.pos++
		keyRawStart := d.pos

		for d.pos < len(d.src) {
			if d.src[d.pos] == '"' {
				break
			}
			if d.src[d.pos] == '\\' {
				return d.decodeEscapedMapAnyKey(result)
			}
			d.pos++
		}

		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated string key"}
		}

		keyStr := Mem2Str(d.src[keyRawStart:d.pos])
		d.pos++

		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] != ':' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
		}
		d.pos++

		// Parse value
		var val any
		if err := d.decodeValueAny(&val); err != nil {
			return err
		}
		result[keyStr] = val

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
		}
		if d.src[d.pos] == '}' {
			d.pos++
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
		}
		d.pos++
	}
}

// decodeEscapedMapAnyKey handles escaped keys in map[string]any decoding.
func (d *decoder) decodeEscapedMapAnyKey(result map[string]any) error {
	// Reset and re-parse
	for d.pos > 0 && d.src[d.pos-1] != '"' {
		d.pos--
	}
	d.pos--

	var keyStr string
	if err := d.decodeString(reflect.ValueOf(&keyStr).Elem()); err != nil {
		return err
	}

	d.skipWhitespace()
	if d.pos >= len(d.src) || d.src[d.pos] != ':' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ':'"}
	}
	d.pos++

	var val any
	if err := d.decodeValueAny(&val); err != nil {
		return err
	}
	result[keyStr] = val

	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unterminated object"}
	}
	if d.src[d.pos] == '}' {
		d.pos++
		return nil
	}
	if d.src[d.pos] != ',' {
		return &SyntaxError{Offset: d.pos, Msg: "expected ',' or '}'"}
	}
	d.pos++
	return nil
}

// decodeArray decodes a JSON array into the given reflect.Value.
func (d *decoder) decodeArray(rv reflect.Value) error {
	if d.src[d.pos] != '[' {
		return &SyntaxError{Offset: d.pos, Msg: "expected '['"}
	}
	d.pos++
	d.skipWhitespace()

	switch rv.Kind() {
	case reflect.Slice:
		return d.decodeArrayToSlice(rv)
	case reflect.Array:
		return d.decodeArrayToArray(rv)
	case reflect.Interface:
		return d.decodeArrayToInterface(rv)
	default:
		return fmt.Errorf("zjson: cannot unmarshal array into %s", rv.Type())
	}
}

// decodeArrayToSlice decodes a JSON array into a slice.
func (d *decoder) decodeArrayToSlice(rv reflect.Value) error {
	elemType := rv.Type().Elem()

	if d.pos < len(d.src) && d.src[d.pos] == ']' {
		d.pos++
		if rv.IsNil() {
			rv.Set(reflect.MakeSlice(rv.Type(), 0, 0))
		}
		return nil
	}

	// Pre-allocate with reasonable capacity
	slice := reflect.MakeSlice(rv.Type(), 0, 16)

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			rv.Set(slice)
			return nil
		}

		elemVal := reflect.New(elemType).Elem()
		if err := d.decodeValue(elemVal); err != nil {
			return err
		}
		slice = reflect.Append(slice, elemVal)

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			rv.Set(slice)
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or ']"}
		}
		d.pos++
	}
}

// decodeArrayToArray decodes a JSON array into a fixed-size array.
func (d *decoder) decodeArrayToArray(rv reflect.Value) error {
	length := rv.Len()

	if d.pos < len(d.src) && d.src[d.pos] == ']' {
		d.pos++
		return nil
	}

	for i := 0; i < length; i++ {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			return nil
		}

		elemVal := rv.Index(i)
		if err := d.decodeValue(elemVal); err != nil {
			return err
		}

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or ']"}
		}
		d.pos++
	}

	// Skip remaining elements
	return d.skipValue()
}

// decodeArrayToInterface decodes a JSON array into []any.
func (d *decoder) decodeArrayToInterface(rv reflect.Value) error {
	result := make([]any, 0, 16)
	if err := d.decodeArrayToSliceAny(&result); err != nil {
		return err
	}
	rv.Set(reflect.ValueOf(result))
	return nil
}

// decodeArrayToSliceAny decodes into []any.
func (d *decoder) decodeArrayToSliceAny(result *[]any) error {
	if d.pos < len(d.src) && d.src[d.pos] == ']' {
		d.pos++
		return nil
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			return nil
		}

		var val any
		if err := d.decodeValueAny(&val); err != nil {
			return err
		}
		*result = append(*result, val)

		d.skipWhitespace()
		if d.pos >= len(d.src) {
			return &SyntaxError{Offset: d.pos, Msg: "unterminated array"}
		}
		if d.src[d.pos] == ']' {
			d.pos++
			return nil
		}
		if d.src[d.pos] != ',' {
			return &SyntaxError{Offset: d.pos, Msg: "expected ',' or ']"}
		}
		d.pos++
	}
}

// decodeBool decodes a JSON boolean.
func (d *decoder) decodeBool(rv reflect.Value) error {
	if d.pos+4 <= len(d.src) && string(d.src[d.pos:d.pos+4]) == "true" {
		d.pos += 4
		switch rv.Kind() {
		case reflect.Bool:
			rv.SetBool(true)
		case reflect.Interface:
			rv.Set(reflect.ValueOf(true))
		default:
			return fmt.Errorf("zjson: cannot unmarshal bool into %s", rv.Type())
		}
		return nil
	}
	if d.pos+5 <= len(d.src) && string(d.src[d.pos:d.pos+5]) == "false" {
		d.pos += 5
		switch rv.Kind() {
		case reflect.Bool:
			rv.SetBool(false)
		case reflect.Interface:
			rv.Set(reflect.ValueOf(false))
		default:
			return fmt.Errorf("zjson: cannot unmarshal bool into %s", rv.Type())
		}
		return nil
	}
	return &SyntaxError{Offset: d.pos, Msg: "invalid boolean"}
}

// decodeNull decodes a JSON null (should not be called directly since null is handled in decodeValue).
func (d *decoder) decodeNull(_ reflect.Value) error {
	// This should not be called directly since null is handled in decodeValue
	return &SyntaxError{Offset: d.pos, Msg: "unexpected null"}
}

// decodeNumber decodes a JSON number.
func (d *decoder) decodeNumber(rv reflect.Value) error {
	start := d.pos

	// Parse number bounds
	if d.src[d.pos] == '-' {
		d.pos++
	}
	if d.pos >= len(d.src) || d.src[d.pos] < '0' || d.src[d.pos] > '9' {
		return &SyntaxError{Offset: d.pos, Msg: "invalid number"}
	}
	if d.src[d.pos] == '0' {
		d.pos++
	} else {
		for d.pos < len(d.src) && d.src[d.pos] >= '0' && d.src[d.pos] <= '9' {
			d.pos++
		}
	}
	if d.pos < len(d.src) && d.src[d.pos] == '.' {
		d.pos++
		for d.pos < len(d.src) && d.src[d.pos] >= '0' && d.src[d.pos] <= '9' {
			d.pos++
		}
	}
	if d.pos < len(d.src) && (d.src[d.pos] == 'e' || d.src[d.pos] == 'E') {
		d.pos++
		if d.pos < len(d.src) && (d.src[d.pos] == '+' || d.src[d.pos] == '-') {
			d.pos++
		}
		for d.pos < len(d.src) && d.src[d.pos] >= '0' && d.src[d.pos] <= '9' {
			d.pos++
		}
	}

	numStr := Mem2Str(d.src[start:d.pos])

	switch rv.Kind() {
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		i, err := strconv.ParseInt(numStr, 10, 64)
		if err != nil {
			return fmt.Errorf("zjson: cannot unmarshal number %s into %s", numStr, rv.Type())
		}
		rv.SetInt(i)
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		u, err := strconv.ParseUint(numStr, 10, 64)
		if err != nil {
			return fmt.Errorf("zjson: cannot unmarshal number %s into %s", numStr, rv.Type())
		}
		rv.SetUint(u)
	case reflect.Float32, reflect.Float64:
		f, err := strconv.ParseFloat(numStr, 64)
		if err != nil {
			return fmt.Errorf("zjson: cannot unmarshal number %s into %s", numStr, rv.Type())
		}
		rv.SetFloat(f)
	case reflect.Interface:
		// Try int first, then float
		if i, err := strconv.ParseInt(numStr, 10, 64); err == nil {
			rv.Set(reflect.ValueOf(i))
		} else {
			f, _ := strconv.ParseFloat(numStr, 64)
			rv.Set(reflect.ValueOf(f))
		}
	default:
		return fmt.Errorf("zjson: cannot unmarshal number into %s", rv.Type())
	}

	return nil
}

// decodeValueAny decodes a JSON value into an any.
func (d *decoder) decodeValueAny(v *any) error {
	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unexpected end of input"}
	}

	switch d.src[d.pos] {
	case '"':
		d.pos++
		start := d.pos
		hasEscape := false

		for d.pos < len(d.src) {
			remaining := d.src[d.pos:]
			idx := findQuoteOrEscape(remaining)

			if d.pos+idx >= len(d.src) {
				return &SyntaxError{Offset: d.pos, Msg: "unterminated string"}
			}

			d.pos += idx

			if d.src[d.pos] == '"' {
				raw := d.src[start:d.pos]
				d.pos++

				if hasEscape {
					buf := GetBuf()
					defer PutBuf(buf, DefaultConfig.MaxBufferSize)
					n := Unescape(*buf, raw)
					*v = string((*buf)[:n])
				} else {
					*v = Mem2Str(raw)
				}
				return nil
			}

			hasEscape = true
			d.pos++
			if d.pos >= len(d.src) {
				return &SyntaxError{Offset: d.pos, Msg: "unexpected end of string"}
			}
			switch d.src[d.pos] {
			case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
				d.pos++
			case 'u':
				d.pos++
				if d.pos+4 > len(d.src) {
					return &SyntaxError{Offset: d.pos, Msg: "invalid unicode escape"}
				}
				d.pos += 4
			default:
				return &SyntaxError{Offset: d.pos - 1, Msg: "invalid escape sequence"}
			}
		}
		return &SyntaxError{Offset: start, Msg: "unterminated string"}

	case '{':
		d.pos++ // skip '{'
		result := make(map[string]any)
		if err := d.decodeObjectToMapAny(result); err != nil {
			return err
		}
		*v = result
		return nil

	case '[':
		d.pos++ // skip '['
		result := make([]any, 0, 16)
		if err := d.decodeArrayToSliceAny(&result); err != nil {
			return err
		}
		*v = result
		return nil

	case 't', 'f':
		if d.pos+4 <= len(d.src) && string(d.src[d.pos:d.pos+4]) == "true" {
			d.pos += 4
			*v = true
			return nil
		}
		if d.pos+5 <= len(d.src) && string(d.src[d.pos:d.pos+5]) == "false" {
			d.pos += 5
			*v = false
			return nil
		}
		return &SyntaxError{Offset: d.pos, Msg: "invalid boolean"}

	case 'n':
		if d.pos+4 <= len(d.src) && string(d.src[d.pos:d.pos+4]) == "null" {
			d.pos += 4
			*v = nil
			return nil
		}
		return &SyntaxError{Offset: d.pos, Msg: "invalid null"}

	default:
		// Number
		start := d.pos
		if d.src[d.pos] == '-' {
			d.pos++
		}
		for d.pos < len(d.src) && ((d.src[d.pos] >= '0' && d.src[d.pos] <= '9') || d.src[d.pos] == '.' || d.src[d.pos] == 'e' || d.src[d.pos] == 'E' || d.src[d.pos] == '+' || d.src[d.pos] == '-') {
			if d.pos > start && (d.src[d.pos] == '-' || d.src[d.pos] == '+') {
				break
			}
			d.pos++
		}

		numStr := Mem2Str(d.src[start:d.pos])
		if i, err := strconv.ParseInt(numStr, 10, 64); err == nil {
			*v = i
		} else {
			f, _ := strconv.ParseFloat(numStr, 64)
			*v = f
		}
		return nil
	}
}

// skipValue skips a JSON value without parsing it.
func (d *decoder) skipValue() error {
	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unexpected end of input"}
	}

	switch d.src[d.pos] {
	case '"':
		d.pos++
		for d.pos < len(d.src) {
			if d.src[d.pos] == '\\' {
				d.pos += 2
				continue
			}
			if d.src[d.pos] == '"' {
				d.pos++
				return nil
			}
			d.pos++
		}
		return &SyntaxError{Offset: d.pos, Msg: "unterminated string"}

	case '{':
		d.pos++
		depth := 1
		for d.pos < len(d.src) && depth > 0 {
			switch d.src[d.pos] {
			case '"':
				d.pos++
				for d.pos < len(d.src) {
					if d.src[d.pos] == '\\' {
						d.pos += 2
						continue
					}
					if d.src[d.pos] == '"' {
						d.pos++
						break
					}
					d.pos++
				}
			case '{':
				depth++
				d.pos++
			case '}':
				depth--
				d.pos++
			default:
				d.pos++
			}
		}
		return nil

	case '[':
		d.pos++
		depth := 1
		for d.pos < len(d.src) && depth > 0 {
			switch d.src[d.pos] {
			case '"':
				d.pos++
				for d.pos < len(d.src) {
					if d.src[d.pos] == '\\' {
						d.pos += 2
						continue
					}
					if d.src[d.pos] == '"' {
						d.pos++
						break
					}
					d.pos++
				}
			case '[':
				depth++
				d.pos++
			case ']':
				depth--
				d.pos++
			default:
				d.pos++
			}
		}
		return nil

	default:
		// Number, bool, null
		for d.pos < len(d.src) {
			switch d.src[d.pos] {
			case ',', '}', ']', ' ', '\t', '\n', '\r':
				return nil
			}
			d.pos++
		}
		return nil
	}
}

// UnmarshalAny parses JSON data into an any value.
func UnmarshalAny(data []byte) (any, error) {
	var result any
	d := &decoder{src: data, pos: 0}
	err := d.decodeValueAny(&result)
	return result, err
}

// GetBytes decodes a JSON string into a byte slice.
// Handles base64-encoded strings if the target is []byte.
func GetBytes(data []byte, target *[]byte) error {
	d := &decoder{src: data, pos: 0}
	d.skipWhitespace()
	if d.pos >= len(d.src) {
		return &SyntaxError{Offset: d.pos, Msg: "unexpected end of input"}
	}

	if d.src[d.pos] == '"' {
		d.pos++
		start := d.pos
		for d.pos < len(d.src) {
			if d.src[d.pos] == '"' {
				raw := d.src[start:d.pos]
				d.pos++
				// Try base64 decode first
				decoded, err := base64.StdEncoding.DecodeString(string(raw))
				if err == nil {
					*target = decoded
					return nil
				}
				// Fall back to raw bytes
				*target = make([]byte, len(raw))
				copy(*target, raw)
				return nil
			}
			if d.src[d.pos] == '\\' {
				d.pos += 2
				continue
			}
			d.pos++
		}
		return &SyntaxError{Offset: start, Msg: "unterminated string"}
	}

	return &SyntaxError{Offset: d.pos, Msg: "expected string"}
}

// UnsafeDecode decodes JSON without copying strings.
// The returned strings reference the input buffer.
// Only safe if the input buffer is not modified while strings are in use.
func UnsafeDecode(data []byte, v any) error {
	return Unmarshal(data, v)
}

// DecodeStream decodes a stream of JSON values from a reader.
// Returns the next decoded value and any error.
func DecodeStream(data []byte, offset int) (any, int, error) {
	d := &decoder{src: data, pos: offset}
	var result any
	err := d.decodeValueAny(&result)
	return result, d.pos, err
}
