package zjson

import (
	"fmt"
	"math"
	"reflect"
	"strconv"
)

// Marshal serializes the given value to JSON.
// Uses pooled buffers for zero-allocation encoding.
func Marshal(v any) ([]byte, error) {
	buf := GetBuf()
	defer PutBuf(buf, DefaultConfig.MaxBufferSize)

	e := &encoder{buf: *buf}
	if err := e.encodeValue(reflect.ValueOf(v)); err != nil {
		return nil, err
	}

	result := make([]byte, len(e.buf))
	copy(result, e.buf)
	return result, nil
}

// MarshalToBuf serializes the given value to JSON, writing into the provided buffer.
// Returns the number of bytes written.
func MarshalToBuf(v any, buf []byte) (int, error) {
	e := &encoder{buf: buf}
	if err := e.encodeValue(reflect.ValueOf(v)); err != nil {
		return 0, err
	}
	return len(e.buf), nil
}

// encoder holds state during recursive descent encoding.
type encoder struct {
	buf []byte
}

// encodeValue encodes a reflect.Value to JSON.
func (e *encoder) encodeValue(rv reflect.Value) error {
	if !rv.IsValid() {
		e.buf = append(e.buf, "null"...)
		return nil
	}

	// Handle interface values
	if rv.Kind() == reflect.Interface {
		if rv.IsNil() {
			e.buf = append(e.buf, "null"...)
			return nil
		}
		rv = rv.Elem()
	}

	// Handle pointer values
	if rv.Kind() == reflect.Ptr {
		if rv.IsNil() {
			e.buf = append(e.buf, "null"...)
			return nil
		}
		return e.encodeValue(rv.Elem())
	}

	switch rv.Kind() {
	case reflect.Bool:
		e.encodeBool(rv.Bool())
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		e.encodeInt(rv.Int())
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64:
		e.encodeUint(rv.Uint())
	case reflect.Float32, reflect.Float64:
		e.encodeFloat(rv.Float())
	case reflect.String:
		e.encodeString(rv.String())
	case reflect.Slice:
		if rv.Type().Elem().Kind() == reflect.Uint8 {
			// []byte -> base64 string
			e.encodeBytes(rv.Bytes())
		} else {
			return e.encodeSlice(rv)
		}
	case reflect.Array:
		return e.encodeArray(rv)
	case reflect.Map:
		return e.encodeMap(rv)
	case reflect.Struct:
		return e.encodeStruct(rv)
	default:
		return fmt.Errorf("zjson: unsupported type %s", rv.Type())
	}

	return nil
}

// encodeBool encodes a boolean value.
func (e *encoder) encodeBool(b bool) {
	if b {
		e.buf = append(e.buf, "true"...)
	} else {
		e.buf = append(e.buf, "false"...)
	}
}

// encodeInt encodes an int64 value.
func (e *encoder) encodeInt(i int64) {
	e.buf = strconv.AppendInt(e.buf, i, 10)
}

// encodeUint encodes a uint64 value.
func (e *encoder) encodeUint(u uint64) {
	e.buf = strconv.AppendUint(e.buf, u, 10)
}

// encodeFloat encodes a float64 value.
func (e *encoder) encodeFloat(f float64) {
	if math.IsInf(f, 0) || math.IsNaN(f) {
		e.buf = append(e.buf, "null"...)
		return
	}
	e.buf = strconv.AppendFloat(e.buf, f, 'g', -1, 64)
}

// encodeString encodes a string value with proper escaping.
func (e *encoder) encodeString(s string) {
	e.buf = append(e.buf, '"')

	// Check if string needs escaping
	needsEscape := false
	for i := 0; i < len(s); i++ {
		c := s[i]
		if c == '"' || c == '\\' || c < 0x20 {
			needsEscape = true
			break
		}
	}

	if !needsEscape {
		// Fast path: no escaping needed
		e.buf = append(e.buf, s...)
	} else {
		// Slow path: escape characters
		for i := 0; i < len(s); i++ {
			switch s[i] {
			case '"':
				e.buf = append(e.buf, '\\', '"')
			case '\\':
				e.buf = append(e.buf, '\\', '\\')
			case '\b':
				e.buf = append(e.buf, '\\', 'b')
			case '\f':
				e.buf = append(e.buf, '\\', 'f')
			case '\n':
				e.buf = append(e.buf, '\\', 'n')
			case '\r':
				e.buf = append(e.buf, '\\', 'r')
			case '\t':
				e.buf = append(e.buf, '\\', 't')
			default:
				if s[i] < 0x20 {
					// Unicode escape for control characters
					e.buf = append(e.buf, '\\', 'u', '0', '0')
					e.buf = append(e.buf, hexDigit(s[i]>>4), hexDigit(s[i]&0x0F))
				} else {
					e.buf = append(e.buf, s[i])
				}
			}
		}
	}

	e.buf = append(e.buf, '"')
}

// hexDigit returns the hex digit for a 4-bit value.
func hexDigit(b byte) byte {
	if b < 10 {
		return '0' + b
	}
	return 'a' + b - 10
}

// encodeBytes encodes a byte slice as a base64 string.
func (e *encoder) encodeBytes(b []byte) {
	if b == nil {
		e.buf = append(e.buf, "null"...)
		return
	}
	e.buf = append(e.buf, '"')
	// Simple base64 encoding
	const base64Chars = "ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/"
	for i := 0; i < len(b); i += 3 {
		n := min(3, len(b)-i)
		val := uint32(b[i]) << 16
		if n > 1 {
			val |= uint32(b[i+1]) << 8
		}
		if n > 2 {
			val |= uint32(b[i+2])
		}
		e.buf = append(e.buf, base64Chars[val>>18&0x3F])
		e.buf = append(e.buf, base64Chars[val>>12&0x3F])
		if n > 1 {
			e.buf = append(e.buf, base64Chars[val>>6&0x3F])
		} else {
			e.buf = append(e.buf, '=')
		}
		if n > 2 {
			e.buf = append(e.buf, base64Chars[val&0x3F])
		} else {
			e.buf = append(e.buf, '=')
		}
	}
	e.buf = append(e.buf, '"')
}

// encodeSlice encodes a slice value.
func (e *encoder) encodeSlice(rv reflect.Value) error {
	if rv.IsNil() {
		e.buf = append(e.buf, "null"...)
		return nil
	}

	e.buf = append(e.buf, '[')
	length := rv.Len()
	for i := 0; i < length; i++ {
		if i > 0 {
			e.buf = append(e.buf, ',')
		}
		if err := e.encodeValue(rv.Index(i)); err != nil {
			return err
		}
	}
	e.buf = append(e.buf, ']')
	return nil
}

// encodeArray encodes an array value.
func (e *encoder) encodeArray(rv reflect.Value) error {
	e.buf = append(e.buf, '[')
	length := rv.Len()
	for i := 0; i < length; i++ {
		if i > 0 {
			e.buf = append(e.buf, ',')
		}
		if err := e.encodeValue(rv.Index(i)); err != nil {
			return err
		}
	}
	e.buf = append(e.buf, ']')
	return nil
}

// encodeMap encodes a map value.
func (e *encoder) encodeMap(rv reflect.Value) error {
	if rv.IsNil() {
		e.buf = append(e.buf, "null"...)
		return nil
	}

	e.buf = append(e.buf, '{')
	first := true
	iter := rv.MapRange()
	for iter.Next() {
		if !first {
			e.buf = append(e.buf, ',')
		}
		first = false

		// Encode key
		key := iter.Key()
		if key.Kind() == reflect.String {
			e.encodeString(key.String())
		} else {
			e.encodeString(fmt.Sprintf("%v", key.Interface()))
		}

		e.buf = append(e.buf, ':')

		// Encode value
		if err := e.encodeValue(iter.Value()); err != nil {
			return err
		}
	}
	e.buf = append(e.buf, '}')
	return nil
}

// encodeStruct encodes a struct value.
func (e *encoder) encodeStruct(rv reflect.Value) error {
	typ := rv.Type()

	e.buf = append(e.buf, '{')
	first := true

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
		omitEmpty := false
		if idx := indexOfComma(tag); idx >= 0 {
			name = tag[:idx]
			omitEmpty = tag[idx:] == ",omitempty"
		}
		if name == "" {
			name = f.Name
		}

		// Handle omitempty
		if omitEmpty && isEmptyValue(rv.Field(i)) {
			continue
		}

		if !first {
			e.buf = append(e.buf, ',')
		}
		first = false

		// Encode field name
		e.encodeString(name)
		e.buf = append(e.buf, ':')

		// Encode field value
		if err := e.encodeValue(rv.Field(i)); err != nil {
			return err
		}
	}

	e.buf = append(e.buf, '}')
	return nil
}

// indexOfComma finds the first comma in a string.
func indexOfComma(s string) int {
	for i := 0; i < len(s); i++ {
		if s[i] == ',' {
			return i
		}
	}
	return -1
}

// hasOmitEmpty checks if the tag contains "omitempty".
func hasOmitEmpty(tag string) bool {
	if tag == "" {
		return false
	}
	// Check for ",omitempty" suffix
	for i := 0; i < len(tag); i++ {
		if tag[i] == ',' && i+9 <= len(tag) && tag[i:i+10] == ",omitempty" {
			return true
		}
	}
	return false
}

// isEmptyValue checks if a value is empty.
func isEmptyValue(rv reflect.Value) bool {
	switch rv.Kind() {
	case reflect.Array, reflect.Map, reflect.Slice, reflect.String:
		return rv.Len() == 0
	case reflect.Bool:
		return !rv.Bool()
	case reflect.Int, reflect.Int8, reflect.Int16, reflect.Int32, reflect.Int64:
		return rv.Int() == 0
	case reflect.Uint, reflect.Uint8, reflect.Uint16, reflect.Uint32, reflect.Uint64, reflect.Uintptr:
		return rv.Uint() == 0
	case reflect.Float32, reflect.Float64:
		return rv.Float() == 0
	case reflect.Interface, reflect.Ptr:
		return rv.IsNil()
	}
	return false
}

// min returns the minimum of two integers.
func min(a, b int) int {
	if a < b {
		return a
	}
	return b
}

// AppendJSON appends a JSON-encoded value to the given buffer.
func AppendJSON(buf []byte, v any) ([]byte, error) {
	e := &encoder{buf: buf}
	if err := e.encodeValue(reflect.ValueOf(v)); err != nil {
		return buf, err
	}
	return e.buf, nil
}

// EncodeToWriter encodes a value and writes it to a buffer, returning the result.
func EncodeToWriter(v any) ([]byte, error) {
	buf := GetBuf()
	defer PutBuf(buf, DefaultConfig.MaxBufferSize)

	e := &encoder{buf: *buf}
	if err := e.encodeValue(reflect.ValueOf(v)); err != nil {
		return nil, err
	}

	result := make([]byte, len(e.buf))
	copy(result, e.buf)
	return result, nil
}
