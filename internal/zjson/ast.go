package zjson

import (
	"fmt"
	"strconv"
)

// Node represents a lazy JSON AST node.
// Values are not parsed until accessed, enabling zero-allocation parsing.
type Node struct {
	// raw is the raw JSON bytes for this node.
	raw []byte

	// start and end are offsets into the original input.
	start int
	end   int

	// typ is the cached value type (computed on first access).
	typ ValueType

	// children are lazily parsed child nodes.
	children []Node

	// parsed indicates whether children have been parsed.
	parsed bool
}

// Parse parses JSON input into a lazy AST root node.
func Parse(data []byte) *Node {
	return &Node{
		raw:   data,
		start: 0,
		end:   len(data),
	}
}

// Type returns the JSON value type.
func (n *Node) Type() ValueType {
	if n.typ == 0 && len(n.raw) > 0 {
		n.typ = n.detectType()
	}
	return n.typ
}

// detectType determines the type from raw JSON.
func (n *Node) detectType() ValueType {
	if len(n.raw) == 0 {
		return VNull
	}

	// Skip whitespace
	i := 0
	for i < len(n.raw) && (n.raw[i] == ' ' || n.raw[i] == '\t' || n.raw[i] == '\n' || n.raw[i] == '\r') {
		i++
	}

	if i >= len(n.raw) {
		return VNull
	}

	switch n.raw[i] {
	case '"':
		return VString
	case '{':
		return VObject
	case '[':
		return VArray
	case 't', 'f':
		return VBool
	case 'n':
		return VNull
	default:
		return VNumber
	}
}

// String returns the string value.
func (n *Node) String() (string, error) {
	if n.Type() != VString {
		return "", fmt.Errorf("zjson: expected string, got %v", n.Type())
	}

	// Find string content
	i := 0
	for i < len(n.raw) && n.raw[i] != '"' {
		i++
	}
	if i >= len(n.raw) {
		return "", &SyntaxError{Offset: i, Msg: "expected '\"'"}
	}
	i++ // skip opening quote

	start := i
	hasEscape := false

	for i < len(n.raw) {
		if n.raw[i] == '"' {
			raw := n.raw[start:i]
			if hasEscape {
				buf := GetBuf()
				defer PutBuf(buf, DefaultConfig.MaxBufferSize)
				used := Unescape(*buf, raw)
				return string((*buf)[:used]), nil
			}
			return Mem2Str(raw), nil
		}
		if n.raw[i] == '\\' {
			hasEscape = true
			i++
			if i < len(n.raw) {
				switch n.raw[i] {
				case '"', '\\', '/', 'b', 'f', 'n', 'r', 't':
					i++
				case 'u':
					i += 5
				}
			}
			continue
		}
		i++
	}

	return "", &SyntaxError{Offset: start, Msg: "unterminated string"}
}

// Int returns the integer value.
func (n *Node) Int() (int64, error) {
	if n.Type() != VNumber {
		return 0, fmt.Errorf("zjson: expected number, got %v", n.Type())
	}

	// Extract number string
	i := 0
	for i < len(n.raw) && (n.raw[i] == ' ' || n.raw[i] == '\t' || n.raw[i] == '\n' || n.raw[i] == '\r') {
		i++
	}

	start := i
	for i < len(n.raw) {
		switch n.raw[i] {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '+':
			i++
		default:
			goto done
		}
	}
done:
	numStr := Mem2Str(n.raw[start:i])
	return strconv.ParseInt(numStr, 10, 64)
}

// Float returns the float value.
func (n *Node) Float() (float64, error) {
	if n.Type() != VNumber {
		return 0, fmt.Errorf("zjson: expected number, got %v", n.Type())
	}

	i := 0
	for i < len(n.raw) && (n.raw[i] == ' ' || n.raw[i] == '\t' || n.raw[i] == '\n' || n.raw[i] == '\r') {
		i++
	}

	start := i
	for i < len(n.raw) {
		switch n.raw[i] {
		case '0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '-', '.', 'e', 'E', '+':
			i++
		default:
			goto done
		}
	}
done:
	numStr := Mem2Str(n.raw[start:i])
	return strconv.ParseFloat(numStr, 64)
}

// Bool returns the boolean value.
func (n *Node) Bool() (bool, error) {
	if n.Type() != VBool {
		return false, fmt.Errorf("zjson: expected bool, got %v", n.Type())
	}

	i := 0
	for i < len(n.raw) && (n.raw[i] == ' ' || n.raw[i] == '\t' || n.raw[i] == '\n' || n.raw[i] == '\r') {
		i++
	}

	if i+4 <= len(n.raw) && string(n.raw[i:i+4]) == "true" {
		return true, nil
	}
	if i+5 <= len(n.raw) && string(n.raw[i:i+5]) == "false" {
		return false, nil
	}

	return false, &SyntaxError{Offset: i, Msg: "invalid boolean"}
}

// IsNull returns true if the value is null.
func (n *Node) IsNull() bool {
	return n.Type() == VNull
}

// Len returns the number of elements in an object or array.
func (n *Node) Len() int {
	if n.Type() != VObject && n.Type() != VArray {
		return 0
	}

	n.ensureParsed()
	if n.Type() == VObject {
		return len(n.children) / 2
	}
	return len(n.children)
}

// Get returns a child node by index (for arrays) or key (for objects).
func (n *Node) Get(key string) *Node {
	if n.Type() == VArray {
		idx, err := strconv.Atoi(key)
		if err != nil {
			return nil
		}
		n.ensureParsed()
		if idx >= 0 && idx < len(n.children) {
			return &n.children[idx]
		}
		return nil
	}

	if n.Type() == VObject {
		n.ensureParsed()
		for i := range n.children {
			// Children are stored as key-value pairs
			if i%2 == 0 {
				// This is a key
				k, _ := n.children[i].String()
				if k == key {
					// Next child is the value
					if i+1 < len(n.children) {
						return &n.children[i+1]
					}
				}
			}
		}
	}

	return nil
}

// At returns a child node by index.
func (n *Node) At(index int) *Node {
	n.ensureParsed()
	if index >= 0 && index < len(n.children) {
		return &n.children[index]
	}
	return nil
}

// Keys returns all keys in an object.
func (n *Node) Keys() []string {
	if n.Type() != VObject {
		return nil
	}

	n.ensureParsed()
	keys := make([]string, 0, len(n.children)/2)
	for i := 0; i < len(n.children); i += 2 {
		k, _ := n.children[i].String()
		keys = append(keys, k)
	}
	return keys
}

// Each iterates over array elements or object key-value pairs.
func (n *Node) Each(fn func(key string, value *Node) bool) {
	if n.Type() == VArray {
		n.ensureParsed()
		for i := range n.children {
			if !fn("", &n.children[i]) {
				return
			}
		}
	} else if n.Type() == VObject {
		n.ensureParsed()
		for i := 0; i < len(n.children); i += 2 {
			key, _ := n.children[i].String()
			if i+1 < len(n.children) {
				if !fn(key, &n.children[i+1]) {
					return
				}
			}
		}
	}
}

// Raw returns the raw JSON bytes for this node.
func (n *Node) Raw() []byte {
	return n.raw
}

// ensureParsed parses children if not already done.
func (n *Node) ensureParsed() {
	if n.parsed {
		return
	}
	n.parsed = true

	if len(n.raw) == 0 {
		return
	}

	d := &decoder{src: n.raw, pos: 0}
	d.skipWhitespace()

	if d.pos >= len(d.src) {
		return
	}

	switch d.src[d.pos] {
	case '[':
		n.parseArray(d)
	case '{':
		n.parseObject(d)
	}
}

// parseArray parses array children.
func (n *Node) parseArray(d *decoder) {
	d.pos++ // skip '['
	d.skipWhitespace()

	if d.pos < len(d.src) && d.src[d.pos] == ']' {
		d.pos++
		return
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] == ']' {
			d.pos++
			return
		}

		start := d.pos
		if err := d.skipValue(); err != nil {
			return
		}

		child := Node{
			raw:   d.src[start:d.pos],
			start: start,
			end:   d.pos,
		}
		n.children = append(n.children, child)

		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] == ']' {
			d.pos++
			return
		}
		if d.src[d.pos] == ',' {
			d.pos++
		}
	}
}

// parseObject parses object children (key-value pairs).
func (n *Node) parseObject(d *decoder) {
	d.pos++ // skip '{'
	d.skipWhitespace()

	if d.pos < len(d.src) && d.src[d.pos] == '}' {
		d.pos++
		return
	}

	for {
		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] == '}' {
			d.pos++
			return
		}

		// Parse key
		if d.src[d.pos] != '"' {
			return
		}
		keyStart := d.pos
		d.pos++
		for d.pos < len(d.src) && d.src[d.pos] != '"' {
			if d.src[d.pos] == '\\' {
				d.pos++
			}
			d.pos++
		}
		if d.pos < len(d.src) {
			d.pos++ // skip closing quote
		}

		keyNode := Node{
			raw:   d.src[keyStart:d.pos],
			start: keyStart,
			end:   d.pos,
		}
		n.children = append(n.children, keyNode)

		// Skip ':'
		d.skipWhitespace()
		if d.pos < len(d.src) && d.src[d.pos] == ':' {
			d.pos++
		}

		// Parse value
		d.skipWhitespace()
		valStart := d.pos
		if err := d.skipValue(); err != nil {
			return
		}

		valNode := Node{
			raw:   d.src[valStart:d.pos],
			start: valStart,
			end:   d.pos,
		}
		n.children = append(n.children, valNode)

		d.skipWhitespace()
		if d.pos >= len(d.src) || d.src[d.pos] == '}' {
			d.pos++
			return
		}
		if d.src[d.pos] == ',' {
			d.pos++
		}
	}
}

// Exists checks if a path exists in the JSON.
func (n *Node) Exists(path ...string) bool {
	current := n
	for _, p := range path {
		current = current.Get(p)
		if current == nil {
			return false
		}
	}
	return true
}

// GetString returns a string value at the given path.
func (n *Node) GetString(path ...string) (string, bool) {
	current := n
	for _, p := range path {
		current = current.Get(p)
		if current == nil {
			return "", false
		}
	}
	s, err := current.String()
	return s, err == nil
}

// GetInt returns an int value at the given path.
func (n *Node) GetInt(path ...string) (int64, bool) {
	current := n
	for _, p := range path {
		current = current.Get(p)
		if current == nil {
			return 0, false
		}
	}
	i, err := current.Int()
	return i, err == nil
}

// GetFloat returns a float value at the given path.
func (n *Node) GetFloat(path ...string) (float64, bool) {
	current := n
	for _, p := range path {
		current = current.Get(p)
		if current == nil {
			return 0, false
		}
	}
	f, err := current.Float()
	return f, err == nil
}

// GetBool returns a bool value at the given path.
func (n *Node) GetBool(path ...string) (bool, bool) {
	current := n
	for _, p := range path {
		current = current.Get(p)
		if current == nil {
			return false, false
		}
	}
	b, err := current.Bool()
	return b, err == nil
}
