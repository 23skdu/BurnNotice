package zjson

import "unsafe"

// GoSlice mirrors runtime.slice for direct manipulation.
type GoSlice struct {
	Ptr unsafe.Pointer
	Len int
	Cap int
}

// GoString mirrors runtime.stringStruct for direct manipulation.
type GoString struct {
	Ptr unsafe.Pointer
	Len int
}

// Mem2Str converts a byte slice to a string without copying.
// The caller must ensure the backing array is not modified while the string is in use.
func Mem2Str(b []byte) string {
	// #nosec G103 -- intentional zero-copy conversion for performance
	return *(*string)(unsafe.Pointer(&b))
}

// Str2Mem converts a string to a byte slice without copying.
// The returned slice must NOT be modified (read-only access only).
func Str2Mem(s string) []byte {
	// #nosec G103 -- intentional zero-copy conversion for performance
	return unsafe.Slice((*byte)(unsafe.Pointer(unsafe.StringData(s))), len(s))
}

// SliceHeader returns a mutable GoSlice pointer for a byte slice.
func SliceHeader(b *[]byte) *GoSlice {
	// #nosec G103 -- direct memory reflection for zero-allocation parsing
	return (*GoSlice)(unsafe.Pointer(b))
}

// StringHeader returns a GoString pointer for a string.
func StringHeader(s *string) *GoString {
	// #nosec G103 -- direct memory reflection for zero-allocation parsing
	return (*GoString)(unsafe.Pointer(s))
}

// GrowSlice grows a byte slice to at least newSize capacity.
// Uses Go's runtime growslice via append pattern.
func GrowSlice(b []byte, newSize int) []byte {
	if cap(b) >= newSize {
		return b[:newSize]
	}
	newCap := cap(b) * 2
	if newCap < newSize {
		newCap = newSize
	}
	new := make([]byte, len(b), newCap)
	copy(new, b)
	return new[:newSize]
}

// CanSizeReuse checks if a buffer capacity is within acceptable bounds
// for returning to the sync.Pool. Prevents unbounded memory growth.
func CanSizeReuse(cap int, maxAllowed int) bool {
	return cap <= maxAllowed
}
