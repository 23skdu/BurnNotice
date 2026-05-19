package converter

import (
	"encoding/binary"
	"io"
)

func readUint64LE(r io.Reader, val *uint64) error {
	buf := make([]byte, 8)
	if _, err := io.ReadFull(r, buf); err != nil {
		return err
	}
	*val = binary.LittleEndian.Uint64(buf)
	return nil
}
