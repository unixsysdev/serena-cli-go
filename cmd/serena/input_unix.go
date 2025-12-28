//go:build !windows

package main

import (
	"os"
	"syscall"
)

func stdinHasData() bool {
	fd := int(os.Stdin.Fd())
	if fd < 0 {
		return false
	}

	var set syscall.FdSet
	set.Bits[fd/64] |= 1 << (uint(fd) % 64)
	timeout := syscall.Timeval{Sec: 0, Usec: 0}
	n, err := syscall.Select(fd+1, &set, nil, nil, &timeout)
	return err == nil && n > 0
}
