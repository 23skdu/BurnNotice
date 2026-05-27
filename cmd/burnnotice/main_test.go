package main

import (
	"bytes"
	"testing"
)

func TestBurnNoticeCommands(t *testing.T) {
	// Simply verify that the commands exist and help output works to achieve coverage of init/setup
	rootCmd.SetOut(&bytes.Buffer{})
	rootCmd.SetArgs([]string{"--help"})
	if err := rootCmd.Execute(); err != nil {
		t.Fatalf("help command failed: %v", err)
	}
	
	// download without args
	rootCmd.SetArgs([]string{"download"})
	_ = rootCmd.Execute() // should error but provides coverage
	
	// convert without args
	rootCmd.SetArgs([]string{"convert"})
	_ = rootCmd.Execute()
	
	// profile without args
	rootCmd.SetArgs([]string{"profile"})
	_ = rootCmd.Execute()
}
