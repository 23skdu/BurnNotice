package main

import (
	"bytes"
	"strings"
	"testing"

	"github.com/spf13/cobra"
)

func TestProfileCmd_FlagsRegistered(t *testing.T) {
	wantFlags := []string{
		"whitened-svd",
		"model-size",
		"num-directions",
		"shrinkage",
		"eig-floor",
		"dataset",
		"layers",
		"output",
	}
	for _, name := range wantFlags {
		if profileCmd.Flags().Lookup(name) == nil {
			t.Errorf("profileCmd missing flag --%s", name)
		}
	}
}

func TestProfileCmd_FlagDefaults(t *testing.T) {
	tests := []struct {
		flag      string
		defValue  string
		defIsBool bool
	}{
		{"whitened-svd", "false", true},
		{"model-size", "0.5B", false},
		{"num-directions", "1", false},
		{"shrinkage", "-1", false},
		{"eig-floor", "0", false},
	}
	for _, tt := range tests {
		t.Run(tt.flag, func(t *testing.T) {
			f := profileCmd.Flags().Lookup(tt.flag)
			if f == nil {
				t.Fatalf("flag --%s not registered", tt.flag)
			}
			if f.DefValue != tt.defValue {
				t.Errorf("flag --%s default = %q, want %q", tt.flag, f.DefValue, tt.defValue)
			}
			if f.Value.Type() == "bool" != tt.defIsBool {
				t.Errorf("flag --%s type = %s, want bool=%v", tt.flag, f.Value.Type(), tt.defIsBool)
			}
		})
	}
}

func TestProfileCmd_FlagWiringHonorsModelSizeDefault(t *testing.T) {
	cases := []struct {
		name            string
		args            []string
		wantChanged     bool
		wantFlagValue   bool
		wantModelSize   string
		wantUseWhitened bool
	}{
		{
			name:            "default (no flags): small model → DIM",
			args:            []string{},
			wantChanged:     false,
			wantFlagValue:   false,
			wantModelSize:   "0.5B",
			wantUseWhitened: false,
		},
		{
			name:            "--model-size=13B alone: auto → whitened SVD",
			args:            []string{"--model-size", "13B"},
			wantChanged:     false,
			wantFlagValue:   false,
			wantModelSize:   "13B",
			wantUseWhitened: true,
		},
		{
			name:            "--whitened-svd=false with large model: explicit false wins",
			args:            []string{"--model-size", "70B", "--whitened-svd=false"},
			wantChanged:     true,
			wantFlagValue:   false,
			wantModelSize:   "70B",
			wantUseWhitened: false,
		},
		{
			name:            "--whitened-svd=true with small model: explicit true wins",
			args:            []string{"--model-size", "0.5B", "--whitened-svd=true"},
			wantChanged:     true,
			wantFlagValue:   true,
			wantModelSize:   "0.5B",
			wantUseWhitened: true,
		},
	}

	for _, tt := range cases {
		t.Run(tt.name, func(t *testing.T) {
			cmd := cloneProfileCmd()
			cmd.SetArgs(append([]string{"ignored.onnx"}, tt.args...))
			// Suppress execution errors; we only inspect flag state.
			cmd.SetOut(&bytes.Buffer{})
			cmd.SetErr(&bytes.Buffer{})
			_ = cmd.ParseFlags(append([]string{"ignored.onnx"}, tt.args...))

			whitened, _ := cmd.Flags().GetBool("whitened-svd")
			modelSize, _ := cmd.Flags().GetString("model-size")
			changed := cmd.Flags().Changed("whitened-svd")

			if changed != tt.wantChanged {
				t.Errorf("Changed(whitened-svd) = %v, want %v", changed, tt.wantChanged)
			}
			if whitened != tt.wantFlagValue {
				t.Errorf("whitened-svd value = %v, want %v", whitened, tt.wantFlagValue)
			}
			if modelSize != tt.wantModelSize {
				t.Errorf("model-size = %q, want %q", modelSize, tt.wantModelSize)
			}

			got := shouldUseWhitenedSVD(changed, whitened, modelSize)
			if got != tt.wantUseWhitened {
				t.Errorf("shouldUseWhitenedSVD = %v, want %v", got, tt.wantUseWhitened)
			}
		})
	}
}

func TestProfileCmd_HelpMentionsWhitenedSVD(t *testing.T) {
	buf := &bytes.Buffer{}
	profileCmd.SetOut(buf)
	profileCmd.SetErr(buf)
	if err := profileCmd.Help(); err != nil {
		t.Fatalf("Help failed: %v", err)
	}
	out := buf.String()
	if !strings.Contains(out, "whitened-svd") {
		t.Error("help text does not mention --whitened-svd")
	}
	if !strings.Contains(out, "7B") {
		t.Error("help text does not mention the >7B default behavior")
	}
}

func cloneProfileCmd() *cobra.Command {
	clone := *profileCmd
	clone.Flags() // ensure flag set is initialized
	clone.ResetFlags()
	// Re-register the same flags on the clone.
	clone.Flags().StringP("output", "o", "./profiles", "Profile output directory")
	clone.Flags().String("dataset", "", "Path to profiling dataset (JSON/JSONL)")
	clone.Flags().IntSlice("layers", []int{0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31}, "Specific layers to profile (default: all)")
	clone.Flags().Bool("whitened-svd", false, "Use whitened SVD extraction instead of DIM (default: true for models > 7B)")
	clone.Flags().String("model-size", "0.5B", "Model parameter count (e.g. 0.5B, 7B, 13B, 70B); drives the default --whitened-svd choice")
	clone.Flags().Int("num-directions", 1, "Number of top directions to extract when --whitened-svd is set")
	clone.Flags().Float64("shrinkage", -1, "Ledoit-Wolf shrinkage intensity for the whitening covariance (<0 = auto-estimate)")
	clone.Flags().Float64("eig-floor", 0, "Minimum eigenvalue for the whitening matrix (0 = auto, 1e-10 of max)")
	clone.Flags().String("oversight-dataset", "", "Path to oversight contrastive dataset (optional)")
	clone.Flags().Bool("cone-analysis", false, "Run concept cone geometry analysis")
	clone.Flags().Bool("rdo", false, "Refine extracted direction via Refusal Direction Optimization")
	clone.Flags().Bool("commit-window", false, "Analyze behavioral commitment window")
	clone.Flags().Bool("calibrate", false, "Calibrate optimal per-layer steering thresholds")
	clone.Flags().Bool("obfuscate", false, "Apply orthogonal obfuscation to extracted directions")
	return &clone
}
