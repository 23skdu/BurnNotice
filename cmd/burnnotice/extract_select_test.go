package main

import "testing"

func TestParseModelSize(t *testing.T) {
	tests := []struct {
		in   string
		want float64
		ok   bool
	}{
		{"0.5B", 0.5e9, true},
		{"7B", 7e9, true},
		{"7.1B", 7.1e9, true},
		{"13B", 13e9, true},
		{"70B", 70e9, true},
		{"175B", 175e9, true},
		{"500M", 500e6, true},
		{"100M", 100e6, true},
		{"7b", 7e9, true},
		{"7.5b", 7.5e9, true},
		{"100m", 100e6, true},
		{"  13B  ", 13e9, true},
		{"7000000000", 7e9, true},
		{"", 0, false},
		{"B", 0, false},
		{"abc", 0, false},
		{"13X", 0, false},
	}
	for _, tt := range tests {
		t.Run(tt.in, func(t *testing.T) {
			got, ok := parseModelSize(tt.in)
			if ok != tt.ok {
				t.Errorf("parseModelSize(%q) ok = %v, want %v", tt.in, ok, tt.ok)
				return
			}
			if ok && got != tt.want {
				t.Errorf("parseModelSize(%q) = %f, want %f", tt.in, got, tt.want)
			}
		})
	}
}

func TestIsLargeModel(t *testing.T) {
	tests := []struct {
		modelSize string
		want      bool
	}{
		{"0.5B", false},
		{"3B", false},
		{"7B", false}, // boundary: strict > 7B
		{"7.1B", true},
		{"8B", true},
		{"13B", true},
		{"70B", true},
		{"500M", false},
		{"8000M", true},
		{"garbage", false}, // unparseable → safe DIM default
		{"", false},
	}
	for _, tt := range tests {
		t.Run(tt.modelSize, func(t *testing.T) {
			if got := isLargeModel(tt.modelSize); got != tt.want {
				t.Errorf("isLargeModel(%q) = %v, want %v", tt.modelSize, got, tt.want)
			}
		})
	}
}

func TestShouldUseWhitenedSVD(t *testing.T) {
	tests := []struct {
		name        string
		flagChanged bool
		flagValue   bool
		modelSize   string
		want        bool
	}{
		{
			name:        "explicit true wins regardless of size",
			flagChanged: true, flagValue: true, modelSize: "0.5B",
			want: true,
		},
		{
			name:        "explicit false wins even for large model",
			flagChanged: true, flagValue: false, modelSize: "70B",
			want: false,
		},
		{
			name:        "auto: small model uses DIM",
			flagChanged: false, modelSize: "0.5B",
			want: false,
		},
		{
			name:        "auto: boundary 7B uses DIM",
			flagChanged: false, modelSize: "7B",
			want: false,
		},
		{
			name:        "auto: 7.1B uses whitened SVD",
			flagChanged: false, modelSize: "7.1B",
			want: true,
		},
		{
			name:        "auto: 13B uses whitened SVD",
			flagChanged: false, modelSize: "13B",
			want: true,
		},
		{
			name:        "auto: 70B uses whitened SVD",
			flagChanged: false, modelSize: "70B",
			want: true,
		},
		{
			name:        "auto: unparseable size defaults to DIM",
			flagChanged: false, modelSize: "weird",
			want: false,
		},
	}
	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := shouldUseWhitenedSVD(tt.flagChanged, tt.flagValue, tt.modelSize)
			if got != tt.want {
				t.Errorf("shouldUseWhitenedSVD(%v, %v, %q) = %v, want %v",
					tt.flagChanged, tt.flagValue, tt.modelSize, got, tt.want)
			}
		})
	}
}
