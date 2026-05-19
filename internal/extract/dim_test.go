package extract

import (
	"math"
	"testing"
)

func TestExtractDIM(t *testing.T) {
	tests := []struct {
		name     string
		positive [][]float64
		negative [][]float64
		want     []float64
	}{
		{
			name:     "simple 2d",
			positive: [][]float64{{1, 0}, {1, 0}},
			negative: [][]float64{{0, 1}, {0, 1}},
			want:     []float64{1 / math.Sqrt(2), -1 / math.Sqrt(2)},
		},
		{
			name:     "identical sets returns zero vector",
			positive: [][]float64{{1, 2}, {3, 4}},
			negative: [][]float64{{1, 2}, {3, 4}},
			want:     []float64{0, 0},
		},
		{
			name:     "single pair",
			positive: [][]float64{{3, 0, 0}},
			negative: [][]float64{{0, 4, 0}},
			want:     []float64{0.6, -0.8, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := ExtractDIM(tt.positive, tt.negative)
			if len(got) != len(tt.want) {
				t.Fatalf("len(got) = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-9 {
					t.Errorf("got[%d] = %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestExtractDIM_Empty(t *testing.T) {
	if got := ExtractDIM(nil, [][]float64{{1}}); got != nil {
		t.Errorf("ExtractDIM(nil, ...) = %v, want nil", got)
	}
	if got := ExtractDIM([][]float64{{1}}, nil); got != nil {
		t.Errorf("ExtractDIM(..., nil) = %v, want nil", got)
	}
}

func TestNormalize(t *testing.T) {
	tests := []struct {
		name string
		vec  []float64
		want []float64
	}{
		{
			name: "unit vector unchanged",
			vec:  []float64{1, 0, 0},
			want: []float64{1, 0, 0},
		},
		{
			name: "scale to unit",
			vec:  []float64{3, 4},
			want: []float64{0.6, 0.8},
		},
		{
			name: "zero vector unchanged",
			vec:  []float64{0, 0, 0},
			want: []float64{0, 0, 0},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Normalize(tt.vec)
			if len(got) != len(tt.want) {
				t.Fatalf("len(got) = %d, want %d", len(got), len(tt.want))
			}
			for i := range got {
				if math.Abs(got[i]-tt.want[i]) > 1e-9 {
					t.Errorf("got[%d] = %f, want %f", i, got[i], tt.want[i])
				}
			}
		})
	}
}

func TestNorm(t *testing.T) {
	tests := []struct {
		name string
		vec  []float64
		want float64
	}{
		{"unit x", []float64{1, 0, 0}, 1},
		{"3-4-5", []float64{3, 4}, 5},
		{"zero", []float64{0, 0, 0}, 0},
		{"negative", []float64{-3, 4}, 5},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Norm(tt.vec)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Norm(%v) = %f, want %f", tt.vec, got, tt.want)
			}
		})
	}
}

func TestDot(t *testing.T) {
	tests := []struct {
		name string
		a, b []float64
		want float64
	}{
		{"orthogonal", []float64{1, 0}, []float64{0, 1}, 0},
		{"parallel", []float64{3, 0}, []float64{4, 0}, 12},
		{"mixed", []float64{1, 2, 3}, []float64{4, 5, 6}, 32},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Dot(tt.a, tt.b)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Dot(%v, %v) = %f, want %f", tt.a, tt.b, got, tt.want)
			}
		})
	}
}

func TestSigmoid(t *testing.T) {
	tests := []struct {
		name  string
		input float64
		want  float64
	}{
		{"zero", 0, 0.5},
		{"large positive", 100, 1.0},
		{"large negative", -100, 0.0},
		{"one", 1, 1.0 / (1.0 + math.Exp(-1))},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			got := Sigmoid(tt.input)
			if math.Abs(got-tt.want) > 1e-9 {
				t.Errorf("Sigmoid(%f) = %f, want %f", tt.input, got, tt.want)
			}
		})
	}
}

func TestNormalize_OutputNorm(t *testing.T) {
	vecs := [][]float64{
		{1, 2, 3, 4, 5},
		{-10, 20, -30},
		{0.001, 0.002, 0.003},
	}

	for _, v := range vecs {
		n := Normalize(v)
		norm := Norm(n)
		if math.Abs(norm-1.0) > 1e-9 {
			t.Errorf("Normalize(%v) has norm %f, want 1.0", v, norm)
		}
	}
}

func TestColumnMean_Empty(t *testing.T) {
	res := columnMean(nil)
	if res != nil {
		t.Errorf("columnMean(nil) = %v, want nil", res)
	}

	res = columnMean([][]float64{})
	if res != nil {
		t.Errorf("columnMean(empty) = %v, want nil", res)
	}
}
