package extract

import "gonum.org/v1/gonum/mat"

// ExtractPCA computes the top-k principal directions from activation differences.
// Uses thin SVD on the difference matrix (positive activations minus negative mean).
func ExtractPCA(positive, negative [][]float64, k int) (directions [][]float64, singularValues []float64) {
	if len(positive) == 0 || len(negative) == 0 {
		return nil, nil
	}

	negMean := columnMean(negative)

	rows := len(positive)
	cols := len(positive[0])
	maxK := cols
	if rows < maxK {
		maxK = rows
	}
	if k > maxK {
		k = maxK
	}
	if k <= 0 {
		return nil, nil
	}

	diff := mat.NewDense(rows, cols, nil)
	for i := range positive {
		for j := range positive[i] {
			diff.Set(i, j, positive[i][j]-negMean[j])
		}
	}

	var svd mat.SVD
	if !svd.Factorize(diff, mat.SVDThin) {
		return nil, nil
	}

	// For SVDThin: U is rows×min, V is cols×min
	minDim := rows
	if cols < rows {
		minDim = cols
	}
	vt := mat.NewDense(cols, minDim, nil)
	svd.VTo(vt)

	allValues := svd.Values(nil)
	singularValues = make([]float64, k)
	for i := range singularValues {
		singularValues[i] = allValues[i]
	}

	directions = make([][]float64, k)
	for i := 0; i < k; i++ {
		colVec := make([]float64, cols)
		for rowIdx := 0; rowIdx < cols; rowIdx++ {
			colVec[rowIdx] = vt.At(rowIdx, i)
		}
		directions[i] = Normalize(colVec)
	}

	return directions, singularValues
}
