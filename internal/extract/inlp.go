package extract

// ExtractINLP computes k orthogonal directions of deception using Iterative Nullspace Projection.
func ExtractINLP(positive, negative [][]float64, k int) [][]float64 {
	if len(positive) == 0 || len(negative) == 0 || k <= 0 {
		return nil
	}

	dim := len(positive[0])
	if k > dim {
		k = dim
	}

	// Copy dataset to avoid mutating input slices
	posProj := make([][]float64, len(positive))
	for i := range positive {
		posProj[i] = make([]float64, len(positive[i]))
		copy(posProj[i], positive[i])
	}
	negProj := make([][]float64, len(negative))
	for i := range negative {
		negProj[i] = make([]float64, len(negative[i]))
		copy(negProj[i], negative[i])
	}

	var directions [][]float64

	for step := 0; step < k; step++ {
		// 1. Extract the top principal component of the difference matrix using PCA/SVD
		pcaDirs, _ := ExtractPCA(posProj, negProj, 1)
		if len(pcaDirs) == 0 {
			break
		}
		v := pcaDirs[0]
		if Norm(v) < 1e-10 {
			break
		}

		// 2. Gram-Schmidt orthogonalization against previously extracted directions
		orthoV := make([]float64, len(v))
		copy(orthoV, v)
		for _, prev := range directions {
			proj := Dot(orthoV, prev)
			for i := range orthoV {
				orthoV[i] -= proj * prev[i]
			}
		}

		orthoV = Normalize(orthoV)
		if Norm(orthoV) < 1e-10 {
			break
		}

		directions = append(directions, orthoV)

		// 3. Project current datasets onto the nullspace of orthoV
		for i := range posProj {
			proj := Dot(posProj[i], orthoV)
			for j := range posProj[i] {
				posProj[i][j] -= proj * orthoV[j]
			}
		}
		for i := range negProj {
			proj := Dot(negProj[i], orthoV)
			for j := range negProj[i] {
				negProj[i][j] -= proj * orthoV[j]
			}
		}
	}

	return directions
}
