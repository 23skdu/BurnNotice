#include "kernel.h"
#include <cmath>

extern "C" {

void gpu_cosine_similarity(const float* activation, const float* directions, int dim, int num_directions, float* output_scores) {
    if (dim <= 0 || num_directions <= 0) return;

    // Compute activation norm
    float act_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        act_norm_sq += activation[i] * activation[i];
    }
    float act_norm = std::sqrt(act_norm_sq);
    if (act_norm < 1e-10f) act_norm = 1e-10f;

    // Compute cosine similarity for each concept direction (simulating parallel GPU thread blocks)
    for (int d = 0; d < num_directions; ++d) {
        float dot = 0.0f;
        float dir_norm_sq = 0.0f;
        int offset = d * dim;

        for (int i = 0; i < dim; ++i) {
            dot += activation[i] * directions[offset + i];
            dir_norm_sq += directions[offset + i] * directions[offset + i];
        }

        float dir_norm = std::sqrt(dir_norm_sq);
        if (dir_norm < 1e-10f) dir_norm = 1e-10f;

        float similarity = dot / (act_norm * dir_norm);
        // Scaled Sigmoid mapping typical of production BurnNotice policy engines
        output_scores[d] = 1.0f / (1.0f + std::exp(-similarity * 10.0f));
    }
}

void gpu_dynamic_rotation(float* activation, const float* direction, int dim, float theta, float* output_activation) {
    if (dim <= 0) return;

    // Normalize the steering direction vector
    float dir_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        dir_norm_sq += direction[i] * direction[i];
    }
    float dir_norm = std::sqrt(dir_norm_sq);
    if (dir_norm < 1e-10f) dir_norm = 1e-10f;

    // Compute dot product (projection)
    float proj = 0.0f;
    for (int i = 0; i < dim; ++i) {
        proj += activation[i] * direction[i] / dir_norm;
    }

    // Compute orthogonal component
    float* ortho = new float[dim];
    float ortho_norm_sq = 0.0f;
    for (int i = 0; i < dim; ++i) {
        ortho[i] = activation[i] - proj * (direction[i] / dir_norm);
        ortho_norm_sq += ortho[i] * ortho[i];
    }
    float ortho_norm = std::sqrt(ortho_norm_sq);
    if (ortho_norm < 1e-10f) ortho_norm = 1e-10f;

    float cos_t = std::cos(theta);
    float sin_t = std::sin(theta);

    // Dynamic rotation: rotated = cos(theta)*act + sin(theta)*(ortho_norm * normalized_dir - proj * normalized_ortho)
    for (int i = 0; i < dim; ++i) {
        float norm_dir = direction[i] / dir_norm;
        float norm_ortho = ortho[i] / ortho_norm;
        output_activation[i] = cos_t * activation[i] + sin_t * (ortho_norm * norm_dir - proj * norm_ortho);
    }

    delete[] ortho;
}

}
