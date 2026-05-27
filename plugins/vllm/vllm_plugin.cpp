#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include "kernel.h"

namespace py = pybind11;

// PyBind11 wrapper for gpu_cosine_similarity
py::array_t<float> py_gpu_cosine_similarity(py::array_t<float> activation, py::array_t<float> directions) {
    py::buffer_info act_buf = activation.request();
    py::buffer_info dir_buf = directions.request();

    if (act_buf.ndim != 1) {
        throw std::runtime_error("Activation must be a 1D array");
    }
    if (dir_buf.ndim != 2) {
        throw std::runtime_error("Directions must be a 2D array");
    }

    int dim = act_buf.shape[0];
    int num_directions = dir_buf.shape[0];

    if (dir_buf.shape[1] != dim) {
        throw std::runtime_error("Direction dimension must match activation dimension");
    }

    py::array_t<float> output_scores(num_directions);
    py::buffer_info out_buf = output_scores.request();

    gpu_cosine_similarity(
        static_cast<float*>(act_buf.ptr),
        static_cast<float*>(dir_buf.ptr),
        dim,
        num_directions,
        static_cast<float*>(out_buf.ptr)
    );

    return output_scores;
}

// PyBind11 wrapper for gpu_dynamic_rotation
py::array_t<float> py_gpu_dynamic_rotation(py::array_t<float> activation, py::array_t<float> direction, float theta) {
    py::buffer_info act_buf = activation.request();
    py::buffer_info dir_buf = direction.request();

    if (act_buf.ndim != 1 || dir_buf.ndim != 1) {
        throw std::runtime_error("Inputs must be 1D arrays");
    }

    int dim = act_buf.shape[0];

    if (dir_buf.shape[0] != dim) {
        throw std::runtime_error("Direction dimension must match activation dimension");
    }

    py::array_t<float> output_activation(dim);
    py::buffer_info out_buf = output_activation.request();

    gpu_dynamic_rotation(
        static_cast<float*>(act_buf.ptr),
        static_cast<float*>(dir_buf.ptr),
        dim,
        theta,
        static_cast<float*>(out_buf.ptr)
    );

    return output_activation;
}

PYBIND11_MODULE(burnnotice_vllm_plugin, m) {
    m.doc() = "BurnNotice vLLM CUDA Plugin"; // optional module docstring
    m.def("gpu_cosine_similarity", &py_gpu_cosine_similarity, "Compute parallel cosine similarity scores on GPU VRAM");
    m.def("gpu_dynamic_rotation", &py_gpu_dynamic_rotation, "Execute norm-preserving 2D dynamic coordinate rotations directly inside GPU space");
}
