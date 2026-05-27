#include <NvInfer.h>
#include <vector>
#include <cstring>
#include "kernel.h"

using namespace nvinfer1;

namespace burnnotice {

class BurnNoticePlugin : public IPluginV2DynamicExt {
public:
    BurnNoticePlugin() {}
    
    // Virtual destructor is required
    ~BurnNoticePlugin() override = default;

    int getNbOutputs() const noexcept override {
        return 1; // Output activation
    }

    DimsExprs getOutputDimensions(int outputIndex, const DimsExprs* inputs, int nbInputs, IExprBuilder& exprBuilder) noexcept override {
        return inputs[0]; // Same dimension as input activation
    }

    int initialize() noexcept override {
        return 0;
    }

    void terminate() noexcept override {}

    size_t getWorkspaceSize(const PluginTensorDesc* inputs, int nbInputs, const PluginTensorDesc* outputs, int nbOutputs) const noexcept override {
        return 0; // No extra workspace needed
    }

    int enqueue(const PluginTensorDesc* inputDesc, const PluginTensorDesc* outputDesc, const void* const* inputs, void* const* outputs, void* workspace, cudaStream_t stream) noexcept override {
        int dim = inputDesc[0].dims.d[0]; // Assuming 1D input

        const float* activation = static_cast<const float*>(inputs[0]);
        const float* direction = static_cast<const float*>(inputs[1]);
        const float theta = 0.1f; // Fixed theta for demonstration, would typically be passed as param

        float* output_activation = static_cast<float*>(outputs[0]);

        gpu_dynamic_rotation(
            const_cast<float*>(activation),
            direction,
            dim,
            theta,
            output_activation
        );

        return 0;
    }

    size_t getSerializationSize() const noexcept override {
        return 0;
    }

    void serialize(void* buffer) const noexcept override {}

    bool supportsFormatCombination(int pos, const PluginTensorDesc* inOut, int nbInputs, int nbOutputs) noexcept override {
        return inOut[pos].type == DataType::kFLOAT && inOut[pos].format == TensorFormat::kLINEAR;
    }

    const char* getPluginType() const noexcept override {
        return "BurnNoticePlugin";
    }

    const char* getPluginVersion() const noexcept override {
        return "1";
    }

    void destroy() noexcept override {
        delete this;
    }

    IPluginV2DynamicExt* clone() const noexcept override {
        return new BurnNoticePlugin();
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

    DataType getOutputDataType(int index, const nvinfer1::DataType* inputTypes, int nbInputs) const noexcept override {
        return DataType::kFLOAT;
    }

    void configurePlugin(const DynamicPluginTensorDesc* in, int nbInputs, const DynamicPluginTensorDesc* out, int nbOutputs) noexcept override {}

private:
    std::string mNamespace;
};

class BurnNoticePluginCreator : public IPluginCreator {
public:
    BurnNoticePluginCreator() {}

    const char* getPluginName() const noexcept override {
        return "BurnNoticePlugin";
    }

    const char* getPluginVersion() const noexcept override {
        return "1";
    }

    const PluginFieldCollection* getFieldNames() noexcept override {
        return &mFC;
    }

    IPluginV2DynamicExt* createPlugin(const char* name, const PluginFieldCollection* fc) noexcept override {
        return new BurnNoticePlugin();
    }

    IPluginV2DynamicExt* deserializePlugin(const char* name, const void* serialData, size_t serialLength) noexcept override {
        return new BurnNoticePlugin();
    }

    void setPluginNamespace(const char* pluginNamespace) noexcept override {
        mNamespace = pluginNamespace;
    }

    const char* getPluginNamespace() const noexcept override {
        return mNamespace.c_str();
    }

private:
    std::string mNamespace;
    PluginFieldCollection mFC{};
};

REGISTER_TENSORRT_PLUGIN(BurnNoticePluginCreator);

} // namespace burnnotice
