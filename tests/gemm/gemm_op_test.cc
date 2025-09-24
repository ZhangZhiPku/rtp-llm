
#include "rtp_llm/cpp/devices/DeviceFactory.h"
#include "rtp_llm/cpp/core/torch_utils/BufferTorchUtils.h"
#include "rtp_llm/cpp/devices/OpData.h"

using namespace rtp_llm;
namespace unittest {

class GemmOp: public torch::jit::CustomClassHolder {
public:
    GemmOp();

    void forward(torch::Tensor input, torch::Tensor weight, torch::Tensor weight_scale, torch::Tensor output, std::optional<torch::Tensor> bias = std::nullopt);

private:
    DeviceBase* device = nullptr;
};

GemmOp::GemmOp() {
    rtp_llm::initLogger();
    DeviceFactory::initDevices(GptInitParameter());
    device = DeviceFactory::getDefaultDevice();
}

void GemmOp::forward(torch::Tensor input, torch::Tensor weight, torch::Tensor weight_scale, torch::Tensor output, std::optional<torch::Tensor> bias) {
    auto hidden        = torchTensor2Buffer(input);
    auto kernel        = torchTensor2Buffer(weight);
    auto scales_buffer = torchTensor2Buffer(weight_scale);
    auto bias_buffer_ptr   = bias ? torchTensor2Buffer(bias.value()) : nullptr;
    auto shape         = kernel->shape();
    auto dtype         = kernel->type();
    auto output_buffer = torchTensor2Buffer(output);

    auto weight_buffer =
        new QBuffer(BufferPtr(new Buffer(kernel->where(), dtype, shape, kernel->data())),
                    std::move(scales_buffer),
                    std::move(BufferPtr(new Buffer(MemoryType::MEMORY_GPU, DataType::TYPE_INVALID, {0}, nullptr))));
    GemmParams params{*hidden, *weight_buffer, bias ? OptionalConstBufferRef(*bias_buffer_ptr) : std::nullopt, output_buffer};
    device->gemm(params);
}
}  // namespace unittest

static auto GemmOp = torch::jit::class_<unittest::GemmOp>("unittest", "GemmOp")
                         .def(torch::jit::init<>())
                         .def("forward", &unittest::GemmOp::forward);
