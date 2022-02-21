#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/framework/register_types.h"
#include "gpu_op.h"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace tensorflow;

REGISTER_OP("QuantizeToInt")
    .Input("to_quantize: float32")
    .Input("quantize_range: float32")
    .Output("quantized_out: int16")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                { return Status::OK(); });

template <typename T>
class QuantizeToIntOp : public OpKernel
{
public:
    explicit QuantizeToIntOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input_tensor = context->input(0);
        const Tensor &range_tensor = context->input(1);
        auto input = input_tensor.flat<float>();
        auto range = range_tensor.flat<float>();

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<T>();

        Functor().Quantize(
            input.data(), output.data(), range.data(), input.size());
    }
};

REGISTER_KERNEL_BUILDER(Name("QuantizeToInt")
                            .Device(DEVICE_GPU),
                        QuantizeToIntOp<int16>);

REGISTER_OP("DequantizeFromInt")
    .Input("to_dequantize: int16")
    .Input("quantize_range: float32")
    .Output("dequantized: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                { return Status::OK(); });

template <typename T>
class DequantizeFromIntOp : public OpKernel
{
public:
    explicit DequantizeFromIntOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        const Tensor &input_tensor = context->input(0);
        const Tensor &range_tensor = context->input(1);
        auto input = input_tensor.flat<T>();
        auto range = range_tensor.flat<float>();

        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output = output_tensor->flat<float>();
        Functor().Dequantize(
            input.data(), output.data(), range.data(), input.size());
    }
};

REGISTER_KERNEL_BUILDER(Name("DequantizeFromInt")
                            .Device(DEVICE_GPU),
                        DequantizeFromIntOp<int16>);