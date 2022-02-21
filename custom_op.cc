#include "tensorflow/core/framework/op.h"
#include "tensorflow/core/framework/shape_inference.h"
#include "tensorflow/core/framework/op_kernel.h"
#include "tensorflow/core/platform/types.h"

#include <iostream>
#include <cmath>
#include <algorithm>

using namespace tensorflow;

REGISTER_OP("ZeroOut")
    .Input("to_zero: int32")
    .Output("zeroed: int32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                {
      c->set_output(0, c->input(0));
      return Status::OK(); });

class ZeroOutOp : public OpKernel
{
public:
    explicit ZeroOutOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_tensor = context->input(0);
        auto input = input_tensor.flat<int32>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_tensor.shape(),
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<int32>();

        // Set all but the first element of the output tensor to 0.
        const int N = input.size();
        for (int i = 1; i < N; i++)
        {
            output_flat(i) = 0;
        }

        // Preserve the first input value if possible.
        if (N > 0)
            output_flat(0) = input(0);
    }
};

REGISTER_KERNEL_BUILDER(Name("ZeroOut").Device(DEVICE_CPU), ZeroOutOp);

REGISTER_OP("QuantizeToInt")
    .Input("to_quantize: float32")
    .Output("quantized_out: int16")
    .Output("quantize_range: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                { return Status::OK(); });

class QuantizeToIntOp : public OpKernel
{
public:
    explicit QuantizeToIntOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_x = context->input(0);
        auto input = input_x.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        Tensor *output_s = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_x.shape(),
                                                         &output_tensor));
        OP_REQUIRES_OK(context, context->allocate_output(1, {2}, &output_s));
        auto output_flat = output_tensor->flat<int16>();
        auto output_s_flat = output_s->flat<float>();

        const int N = input.size();
        int n = 65534;
        short int z = -32767;
        float a, b;
        a = b = input(0);
        for (int i = 1; i < N; i++)
        {
            a = std::min(a, input(i));
            b = std::max(b, input(i));
        }
        // a = 0; // why?
        output_s_flat(0) = a;
        output_s_flat(1) = b;

        // quantization scheme: r/S+z=q
        // S=(b-a)/n
        // q=(r-a)*n/(b-a)+z

        for (int i = 0; i < N; i++)
        {
            output_flat(i) = (short int)(std::round(((input(i) - a) * n) / (b - a)) + (int)z);
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("QuantizeToInt").Device(DEVICE_CPU), QuantizeToIntOp);

REGISTER_OP("DequantizeFromInt")
    .Input("to_dequantize: int16")
    .Input("quantize_range: float32")
    .Output("dequantized: float32")
    .SetShapeFn([](::tensorflow::shape_inference::InferenceContext *c)
                { return Status::OK(); });

class DequantizeFromIntOp : public OpKernel
{
public:
    explicit DequantizeFromIntOp(OpKernelConstruction *context) : OpKernel(context) {}

    void Compute(OpKernelContext *context) override
    {
        // Grab the input tensor
        const Tensor &input_x = context->input(0);
        const Tensor &q_range = context->input(1);
        auto input = input_x.flat<int16>();
        auto range = q_range.flat<float>();

        // Create an output tensor
        Tensor *output_tensor = NULL;
        OP_REQUIRES_OK(context, context->allocate_output(0, input_x.shape(),
                                                         &output_tensor));
        auto output_flat = output_tensor->flat<float>();
        const int N = input.size();
        int n = 65534;
        short int z = -32767;
        float a = range(0), b = range(1);
        // dequantization scheme: r/S+z=q r=(q-z)*S
        // S=(b-a)/n
        // r=(q-z)(b-a)/n+a

        for (int i = 0; i < N; i++)
        {
            output_flat(i) = ((input(i) - z) * (b - a)) / n + a;
        }
    }
};

REGISTER_KERNEL_BUILDER(Name("DequantizeFromInt").Device(DEVICE_CPU), DequantizeFromIntOp);