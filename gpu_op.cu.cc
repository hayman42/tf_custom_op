#include "gpu_op.h"
#include <algorithm>

#define N_BLOCKS 32
#define N_THREADS 1024
#define TOTAL_THREADS ((N_BLOCKS) * (N_THREADS))

using namespace std;

__global__ void quantize(const float *in, short *out, const float *range, const int size)
{
    int n_bits = 8 * sizeof(short);
    int n_nums = (1 << n_bits) - 2;
    short Z = n_nums / 2;
    int g = blockIdx.x * blockDim.x + threadIdx.x, last = TOTAL_THREADS - 1;
    int per_thread = size / TOTAL_THREADS;
    int start = g * per_thread;
    if (g == last)
    {
        per_thread += size % TOTAL_THREADS;
    }
    for (register int i = 0; i < per_thread; i++)
    {
        out[start + i] = (short)(lrintf(((in[start + i] - range[0]) * n_nums) / (range[1] - range[0])) - Z);
    }
}

__global__ void dequantize(const short *in, float *out, const float *range, const int size)
{
    int n_bits = 8 * sizeof(short);
    int n_nums = (1 << n_bits) - 2;
    short Z = n_nums / 2;
    int g = blockIdx.x * blockDim.x + threadIdx.x, last = TOTAL_THREADS - 1;
    int per_thread = size / TOTAL_THREADS;
    int start = g * per_thread;
    if (g == last)
    {
        per_thread += size % TOTAL_THREADS;
    }
    for (register int i = 0; i < per_thread; i++)
    {
        out[start + i] = ((in[start + i] + Z) * (range[1] - range[0])) / n_nums + range[0];
    }
}

void Functor::Quantize(
    const float *in, short *out, const float *range, const int size)
{
    quantize<<<N_BLOCKS, N_THREADS>>>(in, out, range, size);
    return;
}

void Functor::Dequantize(
    const short *in, float *out, const float *range, const int size)
{
    dequantize<<<N_BLOCKS, N_THREADS>>>(in, out, range, size);
    return;
}