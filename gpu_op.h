// Partially specialize functor for GpuDevice.
struct Functor
{
    Functor() {}
    void Quantize(const float *in, short *out, const float *range, const int size);
    void Dequantize(const short *in, float *out, const float *range, const int size);
};