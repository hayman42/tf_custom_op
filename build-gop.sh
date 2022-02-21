TF_CFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_compile_flags()))') )
TF_LFLAGS=( $(python -c 'import tensorflow as tf; print(" ".join(tf.sysconfig.get_link_flags()))') )


nvcc -std=c++11 -c -o gpu_op.cu.o gpu_op.cu.cc \
  ${TF_CFLAGS[@]} -D GOOGLE_CUDA=1 -x cu -Xcompiler -fPIC

nvcc -std=c++11 -shared -o gpu_op.so gpu_op.cc \
  gpu_op.cu.o ${TF_CFLAGS[@]} ${TF_LFLAGS[@]} -Xcompiler -fPIC -D_GLIBCXX_USE_CXX11_ABI=0 -DGOOGLE_CUDA=1