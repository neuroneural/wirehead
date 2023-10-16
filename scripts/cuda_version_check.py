import tensorflow as tf
print("Built with CUDA:", tf.test.is_built_with_cuda())
print("CUDA Version:", tf.sysconfig.get_build_info()['cuda_version'])
