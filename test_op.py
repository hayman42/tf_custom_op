import tensorflow as tf
import numpy as np
from tensorflow.compat.v1 import Session
import sys

OP_PATH = "/workspace/tfop/gpu_op.so"


lib = tf.load_op_library(OP_PATH)
quantize = lib.quantize_to_int
dequantize = lib.dequantize_from_int


def print_op_result(x, sess):
    a = np.min(x)
    b = np.max(x)
    r = tf.stack([a, b], axis=0)
    q = quantize(x, r)
    dq = dequantize(q, r)
    df = tf.abs(tf.subtract(x, dq))
    l1_d = tf.reduce_mean(tf.abs(tf.subtract(x, dq)))
    q_res = sess.run(q)
    dq_res = sess.run(dq)
    l1_d = sess.run(l1_d)
    print("Original:\n", x, "\nQuantized:\n", q_res, "\nrange:\n", sess.run(r), "\nDequantized:\n", dq_res, "\nDiff:\n", l1_d)
    print("\nEach Diff:")
    print(sess.run(df))
    return r


def print_tf_q_result(x, r, sess):
    print("\nQuantize and dequantize with TF")
    tfdq = tf.quantization.quantize_and_dequantize_v2(x, r[0], r[1], num_bits=16)
    print(sess.run(tfdq))
    print("Diff")
    diff = tf.subtract(x, tfdq)
    df = tf.abs(diff)
    l1_d = tf.reduce_mean(tf.abs(diff))
    print(sess.run(l1_d))
    print("\nEach Diff:")
    print(sess.run(df))


x = tf.random.uniform(shape=[5, 5], dtype=tf.float32)

with Session() as sess:
    x = x.numpy()
    r = print_op_result(x, sess)
    print_tf_q_result(x, r, sess)
