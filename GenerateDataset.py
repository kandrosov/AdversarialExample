import os

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
os.environ["CUDA_VISIBLE_DEVICES"] = '-1'
import tensorflow as tf
import numpy as np

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--output', required=True, type=str)
  parser.add_argument('--class-size', required=True, type=int)
  parser.add_argument('--seed', required=True, type=int)
  args = parser.parse_args()

  np.random.seed(args.seed)
  N = args.class_size

  data = np.zeros((3 * N, 3), dtype=np.float32)
  data[:N, 0] = 0
  data[:N, 1] = np.random.normal(loc=0., scale=1., size=N)
  data[:N, 2] = np.random.normal(loc=0., scale=1., size=N)
  data[N:2*N, 0] = 1
  data[N:2*N, 1] = np.random.normal(loc=0.5, scale=1., size=N)
  data[N:2*N, 2] = np.random.normal(loc=0.5, scale=1., size=N)
  data[2*N:3*N, 0] = 2
  data[2*N:3*N, 1] = np.random.normal(loc=0., scale=1., size=N)
  data[2*N:3*N, 2] = np.random.normal(loc=0.5, scale=1., size=N)

  np.random.shuffle(data)

  x = data[:, 1:]
  y = data[:, 0]
  dataset = tf.data.Dataset.from_tensor_slices((x, y))
  dataset.save(args.output, compression='GZIP')