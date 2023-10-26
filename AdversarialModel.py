import os
import sys
import yaml
import datetime
import shutil

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

if __name__ == "__main__":
  file_dir = os.path.dirname(os.path.abspath(__file__))
  base_dir = os.path.dirname(file_dir)
  base_base_dir = os.path.dirname(base_dir)
  if base_dir not in sys.path:
    sys.path.append(base_dir)
  __package__ = os.path.split(file_dir)[-1]

from .callbacks import ModelCheckpoint

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.layers import Input, Dense
from tensorflow.keras.callbacks import CSVLogger

class AdversarialModel(keras.Model):
  '''Goal: discriminate class0 vs class1 without learning features that can discriminate class0 vs class2'''

  def __init__(self, setup, *args, **kwargs):
    super().__init__(*args, **kwargs)

    self.class_loss = keras.losses.BinaryCrossentropy(from_logits=False)
    self.adv_loss = keras.losses.BinaryCrossentropy(from_logits=False)

    self.adv_optimizer = tf.keras.optimizers.AdamW(learning_rate=setup['adv_learning_rate'])
    self.adv_grad_factor = setup['adv_grad_factor']

    self.class_loss_tracker = keras.metrics.Mean(name="class_loss")
    self.adv_loss_tracker = keras.metrics.Mean(name="adv_loss")
    self.class_accuracy = tf.keras.metrics.BinaryAccuracy(name="class_accuracy")
    self.adv_accuracy = tf.keras.metrics.BinaryAccuracy(name="adv_accuracy")

    self.common_layers = []

    for n in range(setup['n_common_layers']):
      layer = Dense(setup['n_common_units'], activation=setup['activation'], name=f'common_{n}')
      self.common_layers.append(layer)

    self.class_layers = []
    self.adv_layers = []
    for n in range(setup['n_adv_layers']):
      layer = Dense(setup['n_adv_units'], activation=setup['activation'], name=f'class_{n}')
      self.class_layers.append(layer)
      layer = Dense(setup['n_adv_units'], activation=setup['activation'], name=f'adv_{n}')
      self.adv_layers.append(layer)

    self.class_output = Dense(1, activation='sigmoid', name='class_output')
    self.adv_output = Dense(1, activation='sigmoid', name='adv_output')


  def call(self, x):
    for layer in self.common_layers:
      x = layer(x)
    x_common = x
    for layer in self.class_layers:
      x = layer(x)
    class_output = self.class_output(x)
    x = x_common
    for layer in self.adv_layers:
      x = layer(x)
    adv_output = self.adv_output(x)
    return class_output, adv_output

  def train_step(self, data):
    x, y = data

    ones = tf.ones_like(y)
    zeros = tf.zeros_like(y)
    w_class = tf.where((y == 0) | (y == 1), ones, zeros)
    w_adv = tf.where((y == 0) | (y == 1), ones, zeros)
    y_class = tf.where((y == 0) | (y == 2), zeros, ones)
    y_adv = tf.where((y == 0), ones, zeros)

    with tf.GradientTape() as class_tape, tf.GradientTape() as adv_tape:
      y_pred_class, y_pred_adv = self(x, training=True)
      class_loss_vec = self.class_loss(y_class, y_pred_class)
      class_loss = tf.reduce_mean(tf.multiply(class_loss_vec, w_class))
      adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
      adv_loss = tf.reduce_mean(tf.multiply(adv_loss_vec, w_adv))

    common_vars = [ var for var in self.trainable_variables if "/common" in var.name ]
    class_vars = [ var for var in self.trainable_variables if "/class" in var.name ]
    adv_vars = [ var for var in self.trainable_variables if "/adv" in var.name ]
    n_common_vars = len(common_vars)

    grad_class = class_tape.gradient(class_loss, common_vars + class_vars)
    grad_adv = adv_tape.gradient(adv_loss, common_vars + adv_vars)
    grad_class_excl = grad_class[n_common_vars:]
    grad_adv_excl = grad_adv[n_common_vars:]
    grad_common = [ grad_class[i] - self.adv_grad_factor * grad_adv[i] for i in range(len(common_vars)) ]

    self.optimizer.apply_gradients(zip(grad_common + grad_class_excl, common_vars + class_vars))
    self.adv_optimizer.apply_gradients(zip(grad_adv_excl, adv_vars))

    self.class_loss_tracker.update_state(class_loss)
    self.adv_loss_tracker.update_state(adv_loss)
    self.class_accuracy.update_state(y_class, y_pred_class, sample_weight=w_class)
    self.adv_accuracy.update_state(y_adv, y_pred_adv, sample_weight=w_adv)

    return { m.name: m.result() for m in self.metrics }

  def test_step(self, data):
    x, y = data

    ones = tf.ones_like(y)
    zeros = tf.zeros_like(y)
    w_class = tf.where((y == 0) | (y == 1), ones, zeros)
    w_adv = tf.where((y == 0) | (y == 1), ones, zeros)
    y_class = tf.where((y == 0) | (y == 2), zeros, ones)
    y_adv = tf.where((y == 0), ones, zeros)

    y_pred_class, y_pred_adv = self(x, training=False)
    class_loss_vec = self.class_loss(y_class, y_pred_class)
    class_loss = tf.reduce_mean(tf.multiply(class_loss_vec, w_class))
    adv_loss_vec = self.adv_loss(y_adv, y_pred_adv)
    adv_loss = tf.reduce_mean(tf.multiply(adv_loss_vec, w_adv))

    self.class_loss_tracker.update_state(class_loss)
    self.adv_loss_tracker.update_state(adv_loss)
    self.class_accuracy.update_state(y_class, y_pred_class, sample_weight=w_class)
    self.adv_accuracy.update_state(y_adv, y_pred_adv, sample_weight=w_adv)

    return { m.name: m.result() for m in self.metrics }

  @property
  def metrics(self):
    return [
          self.class_loss_tracker,
          self.adv_loss_tracker,
          self.class_accuracy,
          self.adv_accuracy,
    ]

if __name__ == "__main__":
  import argparse

  parser = argparse.ArgumentParser()
  parser.add_argument('--cfg', required=False, default='model.yaml', type=str)
  parser.add_argument('--output', required=False, default='data', type=str)
  parser.add_argument('--gpu', required=False, default='0', type=str)
  parser.add_argument('--batch-size', required=False, type=int, default=100)
  parser.add_argument('--patience', required=False, type=int, default=10)
  parser.add_argument('--n-epochs', required=False, type=int, default=10000)
  parser.add_argument('--dataset-train', required=False, default='data/train', type=str)
  parser.add_argument('--dataset-val', required=False, default='data/val', type=str)

  parser.add_argument('--summary-only', required=False, action='store_true')
  args = parser.parse_args()

  os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
  import tensorflow as tf

  with open(args.cfg) as f:
    cfg = yaml.safe_load(f)

  model = AdversarialModel(cfg)
  model.compile(loss=keras.losses.BinaryCrossentropy(from_logits=False),
                optimizer=tf.keras.optimizers.AdamW(learning_rate=cfg['learning_rate']))

  dataset_train = tf.data.Dataset.load(args.dataset_train, compression='GZIP')
  ds_train = dataset_train.batch(args.batch_size)

  dataset_val = tf.data.Dataset.load(args.dataset_val, compression='GZIP')
  ds_val = dataset_val.batch(args.batch_size)

  for data in ds_train.take(1):
    x, y = data
    model(x)
    break

  model.summary()
  if args.summary_only:
    sys.exit(0)


  output_root = 'data'
  timestamp_str = datetime.datetime.now().strftime('%Y-%m-%dT%H%M%S')
  dirFile = os.path.join(output_root, timestamp_str)
  if os.path.exists(dirFile):
    raise RuntimeError(f'Output directory {dirFile} already exists')
  os.makedirs(dirFile)

  shutil.copy(args.cfg, dirFile)
  shutil.copy('AdversarialModel.py', dirFile)

  dirFile = os.path.join(dirFile, 'model')
  print(dirFile)


  callbacks = [
    ModelCheckpoint(dirFile, verbose=1, monitor="val_class_loss", mode='min', min_rel_delta=1e-3,
                    patience=args.patience, save_callback=None),
    tf.keras.callbacks.CSVLogger(os.path.join(dirFile, 'training_log.csv'), append=True),
  ]

  model.fit(ds_train, validation_data=ds_val, callbacks=callbacks, epochs=args.n_epochs, verbose=1)

