from tensorflow import keras
import numpy as np
import shutil
import os

class ModelCheckpoint(keras.callbacks.Callback):
  def __init__(self, filepath, monitor="val_loss", verbose=0, mode="min", min_delta=None, min_rel_delta=None,
               save_callback=None, patience=None):
    super().__init__()
    self.monitor = monitor
    self.verbose = verbose
    self.filepath = filepath
    self.epochs_since_last_save = 0
    self.msg = None
    self.save_callback = save_callback
    self.patience = patience

    if os.path.exists(filepath):
      shutil.rmtree(filepath)

    self.best = None
    self.monitor_op = self._make_monitor_op(mode, min_delta, min_rel_delta)

  def _make_monitor_op(self, mode, min_delta, min_rel_delta):
    if mode == "min":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or best - current > 0
      if min_delta is None:
        return lambda current, best: best is None or (best - current) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or best - current > min_delta
      return lambda current, best: best is None or (best - current) > min_rel_delta * best or best - current > min_delta
    elif mode == "max":
      if min_delta is None and min_rel_delta is None:
        return lambda current, best: best is None or current - best > 0
      if min_delta is None:
        return lambda current, best: best is None or (current - best) > min_rel_delta * best
      if min_rel_delta is None:
        return lambda current, best: best is None or current - best > min_delta
      return lambda current, best: best is None or (current - best) > min_rel_delta * best or current - best > min_delta
    else:
      raise ValueError(f"Unrecognized mode: {mode}")

  def _print_msg(self):
    if self.msg is not None:
      print(self.msg)
      self.msg = None

  def on_epoch_begin(self, epoch, logs=None):
    self._print_msg()

  def on_train_end(self, logs=None):
    self._print_msg()

  def on_epoch_end(self, epoch, logs=None):
    self.epochs_since_last_save += 1
    current = logs.get(self.monitor)
    if self.monitor_op(current, self.best):
      dir_name = f'epoch_{epoch+1}'
      path = os.path.join(self.filepath, dir_name)
      if self.save_callback is None:
        self.model.save(path)
      else:
        self.save_callback(self.model, path)
      path_best = os.path.join(self.filepath, 'best')
      if os.path.exists(path_best):
        os.remove(path_best)
      os.symlink(dir_name, path_best)

      if self.verbose > 0:
        self.msg = f"\nEpoch {epoch+1}: {self.monitor} "
        if self.best is None:
          self.msg += f"= {current:.5f}."
        else:
          self.msg += f"improved from {self.best:.5f} to {current:.5f} after {self.epochs_since_last_save} epochs."
        self.msg += f" Saving model to {path}\n"
      self.best = current
      self.epochs_since_last_save = 0
    if self.patience is not None and self.epochs_since_last_save >= self.patience:
      self.model.stop_training = True
      if self.verbose > 0:
        if self.msg is None:
          self.msg = '\n'
        self.msg = f"Epoch {epoch+1}: early stopping after {self.epochs_since_last_save} epochs."
