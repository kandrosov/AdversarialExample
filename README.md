# AdversarialExample

Example goal: discriminate class0 vs class1 without learning features that can discriminate class0 vs class2
How to run example:

1. Generate training and validation data
   - class0: N(0, 1), N(0, 1)
   - class1: N(1, 1), N(2, 1)
   - class2: N(0, 1), N(2, 1)

   ```bash
   python GenerateDataset.py --output data/train --class-size 10000 --seed 0
   python GenerateDataset.py --output data/val --class-size 10000 --seed 1
   ```
2. Train model setting `adv_grad_factor=0` (i.e. no adversarial component)
   ```bash
   python AdversarialModel.py --adv-grad-factor 0
   ```
   Example of best model performance:
   ```
   class_loss: 0.3136 - adv_loss: 0.2178 - class_accuracy: 0.7599 - adv_accuracy: 0.8300 - val_class_loss: 0.3169 - val_adv_loss: 0.2197 - val_class_accuracy: 0.7207 - val_adv_accuracy: 0.8432
   ```
3. Train model setting `adv_grad_factor=2` (i.e. importance of the adversarial component is twice that of the classification)
   ```bash
   python AdversarialModel.py --adv-grad-factor 2
   ```
   Example of best model performance:
   ```
   class_loss: 0.3271 - adv_loss: 0.4130 - class_accuracy: 0.7407 - adv_accuracy: 0.5025 - val_class_loss: 0.3269 - val_adv_loss: 0.4277 - val_class_accuracy: 0.7075 - val_adv_accuracy: 0.5000
   ```
