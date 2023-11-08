# AdversarialExample

Example goal: discriminate class0 vs class1 without learning features that can discriminate class0 vs class2
How to run example:

1. Generate training and validation data
   - class0: N(0, 1), N(0, 1)
   - class1: N(1, 1), N(2, 1)
   - class2: N(0, 1), N(1, 1)

   ```bash
   python GenerateDataset.py --output data/train --class-size 100000 --seed 0
   python GenerateDataset.py --output data/val --class-size 100000 --seed 1
   ```
2. Train model setting `adv_grad_factor=0` (i.e. no adversarial component)
   ```bash
   python AdversarialModel.py --adv-grad-factor 0
   ```
   Example of best model performance:
   ```
   class_loss: 0.2048 - adv_loss: 0.3892 - class_accuracy: 0.8673 - adv_accuracy: 0.6904 - val_class_loss: 0.2048 - val_adv_loss: 0.3906 - val_class_accuracy: 0.8676 - val_adv_accuracy: 0.6881
   ```
3. Train model setting `adv_grad_factor=10` (i.e. importance of the adversarial component is ten times that of the classification)
   ```bash
   python AdversarialModel.py --adv-grad-factor 10
   ```
   Example of best model performance:
   ```
   class_loss: 0.2076 - adv_loss: 0.4622 - class_accuracy: 0.8654 - adv_accuracy: 0.5018 - val_class_loss: 0.2048 - val_adv_loss: 0.4611 - val_class_accuracy: 0.8670 - val_adv_accuracy: 0.5059
   ```
