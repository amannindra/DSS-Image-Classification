🎯 1. Learning Curve Analysis Questions


### Is the model still improving or has it plateaued?

Based on the graph the model has plateaued at 4 epoches.The validation stopped around 87% accuracy.

### Is there a large gap between train and val curves?

After 4 epoches there is a significant gap between train and val accuracy curves and loss. 

Large gap → Overfitting (need regularization: dropout, weight decay, data augmentation)
Small gap → Good generalization (can potentially reduce regularization)

### Does validation loss start increasing while train loss decreases?

The validation loss increases while the training loss continues to drop after 7 epoches

Yes → Classic overfitting, stop training earlier


### Are the curves smooth or erratic?

THe curves are smooothing, showing agood learning rate.

Erratic → Learning rate too high, reduce it
Smooth → Good learning rate

🔍 2. Per-Class Performance Questions

### Which classes have consistently low F1 scores (<0.7)?

The only class which has consistent low f1 scores in blank.

These are your problem classes → Need targeted fixes


### Do certain classes improve slowly or not at all?

Antelope_duker, monkey_prosimian, bird, hog, and rodent improve significantly. Lepord and civet_genet improved a little bit. Blank improved a medium amount.

Overall after 4 epochs the classes didn't improve at all.

Slow improvement → Class needs more/better training data
No improvement → Features don't distinguish this class wel

Classes that improved significantly means model CAN learn them given enough data.


### Are low-performing classes also rare (low support)?


No, the number of blank images are relativly high from the eight choices. Also performed a significant increase in performance when training, with an f1 score of 0.8 to 0.95 in 4 epochs

Yes → Class imbalance problem → Use weighted loss, oversampling, or focal loss
No → Visual similarity problem → Need better features or data augmentation


🎭 3. Confusion Matrix Questions

### What are the top 3 most confused class pairs?

These pairs are visually similar to your model

❓ Are confusions symmetric (A→B and B→A) or one-way (A→B but not B→A)?




Symmetric → Classes genuinely look similar (antelope ↔ rodent?)
One-way → One class is "default" when model is uncertain
❓ Does the model confuse many classes with "blank"?
Yes → "Blank" is becoming a dumping ground → Need clearer definition or better training
❓ Are related animals confused more (e.g., monkey vs rodent vs antelope)?
Yes → Model needs better fine-grained features (try larger input size 384→512, or attention mechanisms)
❓ Did confusions decrease from epoch 0 to epoch 9?
Decreased → Model is learning discriminative features ✅
Same or increased → Model can't distinguish these classes → Need architectural change
Model Improvement Actions:
Persistent A↔B confusion → Use hard negative mining or siamese network to learn differences
Many classes → blank → Remove blank class or use it only for truly empty images
Systematic confusion pattern → Add attention mechanism to focus on distinguishing features