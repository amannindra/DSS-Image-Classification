🎯 1. Learning Curve Analysis Questions


### Is the model still improving or has it plateaued?

Based on the graph the model has plateaued at 4 epoches. The validation stopped around 87% accuracy.

I need to deeper model.

### Is there a large gap between train and val curves?

After 4 epoches there is a significant gap between train and val accuracy curves and loss. 

I need to add regularization.

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

### Are confusions symmetric (A→B and B→A) or one-way (A→B but not B→A)?

Yes majority are symmetric.


Symmetric → Classes genuinely look similar (antelope ↔ rodent?)
One-way → One class is "default" when model is uncertain

## Does the model confuse many classes with "blank"?

Yes, the model is predicting blank for alot of the misclassification, whether it is blank to antelope_duiker or rodent to blank.

Yes → "Blank" is becoming a dumping ground → Need clearer definition or better training

### Are related animals confused more (e.g., monkey vs rodent vs antelope)?

About 30 misclassifications is when monkey_prosimian is predicted as antelope_duiker and same for the opposite, however majority of the incorrect predictions are from Blank and antelope. However this can also just be due to the unbalanced class wieghts. 



Yes → Model needs better fine-grained features (try larger input size 384→512, or attention mechanisms)

### Did confusions decrease from epoch 4 to epoch 9?

THe confusion matrix decreased for blank, but for all others it has increased. However it has only increased by less than 3%, so no point training for more than 4 epochs.

💡 4. Confidence vs Performance Questions


### Are there classes with high confidence but low F1?

No there isn't any class with high confidence and low f1

📊 5. Class Imbalance Impact Questions

### Do classes with fewer samples have lower F1 scores?

No, the hog class has a high F1 while it oly contains the half the number of data at antelope_duiker


### What's the ratio between largest and smallest class?

THe largest class whih is antelope_duiker and smallest class whcih is hog has about 2.5x difference in data

There are 2474 images of antelope_duiker
There are 1641 images of bird
There are 2213 images of blank
There are 2423 images of civet_genet
There are 978 images of hog
There are 2254 images of leopard
There are 2492 images of monkey_prosimian
There are 2013 images of rodent


### Are rare classes still improving in later epochs?

No, at 4 epoches majority of the classes reach the highest point. 

🏆 6. Best Epoch Selection Questions

### Did different metrics peak at different epochs?

The epoche perferemance peaked at 4 epoches.

### Is your "best" model (by val accuracy) also best for log loss?

Ya basically, at 4 epoches the loss was 0.4 while the actual lowest log loss is at epoch 7 with a loss of 0.39.

### How much did performance vary between epochs 7-10?

Nothing at all, if anything the loss increased from epoch 8 to 9, showing clear sign of of overfitting.

### Could you have stopped training earlier without losing performance?

Yes, I could have stopped training at 4 epochs.


🎨 8. Architecture Selection Questions

### Is Top-3 accuracy much higher (>10%) than Top-1 accuracy?

No, the f1 scores are quite similar for the the top 5 classification, the only ones it is struggling in is mostly antelope_duiker and blank.

           Class       F1  Samples
             hog 0.969574    244.0
         leopard 0.957597    564.0
     civet_genet 0.955248    606.0
            bird 0.937046    410.0
          rodent 0.894539    503.0
monkey_prosimian 0.888713    623.0
 antelope_duiker 0.770008    619.0
           blank 0.660305    553.0


### Are all classes performing poorly (<0.8 F1) or just a few?

No just 2 classses performing poorly, antelope_duiker and blank

### Is the gap between train and val accuracy growing over time?

Yes, after 2 epoches, the gap between train and validation grows. At 4 epcoehs the difference in 3% however by the time it is 9 epoches its 10%

### Do confusions suggest the model needs more spatial awareness?

idk how to check that

🚀 9. Data Quality Questions

### Are the most confused classes actually visually similar?

Yes, when analyzing the antelope duiker images and blank images, you can see that alot the images is visualy similar, especially since the antelope duiker data isn't as good. Majority of the duikers are far away from the camera and blurred into the background of the tree.


### Do poorly performing classes have consistent confusion patterns?

The rodent class images are poor, even as a human I can barly see rodent

### Are blank images confused with actual animals?

Yes, based on my analysis, around 15% of misclassified Blank images are in antelope, 7% in rodent, monkey_prosimian is 3%, and leopord is 4%, etc

🔴 Top 10 Misclassifications:
            True        Predicted  Count  Pct_of_True
           blank  antelope_duiker     86    15.551537
 antelope_duiker            blank     74    11.954766
           blank           rodent     42     7.594937
           blank monkey_prosimian     33     5.967450
 antelope_duiker monkey_prosimian     32     5.169628
           blank          leopard     22     3.978300
monkey_prosimian  antelope_duiker     22     3.531300
monkey_prosimian            blank     19     3.049759
           blank      civet_genet     17     3.074141
          rodent            blank     16     3.180915
           blank             bird     12     2.169982
 antelope_duiker             bird     10     1.615509
 antelope_duiker      civet_genet     10     1.615509
     civet_genet           rodent      8     1.320132
 antelope_duiker           rodent      8     1.292407
            bird            blank      7     1.707317
         leopard            blank      7     1.241135
 antelope_duiker              hog      6     0.969305
     civet_genet            blank      5     0.825083
monkey_prosimian             bird      5     0.802568
            bird  antelope_duiker      4     0.975610
          rodent      civet_genet      4     0.795229
...
             hog           rodent      0     0.000000
             hog monkey_prosimian      0     0.000000
             hog      civet_genet      0     0.000000
           blank              hog      0     0.000000


Deployment Decision Questions


### What's your target accuracy for deployment?

95%+ 

### Which misclassifications are most costly in production?
 
blank -> antelope_duiker

🎯 Start with these 5 critical questions:

### Is there overfitting? 

Yes there is overfitting after 4 epochs

(Train-val gap) → Determines if you need regularization

Which 2-3 classes perform worst? → Focus your improvement efforts

The 3 classes that performed the worst is blanks, antelope_duiker and monkey_promian.


What are the top 3 confusions? → Shows where the model struggles

Does class size correlate with F1? → Tells you if imbalance is the main issue

Are you using the best epoch? → Might get free performance gain