# Conser-vision Practice Area: Image Classification

----

## Competition Overview:

Can you classify the wildlife species that appear in camera trap images collected by conservation researchers?

Welcome to the African jungle! In recent years, automated surveillance systems called camera traps have helped conservationists study and monitor a wide range of ecologies while limiting human interference. Camera traps are triggered by motion or heat, and passively record the behavior of species in the area without significantly disturbing their natural tendencies.

However, camera traps also generate a vast amount of data that quickly exceeds the capacity of humans to sift through. That's where machine learning can help! Advances in computer vision can help automate tasks like species detection and classification, localization, depth estimation, and individual identification so humans can more effectively learn from and protect these ecologies.

In this challenge, we will take a look at object classification for wildlife species. Classifying wildlife is an important step to sort through images, quantify observations, and quickly find those with individual species.

This is a practice competition designed to be accessible to participants at all levels. That makes it a great place to dive into the world of data science competitions and computer vision. Try your hand at image classification and see what animals your model can find!

[Competition Link](https://www.drivendata.org/competitions/87/competition-image-classification-wildlife-conservation/data/)


Example Images 

![Alt text description](DSSAnimalClassification/readImage/ZJ000006.jpg)
![Alt text description](DSSAnimalClassification/readImage/ZJ000010.jpg)
![Alt text description](DSSAnimalClassification/readImage/ZJ000011.jpg)
![Alt text description](DSSAnimalClassification/readImage/ZJ000013.jpg)

Over the past month, I have trained Resnet18, Swin Transformer, and Convnext-1 models. 

* [Resnet18 Archiecture](DSSAnimalClassification/Archiecture/Resnet18.md)
* [Resnet50 Archiecture](DSSAnimalClassification/Archiecture/Resnet50.md)
* [Swin archiecture](DSSAnimalClassification/Archiecture/Swin1.md)
* [Convnext Archiecture](DS)
* 




**Comparison with Previous Models**:
| Model | Val Accuracy | Train Accuracy | Parameters |
|-------|-------------|----------------|------------|
| ResNet18 (baseline) | 86% | 90% | 11M |
| ResNet50 | 82% | 92% | 25M |
| **Swin-T (Stage 1)** | **89%** ✅ | 97% | 28M |




