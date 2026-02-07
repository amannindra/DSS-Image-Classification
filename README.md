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

Over the past month, I have trained multiple state-of-the-art models for wildlife classification:

* [ResNet18 Architecture](DSSAnimalClassification/Archiecture/Resnet18.md) - Baseline CNN model
* [Swin Transformer Architecture (Stage 1)](DSSAnimalClassification/Archiecture/Swin1.md) - Vision Transformer with shifted windows
* [Swin Transformer Architecture (Stage 3)](DSSAnimalClassification/Archiecture/Swin3.md) - Advanced configuration
* [EVA Large Architecture](DSSAnimalClassification/Archiecture/eva.md) - Enhanced Vision Transformer (Best Model) ✅



**Model Performance Comparison**:
| Model | Val Accuracy | Train Accuracy | Generalization Gap | Parameters | Resolution | Log Loss
|-------|-------------|----------------|-------------------|------------|------------|------|
| ResNet18 (baseline) | 88.50% | 98.37% | +9.87% | 11M | 224 | NA |
| Swin-T (Stage 1) | 91.61% | 99.08% | +7.47% | 28M | 224 | 2.1786 |
| **EVA-Large** ✅ | **91.27%** | **90.63%** | **-0.64%** | **300M** | **336** | 0.6695 |

**Why EVA is the Best Model**:
- **Superior Generalization**: Only model with negative generalization gap (validation accuracy exceeds training accuracy)
- **No Overfitting**: Perfect balance between model capacity and regularization
- **Robust Performance**: Trained on ImageNet-22K (14M images) → ImageNet-1K fine-tuned weights
- **Higher Resolution**: 336×336 input captures finer details for species classification
- **Advanced Architecture**: Vision Transformer with enhanced attention mechanisms
- **Production Ready**: Demonstrates consistent performance suitable for real-world deployment

## Achieved Top 4 out 500+ teams with EVA-Large Archiecture:

![Alt text description](image.png)



