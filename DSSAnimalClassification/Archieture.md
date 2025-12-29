# DSS Animal Classification - ML Pipeline Architecture

## Achieved 74.9% accuracy with base ResNet18. Pipeline in baseArchiecture

## https://arxiv.org/pdf/2201.03545

## Achieved 64.9% accuracy with archiecture and mixup

[METRICS] epoch=4 train_loss=0.9911 train_acc=0.70 val_loss=1.0311 val_acc=0.64


## Achieved 55% accuracy

Changed train transform to RandomResizedCrops because Resize to 224 pixels might remove the small images. Remove GrayScale transformation, as it makes colorJitter inefective. Changed Validation imagae resize to 384 pixels. Removed Schedular as doesn't have an effect in this traing. Schedular only works when I have more than 10 epochs.




Before: 

 train_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=output_channels),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

 val_transform = transforms.Compose(
        [
            transforms.Resize((224, 224)),
            transforms.Grayscale(num_output_channels=output_channels),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    


After: 

img_size = 384


train_transform = transforms.Compose(
        [
            transforms.RandomResizedCrop(
                img_size,
                scale=(0.75, 1.0),
                ratio=(0.9, 1.1)
            ),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(10),
            transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

val_transform = transforms.Compose(
        [
            transforms.Resize((img_size, img_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

[METRICS] epoch=4 train_loss=0.8954 train_acc=0.67 val_loss=1.2715 val_acc=0.55

# Achieved 60% accuracy 

Changed image size back to 224

[METRICS] epoch=4 train_loss=0.7569 train_acc=0.73 val_loss=1.1257 val_acc=0.60




https://arxiv.org/pdf/2011.11778

