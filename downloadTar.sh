# python3 -c 'print("START")'
# mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/models
# mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/output
# mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp
# python3 -c 'print("Making folders finished")'
# aws s3 cp s3://sagemaker-us-east-1-253490779227/convnext-large-2026-02-03-18-50-08-651/output/model.tar.gz /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp
# python3 -c 'print("S3 download finished")'
# tar -tvf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp/model.tar.gz
# python3 -c 'print("Tar listing finished")'


# tar -xzf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp/model.tar.gz -C /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp
# python3 -c 'print("Extraction finished")'

# find /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp \
#   -name "*.pth" -exec mv {} /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/models/ \;

# find /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp \
#   -name "*.json" -exec mv {} /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/output/ \;


# python3 -c 'print("Moving files finished")'

# rm -rf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/convnext-1/temp

# # tar -xzf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test2/model.tar.gz -C /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test2/output
# python3 -c 'print("All finished")'
