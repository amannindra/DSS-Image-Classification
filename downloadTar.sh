python3 -c 'print("START")'
mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/models
mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/output
python3 -c 'print("Making folders finished")'
aws s3 cp s3://sagemaker-us-east-1-253490779227/swinb-test-3-2026-02-01-04-38-13-049/output/model.tar.gz /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3
python3 -c 'print("S3 download finished")'
tar -tvf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/model.tar.gz
python3 -c 'print("Tar listing finished")'


tar -xzf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/model.tar.gz -C /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3
python3 -c 'print("Extraction finished")'

mv /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/**/*.pth /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/models/
mv /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassificationswinb-test3/**/*.json /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test3/output/
python3 -c 'print("Moving files finished")'
# tar -xzf /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test2/model.tar.gz -C /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb-test2/output
python3 -c 'print("All finished")'
