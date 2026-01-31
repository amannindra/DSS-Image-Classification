python3 -c 'print("Hello, World!")'
mkdir -p /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb
aws s3 cp s3://sagemaker-us-east-1-253490779227/swinb-2026-01-30-08-06-08-868/output/model.tar.gz /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb
tar -tvf model.tar.gz
tar -xvzf model.tar.gz -C /Users/amannindra/Projects/DSS-Image-Classification/DSSAnimalClassification/swinb



