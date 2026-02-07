# import boto3
import sagemaker

import boto3
from sagemaker.pytorch import PyTorch
from sagemaker.pytorch.processing import PyTorchProcessor
from sagemaker.processing import ProcessingInput, ProcessingOutput

# Configuration
REGION = "us-east-1"
ROLE_ARN = (
    "arn:aws:iam::253490779227:role/service-role/AmazonSageMakerAdminIAMExecutionRole"
)
BUCKET = "animal-classification-dss-works"
BUCKET_VIRGINIA = "animal-classification-virgina"
S3_INPUT_DATA = f"s3://{BUCKET}/data/"
S3_PREPROCESSED = f"s3://{BUCKET_VIRGINIA}/processed"
S3_SHORT_PREPROCESSED = f"s3://{BUCKET}/short_processed"

boto_session = boto3.Session(region_name=REGION)
sagemaker_session = sagemaker.Session(boto_session=boto_session)


train_file = "DSSAnimalClassification/dss_model.py"

# 📁 Storage Configuration:
# - Models (.pth files) → output_path (automatically uploaded by SageMaker)
# - Metrics (JSON/CSV) → output_data_config (training logs and metrics)


models = [
    "swinb",
    "swinb-test-1",
    "swinb-test-2",
    "swinb-test-3",
    "swinb-test-4",
    "swinb-final",
    "swinb-final-2",
    "swinb-final-3",
    "convnext-large",
    "all-2",
    "all-3",
    "all-4",
    "all-5",
    "all-6",
    "all-7",
    "all-8",
    "all-9",
]

model_output_path = f"s3://amazon-sagemaker-253490779227-us-east-1-cnizlxa57lpnon/animal-classification-{models[-1]}"
metrics_output_path = f"s3://animal-classification-virgina/{models[-1]}_output"

train_file = "DSSAnimalClassification/dss_model.py"


estimator_3 = PyTorch(
    entry_point=train_file,
    # source_dir=".",  # Upload all .py files in directory (respects .sagemakerignore)
    dependencies=[
        "DSSAnimalClassification/requirements.txt",
        "DSSAnimalClassification/dss_datasets.py",
        "DSSAnimalClassification/dss_train_val.py",
        "DSSAnimalClassification/dss_model.py",
        "DSSAnimalClassification/dss_util.py",
        "DSSAnimalClassification/dss_multi_gpu.py",
    ],
    role=ROLE_ARN,
    framework_version="2.1",
    py_version="py310",
    output_data_config={"S3OutputPath": metrics_output_path},
    instance_count=1,
    instance_type="ml.p3.8xlarge",
    # model_data = model_location,
    hyperparameters={
        "epochs": 8,
        # "batch-size": 16,
        "learning-rate": 1e-5,
        "use-cuda": True,
        "image-size": 224,
        "weight-decay": 1e-4,
        "save-file": f"{models[-1]}Weights.pth",
    },
    sagemaker_session=sagemaker_session,
    base_job_name=models[-1],
)

print(f"✓ Estimator configured:")
print(f"  Models → {model_output_path}")
print(f"  Metrics → {metrics_output_path}")


estimator_3.fit(
    {"training": S3_PREPROCESSED},
    wait=True,  # ✅ Wait for job to complete
    logs="All",  # ✅ Stream ALL logs to notebook (shows all print statements!)
)
