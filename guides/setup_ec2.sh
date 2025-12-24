#!/bin/bash
# Setup script for DSS ML EC2 Instance

echo "Setting up environment on EC2 instance..."

# Update system
sudo yum update -y

# Install Python 3.11 if not available
python3.11 --version || sudo amazon-linux-extras install python3.11 -y

# Create virtual environment
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate

# Install necessary packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jupyter jupyterlab notebook
pip install boto3 sagemaker
pip install pandas numpy pillow matplotlib seaborn scikit-learn

# Create workspace directory
mkdir -p ~/DSS-Image-Classification

echo "✓ Setup complete!"
echo "Activate environment with: source ~/ml-env/bin/activate"


