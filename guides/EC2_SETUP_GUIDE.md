# EC2 Instance Setup for VS Code - DSS Image Classification

## ✅ Setup Complete!

Your EC2 instance is ready to use with VS Code!

### Instance Details:
- **Instance ID**: `i-0322d9d5fc9ccb5e8`
- **Instance Type**: `t3.xlarge` (4 vCPUs, 16GB RAM)
- **Public DNS**: `ec2-54-193-175-190.us-west-1.compute.amazonaws.com`
- **Region**: `us-west-1`
- **SSH Key**: `~/.ssh/dss-vscode-ml-key.pem`
- **Security Group**: `sg-0d07fd322590a6465`
- **Storage**: 100GB GP3

---

## 🚀 How to Connect from VS Code

### Step 1: Install VS Code Extension

1. Open VS Code
2. Go to Extensions (Cmd+Shift+X)
3. Search for **"Remote - SSH"**
4. Install it (by Microsoft)

### Step 2: Connect to EC2

1. Press `Cmd+Shift+P` (Command Palette)
2. Type: **"Remote-SSH: Connect to Host"**
3. Select **`dss-ml-aws`** from the list
4. A new VS Code window will open connected to your EC2 instance!

### Step 3: Open Your Project

Once connected:
1. Click "Open Folder"
2. Navigate to `/home/ec2-user/DSS-Image-Classification`
3. Start coding!

---

## 📋 Initial Setup on EC2

### Option 1: Run Setup Script (Automated)

```bash
# From your local machine, copy and run setup script
scp ~/Projects/DSS-Image-Classification/setup_ec2.sh dss-ml-aws:~/ 
ssh dss-ml-aws "chmod +x ~/setup_ec2.sh && ~/setup_ec2.sh"
```

### Option 2: Manual Setup

Connect via terminal and run:

```bash
ssh dss-ml-aws

# Create virtual environment
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate

# Install packages
pip install --upgrade pip
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
pip install jupyter jupyterlab notebook
pip install boto3 sagemaker pandas numpy pillow matplotlib seaborn scikit-learn

# Create project directory
mkdir -p ~/DSS-Image-Classification
```

---

## 📂 Upload Your Project Files

### Method 1: Using SCP (from local terminal)

```bash
# Upload entire project
scp -r ~/Projects/DSS-Image-Classification dss-ml-aws:~/

# Or upload specific files
scp ~/Projects/DSS-Image-Classification/*.ipynb dss-ml-aws:~/DSS-Image-Classification/
scp ~/Projects/DSS-Image-Classification/train.py dss-ml-aws:~/DSS-Image-Classification/
```

### Method 2: Using VS Code (Recommended)

Once connected via Remote-SSH in VS Code:
1. Just drag and drop files into the VS Code file explorer!
2. Or use the upload button in the file explorer

### Method 3: Git Clone

```bash
ssh dss-ml-aws
cd ~/DSS-Image-Classification
# If you have a git repo:
git clone <your-repo-url> .
```

---

## 🏃 Running Jupyter Notebooks

### In VS Code (Recommended):

1. Open your `.ipynb` file in VS Code
2. Select Python interpreter: `/home/ec2-user/ml-env/bin/python`
3. Run cells directly in VS Code!

### Via Jupyter Server:

```bash
ssh dss-ml-aws
source ~/ml-env/bin/activate
cd ~/DSS-Image-Classification
jupyter notebook --no-browser --port=8888

# In another terminal on your local machine:
ssh -L 8888:localhost:8888 dss-ml-aws

# Then open in browser: http://localhost:8888
```

---

## 📊 Working with S3 Data

Your data is already in S3! Access it from the EC2 instance:

```python
import boto3
from PIL import Image
from io import BytesIO

s3_client = boto3.client('s3', region_name='us-west-1')
bucket = 'animal-classification-dss-works'

# Read image from S3
response = s3_client.get_object(Bucket=bucket, Key='data/train_features/ZJ000005.jpg')
image = Image.open(BytesIO(response['Body'].read()))
```

---

## 💰 Cost Management

### Current Costs:
- **t3.xlarge**: ~$0.166/hour
- **100GB GP3 Storage**: ~$8/month
- **Data Transfer**: Varies

### To Stop Instance (to save money):

```bash
# Stop the instance (can restart later, keeps data)
aws ec2 stop-instances --instance-ids i-0322d9d5fc9ccb5e8 --region us-west-1

# Start it again
aws ec2 start-instances --instance-ids i-0322d9d5fc9ccb5e8 --region us-west-1

# Get new public DNS after restart
aws ec2 describe-instances --instance-ids i-0322d9d5fc9ccb5e8 \
  --region us-west-1 --query 'Reservations[0].Instances[0].PublicDnsName' --output text
```

**Important**: After restarting, the public DNS changes! Update your SSH config.

### To Terminate (Delete Everything):

```bash
# WARNING: This deletes the instance permanently!
aws ec2 terminate-instances --instance-ids i-0322d9d5fc9ccb5e8 --region us-west-1
```

---

## 🔧 Troubleshooting

### Can't Connect?

```bash
# Test SSH connection
ssh dss-ml-aws

# If that fails, check instance status
aws ec2 describe-instance-status --instance-ids i-0322d9d5fc9ccb5e8 --region us-west-1
```

### Permission Denied?

```bash
# Check key permissions
chmod 400 ~/.ssh/dss-vscode-ml-key.pem

# Verify SSH config
cat ~/.ssh/config | grep -A 8 "dss-ml-aws"
```

### Update SSH Config After Restart:

```bash
# Get new DNS
NEW_DNS=$(aws ec2 describe-instances --instance-ids i-0322d9d5fc9ccb5e8 \
  --region us-west-1 --query 'Reservations[0].Instances[0].PublicDnsName' --output text)

# Update config
sed -i '' "s/HostName .*/HostName $NEW_DNS/" ~/.ssh/config
```

---

## 🎯 Next Steps

1. ✅ Connect to EC2 via VS Code Remote-SSH
2. ✅ Run the setup script or install packages manually
3. ✅ Upload your notebook files
4. ✅ Start coding with faster compute!
5. ⚠️ Remember to stop the instance when done to save money

---

## 📝 Notes

- **No GPU**: Your AWS account has a 0 vCPU limit for GPU instances. To use GPU (g4dn.xlarge):
  - Go to: https://console.aws.amazon.com/servicequotas
  - Request limit increase for "Running On-Demand G instances"
  - Once approved, launch a new GPU instance

- **S3 Access**: The instance can directly access your S3 bucket without downloading files locally

- **Backups**: Your code and data are on the instance. Back up important changes to S3 or Git!

---

## 🆘 Support

If you need help:
1. Check instance status in AWS Console
2. View instance logs: `aws ec2 get-console-output --instance-id i-0322d9d5fc9ccb5e8 --region us-west-1`
3. SSH directly: `ssh dss-ml-aws`

Happy coding! 🎉


