# Quick Start: Using Your EC2 Instance with VS Code

## 🚀 Connect to EC2 in VS Code (3 Steps)

### 1. Install Extension
- Open VS Code
- Press `Cmd+Shift+X`
- Search "Remote - SSH"
- Click Install

### 2. Connect
- Press `Cmd+Shift+P`
- Type: "Remote-SSH: Connect to Host"
- Select: **`dss-ml-aws`**
- New window opens → You're connected! 🎉

### 3. Open Project
- Click "Open Folder"
- Navigate to: `/home/ec2-user/DSS-Image-Classification`
- Start coding!

---

## 📦 First Time Setup

Run once to set up your environment:

```bash
# From your local terminal
cd ~/Projects/DSS-Image-Classification

# Upload and run setup
./manage_ec2.sh setup

# Upload your files
./manage_ec2.sh upload
```

Or do it manually:
```bash
ssh dss-ml-aws
python3 -m venv ~/ml-env
source ~/ml-env/bin/activate
pip install torch torchvision boto3 sagemaker pandas numpy pillow jupyter
mkdir -p ~/DSS-Image-Classification
```

---

## 🛠️ Management Commands

Use the helper script for common tasks:

```bash
./manage_ec2.sh status      # Check if running
./manage_ec2.sh start       # Start instance
./manage_ec2.sh stop        # Stop instance (saves money!)
./manage_ec2.sh connect     # SSH into instance
./manage_ec2.sh upload      # Upload project files
./manage_ec2.sh setup       # Setup Python environment
./manage_ec2.sh cost        # Show cost estimates
```

---

## 💰 Important: Cost Savings

**The instance costs ~$0.166/hour (~$4/day if left running)**

### Always stop when done:
```bash
./manage_ec2.sh stop
```

### When you want to use it again:
```bash
./manage_ec2.sh start
./manage_ec2.sh update-dns  # Update DNS (changes after restart)
```

---

## 📝 Running Your Notebooks

### In VS Code (Best):
1. Connect via Remote-SSH
2. Open `.ipynb` file
3. Select interpreter: `/home/ec2-user/ml-env/bin/python`
4. Run cells!

### Via Jupyter:
```bash
ssh dss-ml-aws
source ~/ml-env/bin/activate
cd ~/DSS-Image-Classification
jupyter notebook --no-browser --port=8888

# In another terminal:
ssh -L 8888:localhost:8888 dss-ml-aws
# Open: http://localhost:8888
```

---

## 🗂️ Working with S3

Your data is in S3 - access it directly:

```python
import boto3
from PIL import Image
from io import BytesIO

s3 = boto3.client('s3', region_name='us-west-1')
bucket = 'animal-classification-dss-works'

# Read image
response = s3.get_object(Bucket=bucket, Key='data/train_features/image.jpg')
img = Image.open(BytesIO(response['Body'].read()))
```

---

## ⚠️ Troubleshooting

### Can't connect after restart?
```bash
./manage_ec2.sh update-dns
```

### Permission denied?
```bash
chmod 400 ~/.ssh/dss-vscode-ml-key.pem
```

### Check what's running:
```bash
./manage_ec2.sh status
```

---

## 📚 Files Created

- `EC2_SETUP_GUIDE.md` - Complete documentation
- `manage_ec2.sh` - Management helper script
- `setup_ec2.sh` - Environment setup script
- `QUICKSTART.md` - This file
- `~/.ssh/config` - SSH configuration
- `~/.ssh/dss-vscode-ml-key.pem` - SSH key

---

## Instance Info

- **Instance ID**: `i-0322d9d5fc9ccb5e8`
- **Type**: t3.xlarge (4 vCPUs, 16GB RAM)
- **DNS**: `ec2-54-193-175-190.us-west-1.compute.amazonaws.com`
- **Region**: us-west-1
- **SSH Alias**: `dss-ml-aws`

---

## Next Steps

1. ✅ Connect via VS Code Remote-SSH
2. ✅ Run `./manage_ec2.sh setup` (first time only)
3. ✅ Upload files with `./manage_ec2.sh upload`
4. ✅ Start coding faster!
5. ⚠️ **STOP instance when done**: `./manage_ec2.sh stop`

Happy coding! 🎉


