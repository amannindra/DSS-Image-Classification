#!/bin/bash
# EC2 Instance Management Helper Script

INSTANCE_ID="i-0322d9d5fc9ccb5e8"
REGION="us-west-1"
SSH_HOST="dss-ml-aws"

# Colors
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

function show_status() {
    echo -e "${YELLOW}Checking instance status...${NC}"
    aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].[InstanceId,State.Name,InstanceType,PublicDnsName]' \
        --output table
}

function start_instance() {
    echo -e "${YELLOW}Starting instance...${NC}"
    aws ec2 start-instances --instance-ids $INSTANCE_ID --region $REGION
    echo "Waiting for instance to start..."
    aws ec2 wait instance-running --instance-ids $INSTANCE_ID --region $REGION
    echo -e "${GREEN}✓ Instance started!${NC}"
    update_ssh_config
}

function stop_instance() {
    echo -e "${YELLOW}Stopping instance...${NC}"
    aws ec2 stop-instances --instance-ids $INSTANCE_ID --region $REGION
    echo -e "${GREEN}✓ Stop command sent${NC}"
}

function update_ssh_config() {
    echo -e "${YELLOW}Updating SSH config with new DNS...${NC}"
    NEW_DNS=$(aws ec2 describe-instances \
        --instance-ids $INSTANCE_ID \
        --region $REGION \
        --query 'Reservations[0].Instances[0].PublicDnsName' \
        --output text)
    
    if [ ! -z "$NEW_DNS" ]; then
        sed -i '' "s/HostName .*/HostName $NEW_DNS/" ~/.ssh/config
        echo -e "${GREEN}✓ SSH config updated with: $NEW_DNS${NC}"
    else
        echo -e "${RED}Could not get DNS (instance might be stopped)${NC}"
    fi
}

function connect_ssh() {
    echo -e "${YELLOW}Connecting via SSH...${NC}"
    ssh $SSH_HOST
}

function upload_project() {
    echo -e "${YELLOW}Uploading project files...${NC}"
    scp -r ~/Projects/DSS-Image-Classification/*.ipynb \
           ~/Projects/DSS-Image-Classification/train.py \
           ~/Projects/DSS-Image-Classification/requirements.txt \
           $SSH_HOST:~/DSS-Image-Classification/ 2>/dev/null || \
        ssh $SSH_HOST "mkdir -p ~/DSS-Image-Classification" && \
        scp -r ~/Projects/DSS-Image-Classification/*.ipynb \
           ~/Projects/DSS-Image-Classification/train.py \
           ~/Projects/DSS-Image-Classification/requirements.txt \
           $SSH_HOST:~/DSS-Image-Classification/
    echo -e "${GREEN}✓ Files uploaded${NC}"
}

function setup_environment() {
    echo -e "${YELLOW}Setting up Python environment on EC2...${NC}"
    ssh $SSH_HOST << 'ENDSSH'
        python3 -m venv ~/ml-env
        source ~/ml-env/bin/activate
        pip install --upgrade pip
        pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
        pip install jupyter jupyterlab notebook
        pip install boto3 sagemaker pandas numpy pillow matplotlib seaborn scikit-learn
        mkdir -p ~/DSS-Image-Classification
        echo "✓ Setup complete!"
ENDSSH
    echo -e "${GREEN}✓ Environment setup complete${NC}"
}

function show_cost() {
    echo -e "${YELLOW}Estimated costs:${NC}"
    echo "  Instance (t3.xlarge): ~\$0.166/hour"
    echo "  Storage (100GB GP3): ~\$8/month"
    echo "  Running 24h: ~\$4.00/day"
    echo "  Running 8h/day: ~\$1.33/day"
}

# Main menu
case "$1" in
    status)
        show_status
        ;;
    start)
        start_instance
        ;;
    stop)
        stop_instance
        ;;
    connect)
        connect_ssh
        ;;
    upload)
        upload_project
        ;;
    setup)
        setup_environment
        ;;
    update-dns)
        update_ssh_config
        ;;
    cost)
        show_cost
        ;;
    *)
        echo "EC2 Instance Manager"
        echo ""
        echo "Usage: $0 {status|start|stop|connect|upload|setup|update-dns|cost}"
        echo ""
        echo "Commands:"
        echo "  status      - Show instance status and details"
        echo "  start       - Start the instance"
        echo "  stop        - Stop the instance (saves money)"
        echo "  connect     - SSH into the instance"
        echo "  upload      - Upload project files to instance"
        echo "  setup       - Setup Python environment on instance"
        echo "  update-dns  - Update SSH config with new DNS"
        echo "  cost        - Show cost estimates"
        exit 1
esac


