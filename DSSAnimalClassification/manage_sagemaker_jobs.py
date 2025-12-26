#!/usr/bin/env python3
"""
SageMaker Job Manager
Lists and stops all running SageMaker jobs (training, processing, etc.)
"""

import boto3
import argparse
from datetime import datetime
from botocore.exceptions import ClientError


class SageMakerJobManager:
    """Manage SageMaker jobs - list, stop, and monitor"""
    
    def __init__(self, region='us-west-1'):
        self.region = region
        self.sm_client = boto3.client('sagemaker', region_name=region)
    
    def list_training_jobs(self, status_filter=None, max_results=100):
        """List all training jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_training_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('TrainingJobSummaries', []):
                        jobs.append({
                            'type': 'Training',
                            'name': job['TrainingJobName'],
                            'status': job['TrainingJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': self._get_training_instance_type(job['TrainingJobName'])
                        })
            except ClientError as e:
                print(f"Error listing training jobs with status {status}: {e}")
        
        return jobs
    
    def list_processing_jobs(self, status_filter=None, max_results=100):
        """List all processing jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_processing_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('ProcessingJobSummaries', []):
                        jobs.append({
                            'type': 'Processing',
                            'name': job['ProcessingJobName'],
                            'status': job['ProcessingJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': self._get_processing_instance_type(job['ProcessingJobName'])
                        })
            except ClientError as e:
                print(f"Error listing processing jobs with status {status}: {e}")
        
        return jobs
    
    def list_transform_jobs(self, status_filter=None, max_results=100):
        """List all transform (batch inference) jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_transform_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('TransformJobSummaries', []):
                        jobs.append({
                            'type': 'Transform',
                            'name': job['TransformJobName'],
                            'status': job['TransformJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'N/A'
                        })
            except ClientError as e:
                print(f"Error listing transform jobs with status {status}: {e}")
        
        return jobs
    
    def _get_training_instance_type(self, job_name):
        """Get instance type for a training job"""
        try:
            response = self.sm_client.describe_training_job(TrainingJobName=job_name)
            return response.get('ResourceConfig', {}).get('InstanceType', 'N/A')
        except:
            return 'N/A'
    
    def _get_processing_instance_type(self, job_name):
        """Get instance type for a processing job"""
        try:
            response = self.sm_client.describe_processing_job(ProcessingJobName=job_name)
            return response.get('ProcessingResources', {}).get('ClusterConfig', {}).get('InstanceType', 'N/A')
        except:
            return 'N/A'
    
    def stop_training_job(self, job_name):
        """Stop a training job"""
        try:
            self.sm_client.stop_training_job(TrainingJobName=job_name)
            print(f"✓ Stopped training job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Training job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop training job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping training job {job_name}: {e}")
            return False
    
    def stop_processing_job(self, job_name):
        """Stop a processing job"""
        try:
            self.sm_client.stop_processing_job(ProcessingJobName=job_name)
            print(f"✓ Stopped processing job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Processing job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop processing job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping processing job {job_name}: {e}")
            return False
    
    def stop_transform_job(self, job_name):
        """Stop a transform job"""
        try:
            self.sm_client.stop_transform_job(TransformJobName=job_name)
            print(f"✓ Stopped transform job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Transform job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop transform job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping transform job {job_name}: {e}")
            return False
    
    def list_all_running_jobs(self):
        """List all running jobs (training, processing, transform)"""
        print("\n" + "="*80)
        print("SAGEMAKER RUNNING JOBS")
        print("="*80)
        
        all_jobs = []
        
        # Get all job types
        training_jobs = self.list_training_jobs()
        processing_jobs = self.list_processing_jobs()
        transform_jobs = self.list_transform_jobs()
        
        all_jobs.extend(training_jobs)
        all_jobs.extend(processing_jobs)
        all_jobs.extend(transform_jobs)
        
        if not all_jobs:
            print("\n✓ No running jobs found!")
            return []
        
        # Display jobs
        print(f"\nFound {len(all_jobs)} running job(s):\n")
        print(f"{'Type':<12} {'Status':<12} {'Instance':<20} {'Job Name':<50}")
        print("-" * 80)
        
        for job in all_jobs:
            creation_time = job.get('creation_time', 'N/A')
            if creation_time != 'N/A':
                creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')
            
            print(f"{job['type']:<12} {job['status']:<12} {job['instance_type']:<20} {job['name']:<50}")
            print(f"  Created: {creation_time}")
        
        print("\n" + "="*80)
        return all_jobs
    
    def stop_all_running_jobs(self, confirm=True):
        """Stop all running jobs"""
        jobs = self.list_all_running_jobs()
        
        if not jobs:
            return
        
        if confirm:
            response = input(f"\n⚠️  Stop all {len(jobs)} running job(s)? (yes/no): ")
            if response.lower() not in ['yes', 'y']:
                print("Cancelled.")
                return
        
        print("\nStopping jobs...")
        stopped_count = 0
        
        for job in jobs:
            job_type = job['type']
            job_name = job['name']
            
            if job_type == 'Training':
                if self.stop_training_job(job_name):
                    stopped_count += 1
            elif job_type == 'Processing':
                if self.stop_processing_job(job_name):
                    stopped_count += 1
            elif job_type == 'Transform':
                if self.stop_transform_job(job_name):
                    stopped_count += 1
        
        print(f"\n✓ Stopped {stopped_count}/{len(jobs)} job(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Manage SageMaker jobs - list and stop running jobs'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-west-1',
        help='AWS region (default: us-west-1)'
    )
    parser.add_argument(
        '--action',
        type=str,
        choices=['list', 'stop-all'],
        default='list',
        help='Action to perform: list (default) or stop-all'
    )
    parser.add_argument(
        '--no-confirm',
        action='store_true',
        help='Skip confirmation prompt when stopping jobs'
    )
    
    args = parser.parse_args()
    
    manager = SageMakerJobManager(region=args.region)
    
    if args.action == 'list':
        manager.list_all_running_jobs()
    elif args.action == 'stop-all':
        manager.stop_all_running_jobs(confirm=not args.no_confirm)


if __name__ == '__main__':
    main()

