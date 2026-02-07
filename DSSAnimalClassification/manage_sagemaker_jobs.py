#!/usr/bin/env python3
"""
SageMaker Job Manager
Lists and stops all running SageMaker jobs including:
- Training jobs (PyTorch, TensorFlow, etc.)
- Processing jobs (PyTorch preprocessing, data processing, etc.)
- Transform jobs (batch inference jobs)
- Hyperparameter Tuning jobs
- AutoML jobs
- Model Compilation jobs
- Edge Packaging jobs
- Labeling jobs (Ground Truth)
- Data Quality Monitoring jobs
- Model Quality Monitoring jobs
- Model Bias jobs
- Model Explainability jobs
"""

import boto3
import argparse
from datetime import datetime
from botocore.exceptions import ClientError


class SageMakerJobManager:
    """Manage SageMaker jobs - list, stop, and monitor
    
    Supports ALL SageMaker job types:
    - Training jobs (PyTorch Estimator, etc.)
    - Processing jobs (PyTorchProcessor for preprocessing, etc.)
    - Transform jobs (batch inference)
    - Hyperparameter Tuning jobs
    - AutoML jobs
    - Model Compilation jobs
    - Edge Packaging jobs
    - Labeling jobs (Ground Truth)
    - Data Quality Monitoring jobs
    - Model Quality Monitoring jobs
    - Model Bias jobs
    - Model Explainability jobs
    """
    
    def __init__(self, region='us-east-1'):
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
        """List all processing jobs (includes PyTorch preprocessing jobs via PyTorchProcessor)"""
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
                        job_name = job['ProcessingJobName']
                        framework = self._get_processing_framework(job_name)
                        jobs.append({
                            'type': 'Processing',
                            'name': job_name,
                            'status': job['ProcessingJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': self._get_processing_instance_type(job_name),
                            'framework': framework
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
                        job_name = job['TransformJobName']
                        framework = self._get_transform_framework(job_name)
                        jobs.append({
                            'type': 'Transform',
                            'name': job_name,
                            'status': job['TransformJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': self._get_transform_instance_type(job_name),
                            'framework': framework
                        })
            except ClientError as e:
                print(f"Error listing transform jobs with status {status}: {e}")
        
        return jobs
    
    def list_hyperparameter_tuning_jobs(self, status_filter=None, max_results=100):
        """List all hyperparameter tuning jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_hyper_parameter_tuning_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('HyperParameterTuningJobSummaries', []):
                        jobs.append({
                            'type': 'HyperparameterTuning',
                            'name': job['HyperParameterTuningJobName'],
                            'status': job['HyperParameterTuningJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'Multiple',
                            'framework': 'N/A'
                        })
            except ClientError as e:
                print(f"Error listing hyperparameter tuning jobs with status {status}: {e}")
        
        return jobs
    
    def list_automl_jobs(self, status_filter=None, max_results=100):
        """List all AutoML jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_auto_ml_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('AutoMLJobSummaries', []):
                        jobs.append({
                            'type': 'AutoML',
                            'name': job['AutoMLJobName'],
                            'status': job['AutoMLJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'Auto',
                            'framework': 'AutoML'
                        })
            except ClientError as e:
                print(f"Error listing AutoML jobs with status {status}: {e}")
        
        return jobs
    
    def list_compilation_jobs(self, status_filter=None, max_results=100):
        """List all model compilation jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_compilation_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('CompilationJobSummaries', []):
                        jobs.append({
                            'type': 'Compilation',
                            'name': job['CompilationJobName'],
                            'status': job['CompilationJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'N/A',
                            'framework': 'N/A'
                        })
            except ClientError as e:
                print(f"Error listing compilation jobs with status {status}: {e}")
        
        return jobs
    
    def list_edge_packaging_jobs(self, status_filter=None, max_results=100):
        """List all edge packaging jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_edge_packaging_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('EdgePackagingJobSummaries', []):
                        jobs.append({
                            'type': 'EdgePackaging',
                            'name': job['EdgePackagingJobName'],
                            'status': job['EdgePackagingJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'N/A',
                            'framework': 'N/A'
                        })
            except ClientError as e:
                print(f"Error listing edge packaging jobs with status {status}: {e}")
        
        return jobs
    
    def list_labeling_jobs(self, status_filter=None, max_results=100):
        """List all labeling jobs (Ground Truth)"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_labeling_jobs')
                page_iterator = paginator.paginate(
                    StatusEquals=status,
                    MaxResults=max_results
                )
                
                for page in page_iterator:
                    for job in page.get('LabelingJobSummaryList', []):
                        jobs.append({
                            'type': 'Labeling',
                            'name': job['LabelingJobName'],
                            'status': job['LabelingJobStatus'],
                            'creation_time': job.get('CreationTime', 'N/A'),
                            'instance_type': 'N/A',
                            'framework': 'Ground Truth'
                        })
            except ClientError as e:
                print(f"Error listing labeling jobs with status {status}: {e}")
        
        return jobs
    
    def list_data_quality_jobs(self, status_filter=None, max_results=100):
        """List all data quality monitoring jobs"""
        jobs = []
        statuses = status_filter if status_filter else ['InProgress', 'Stopping']
        
        for status in statuses:
            try:
                paginator = self.sm_client.get_paginator('list_data_quality_job_definitions')
                # Note: This lists job definitions, not running jobs
                # We need to check for running jobs differently
                page_iterator = paginator.paginate(MaxResults=max_results)
                
                for page in page_iterator:
                    for job_def in page.get('JobDefinitionSummaries', []):
                        job_name = job_def['JobDefinitionName']
                        # Try to find running jobs with this definition
                        try:
                            # Check if there are running jobs for this definition
                            # This is a simplified approach - actual implementation may vary
                            jobs.append({
                                'type': 'DataQuality',
                                'name': job_name,
                                'status': 'Unknown',
                                'creation_time': job_def.get('CreationTime', 'N/A'),
                                'instance_type': 'N/A',
                                'framework': 'SageMaker'
                            })
                        except:
                            pass
            except ClientError as e:
                # Data quality jobs might not be available in all regions
                pass
        
        return jobs
    
    def list_model_quality_jobs(self, status_filter=None, max_results=100):
        """List all model quality monitoring jobs"""
        jobs = []
        try:
            paginator = self.sm_client.get_paginator('list_model_quality_job_definitions')
            page_iterator = paginator.paginate(MaxResults=max_results)
            
            for page in page_iterator:
                for job_def in page.get('JobDefinitionSummaries', []):
                    jobs.append({
                        'type': 'ModelQuality',
                        'name': job_def['JobDefinitionName'],
                        'status': 'Unknown',
                        'creation_time': job_def.get('CreationTime', 'N/A'),
                        'instance_type': 'N/A',
                        'framework': 'SageMaker'
                    })
        except ClientError as e:
            # Model quality jobs might not be available in all regions
            pass
        
        return jobs
    
    def list_model_bias_jobs(self, status_filter=None, max_results=100):
        """List all model bias monitoring jobs"""
        jobs = []
        try:
            paginator = self.sm_client.get_paginator('list_model_bias_job_definitions')
            page_iterator = paginator.paginate(MaxResults=max_results)
            
            for page in page_iterator:
                for job_def in page.get('JobDefinitionSummaries', []):
                    jobs.append({
                        'type': 'ModelBias',
                        'name': job_def['JobDefinitionName'],
                        'status': 'Unknown',
                        'creation_time': job_def.get('CreationTime', 'N/A'),
                        'instance_type': 'N/A',
                        'framework': 'SageMaker'
                    })
        except ClientError as e:
            # Model bias jobs might not be available in all regions
            pass
        
        return jobs
    
    def list_model_explainability_jobs(self, status_filter=None, max_results=100):
        """List all model explainability monitoring jobs"""
        jobs = []
        try:
            paginator = self.sm_client.get_paginator('list_model_explainability_job_definitions')
            page_iterator = paginator.paginate(MaxResults=max_results)
            
            for page in page_iterator:
                for job_def in page.get('JobDefinitionSummaries', []):
                    jobs.append({
                        'type': 'ModelExplainability',
                        'name': job_def['JobDefinitionName'],
                        'status': 'Unknown',
                        'creation_time': job_def.get('CreationTime', 'N/A'),
                        'instance_type': 'N/A',
                        'framework': 'SageMaker'
                    })
        except ClientError as e:
            # Model explainability jobs might not be available in all regions
            pass
        
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
    
    def _get_processing_framework(self, job_name):
        """Detect framework for a processing job (PyTorch, TensorFlow, etc.)"""
        try:
            response = self.sm_client.describe_processing_job(ProcessingJobName=job_name)
            app_spec = response.get('AppSpecification', {})
            image_uri = app_spec.get('ImageUri', '')
            
            # Check image URI for framework indicators
            if 'pytorch' in image_uri.lower():
                return 'PyTorch'
            elif 'tensorflow' in image_uri.lower() or 'tf' in image_uri.lower():
                return 'TensorFlow'
            elif 'sklearn' in image_uri.lower() or 'scikit' in image_uri.lower():
                return 'Scikit-learn'
            elif 'spark' in image_uri.lower():
                return 'Spark'
            else:
                return 'Unknown'
        except:
            return 'Unknown'
    
    def _get_transform_instance_type(self, job_name):
        """Get instance type for a transform job"""
        try:
            response = self.sm_client.describe_transform_job(TransformJobName=job_name)
            return response.get('TransformResources', {}).get('InstanceType', 'N/A')
        except:
            return 'N/A'
    
    def _get_transform_framework(self, job_name):
        """Detect framework for a transform job"""
        try:
            response = self.sm_client.describe_transform_job(TransformJobName=job_name)
            model_name = response.get('ModelName', '')
            
            # Try to get model details to detect framework
            try:
                model_response = self.sm_client.describe_model(ModelName=model_name)
                image_uri = model_response.get('PrimaryContainer', {}).get('Image', '')
                
                if 'pytorch' in image_uri.lower():
                    return 'PyTorch'
                elif 'tensorflow' in image_uri.lower() or 'tf' in image_uri.lower():
                    return 'TensorFlow'
                elif 'xgboost' in image_uri.lower():
                    return 'XGBoost'
                else:
                    return 'Unknown'
            except:
                return 'Unknown'
        except:
            return 'Unknown'
    
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
        """Stop a processing job (includes PyTorch preprocessing jobs)"""
        try:
            self.sm_client.stop_processing_job(ProcessingJobName=job_name)
            framework = self._get_processing_framework(job_name)
            if framework != 'Unknown':
                print(f"✓ Stopped {framework} processing job: {job_name}")
            else:
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
        """Stop a transform (batch inference) job"""
        try:
            self.sm_client.stop_transform_job(TransformJobName=job_name)
            framework = self._get_transform_framework(job_name)
            if framework != 'Unknown':
                print(f"✓ Stopped {framework} transform job: {job_name}")
            else:
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
    
    def stop_hyperparameter_tuning_job(self, job_name):
        """Stop a hyperparameter tuning job"""
        try:
            self.sm_client.stop_hyper_parameter_tuning_job(HyperParameterTuningJobName=job_name)
            print(f"✓ Stopped hyperparameter tuning job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Hyperparameter tuning job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop hyperparameter tuning job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping hyperparameter tuning job {job_name}: {e}")
            return False
    
    def stop_automl_job(self, job_name):
        """Stop an AutoML job"""
        try:
            self.sm_client.stop_auto_ml_job(AutoMLJobName=job_name)
            print(f"✓ Stopped AutoML job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ AutoML job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop AutoML job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping AutoML job {job_name}: {e}")
            return False
    
    def stop_compilation_job(self, job_name):
        """Stop a model compilation job"""
        try:
            self.sm_client.stop_compilation_job(CompilationJobName=job_name)
            print(f"✓ Stopped compilation job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Compilation job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop compilation job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping compilation job {job_name}: {e}")
            return False
    
    def stop_edge_packaging_job(self, job_name):
        """Stop an edge packaging job"""
        try:
            self.sm_client.stop_edge_packaging_job(EdgePackagingJobName=job_name)
            print(f"✓ Stopped edge packaging job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Edge packaging job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop edge packaging job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping edge packaging job {job_name}: {e}")
            return False
    
    def stop_labeling_job(self, job_name):
        """Stop a labeling job (Ground Truth)"""
        try:
            self.sm_client.stop_labeling_job(LabelingJobName=job_name)
            print(f"✓ Stopped labeling job: {job_name}")
            return True
        except ClientError as e:
            error_code = e.response.get('Error', {}).get('Code', '')
            if error_code == 'ResourceNotFound':
                print(f"✗ Labeling job not found: {job_name}")
            elif error_code == 'ResourceInUse':
                print(f"✗ Cannot stop labeling job (already stopping/completed): {job_name}")
            else:
                print(f"✗ Error stopping labeling job {job_name}: {e}")
            return False
    
    def list_all_running_jobs(self):
        """List all running jobs - ALL SageMaker job types"""
        print("\n" + "="*80)
        print("SAGEMAKER RUNNING JOBS")
        print("(All job types: Training, Processing, Transform, HyperparameterTuning,")
        print(" AutoML, Compilation, EdgePackaging, Labeling, Monitoring)")
        print("="*80)
        
        all_jobs = []
        
        # Get all job types
        print("Scanning job types...")
        training_jobs = self.list_training_jobs()
        processing_jobs = self.list_processing_jobs()  # Includes PyTorchProcessor jobs
        transform_jobs = self.list_transform_jobs()
        hyperparameter_jobs = self.list_hyperparameter_tuning_jobs()
        automl_jobs = self.list_automl_jobs()
        compilation_jobs = self.list_compilation_jobs()
        edge_packaging_jobs = self.list_edge_packaging_jobs()
        labeling_jobs = self.list_labeling_jobs()
        data_quality_jobs = self.list_data_quality_jobs()
        model_quality_jobs = self.list_model_quality_jobs()
        model_bias_jobs = self.list_model_bias_jobs()
        model_explainability_jobs = self.list_model_explainability_jobs()
        
        all_jobs.extend(training_jobs)
        all_jobs.extend(processing_jobs)
        all_jobs.extend(transform_jobs)
        all_jobs.extend(hyperparameter_jobs)
        all_jobs.extend(automl_jobs)
        all_jobs.extend(compilation_jobs)
        all_jobs.extend(edge_packaging_jobs)
        all_jobs.extend(labeling_jobs)
        all_jobs.extend(data_quality_jobs)
        all_jobs.extend(model_quality_jobs)
        all_jobs.extend(model_bias_jobs)
        all_jobs.extend(model_explainability_jobs)
        
        if not all_jobs:
            print("\n✓ No running jobs found!")
            return []
        
        # Display jobs
        print(f"\nFound {len(all_jobs)} running job(s):\n")
        print(f"{'Type':<12} {'Framework':<12} {'Status':<12} {'Instance':<20} {'Job Name':<50}")
        print("-" * 80)
        
        for job in all_jobs:
            creation_time = job.get('creation_time', 'N/A')
            if creation_time != 'N/A':
                creation_time = creation_time.strftime('%Y-%m-%d %H:%M:%S')
            
            framework = job.get('framework', 'N/A')
            job_type = job['type']
            
            print(f"{job_type:<12} {framework:<12} {job['status']:<12} {job['instance_type']:<20} {job['name']:<50}")
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
            elif job_type == 'HyperparameterTuning':
                if self.stop_hyperparameter_tuning_job(job_name):
                    stopped_count += 1
            elif job_type == 'AutoML':
                if self.stop_automl_job(job_name):
                    stopped_count += 1
            elif job_type == 'Compilation':
                if self.stop_compilation_job(job_name):
                    stopped_count += 1
            elif job_type == 'EdgePackaging':
                if self.stop_edge_packaging_job(job_name):
                    stopped_count += 1
            elif job_type == 'Labeling':
                if self.stop_labeling_job(job_name):
                    stopped_count += 1
            elif job_type in ['DataQuality', 'ModelQuality', 'ModelBias', 'ModelExplainability']:
                print(f"⚠️  {job_type} jobs cannot be stopped via API (monitoring job definitions)")
            else:
                print(f"⚠️  Unknown job type: {job_type} - {job_name}")
        
        print(f"\n✓ Stopped {stopped_count}/{len(jobs)} job(s)")


def main():
    parser = argparse.ArgumentParser(
        description='Manage ALL SageMaker jobs - list and stop running jobs\n'
                    'Supports: Training, Processing, Transform, HyperparameterTuning,\n'
                    '          AutoML, Compilation, EdgePackaging, Labeling, Monitoring jobs'
    )
    parser.add_argument(
        '--region',
        type=str,
        default='us-east-1',
        help='AWS region (default: us-east-1)'
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

