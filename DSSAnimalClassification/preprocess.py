    #!/usr/bin/env python3
    """
    Data Preprocessing Script for SageMaker Processing
    Processes images from S3, extracts features, creates dataframe
    Saves preprocessed data for training
    """

    import argparse
    import os
    from typing import Any
    import boto3
    import sys
    import traceback
    import sys
    import pandas as pd
    import numpy as np
    from PIL import Image
    from io import BytesIO
    from botocore.exceptions import ClientError

    import pickle
    import gc  # For garbage collection to free memory
    from tqdm import tqdm  # For progress bars
    import os, psutil

    bucket_name = "animal-classification-dss-works"
    train_folder = "data/train_features/"
    test_folder = "data/test_features/"
    REGION = "us-west-1"

    # Initialize S3 client
    s3_client = boto3.client("s3", region_name=REGION)

    # Parse arguments (for SageMaker Processing)
    parser = argparse.ArgumentParser()
    # parser.add_argument("--input-dir", type=str, default="/opt/ml/processing/input")
    # parser.add_argument("--output-dir", type=str, default="/opt/ml/processing/output")
    args, _ = parser.parse_known_args()

    # INPUT_DIR = args.input_dir
    # OUTPUT_DIR = args.output_dir


    # Create output directory if it doesn't exist
    # os.makedirs(OUTPUT_DIR, exist_ok=True)

    def get_ram_usage():
        # Get the process info for the current Python script
        process = psutil.Process(os.getpid())
        # Return the Resident Set Size (RSS) in megabytes
        return process.memory_info().rss / (1024 * 1024)


    def get_image_from_s3(bucket, key):
        """Get image from S3 without downloading to disk"""
        try:
            response = s3_client.get_object(Bucket=bucket, Key=key)
            image = Image.open(BytesIO(response["Body"].read()))
            return image
        except Exception as e:
            print(f"Error loading")
            return None


    def get_all_image_keys(bucket, prefix):
        """Get all image keys from S3 folder"""
        print(f"Getting image list from s3://{bucket}/{prefix}")
        paginator = s3_client.get_paginator("list_objects_v2")
        pages = paginator.paginate(Bucket=bucket, Prefix=prefix)

        image_keys = []
        for page in pages:
            for obj in page.get("Contents", []):
                key = obj["Key"]
                if key.lower().endswith((".jpg", ".jpeg", ".png")):
                    image_keys.append(key)

        print(f"Found {len(image_keys)} images")
        return image_keys


    def process_images(bucket, image_keys, is_test):
        """
        Process images and extract metadata

        Args:
            save_numpy: If True, also saves full numpy arrays (much larger file)
        """
        # data = []

        print(f"Processing {len(image_keys)} images...")
        amount = 0
        for key in (image_keys):
            amount += 1
            if amount % 100 == 0:
                print(f"Processing {amount} of {len(image_keys)} images, RAM usage: {get_ram_usage():.2f} MB")
            # print(f"Processing {amount} of {len(image_keys)} images...")
            img = get_image_from_s3(bucket, key)
            
            if img is None:
                print(f"Image {key} not found")
                continue
            
            # if img.mode == "L":
            if img.mode == "L":
                print(f"Image {key} is grayscale, converting to RGB")
                img = img.convert("RGB")
            # s3_client.delete_object(Bucket=bucket, Key=key)
            # print(f"Image {key} converted to RGB")
            image_buffer = BytesIO()
            # save_kwargs["quality"] = 90
            # save_kwargs["optimize"] = True
            img.save(image_buffer, format="JPEG") 
            image_buffer.seek(0)
            
            filename = key.split("/")[-1]
            
            if is_test:
                new_key = f"processed/test_features/{filename}"
            else:
                new_key = f"processed/train_features/{filename}"
            s3_client.put_object(Bucket=bucket, Key=new_key, Body=image_buffer)
            
                
        
    def folder_exists_and_not_empty(bucket:str, path:str) -> bool:
        '''
        Folder should exists. 
        Folder should not be empty.
        '''
        s3 = boto3.client('s3')
        if not path.endswith('/'):
            path = path+'/' 
        resp = s3.list_objects(Bucket=bucket, Prefix=path, Delimiter='/',MaxKeys=1)
        return 'Contents' in resp


    def s3_file_exists(bucket_name, object_name):
        """
        Checks if a file exists in an S3 bucket.

        :param bucket_name: Name of the S3 bucket.
        :param object_name: S3 object key (file path).
        :return: True if the object exists, False otherwise.
        """
        s3_client = boto3.client('s3')
        try:
            s3_client.head_object(Bucket=bucket_name, Key=object_name)
            return True # Object exists
        except ClientError as e:
            if e.response['Error']['Code'] == '404':
                # The object does not exist
                return False
            else:
                # Another error occurred (e.g., permission issues, network problems)
                print(f"An error occurred: {e}")
                raise


    def main():

        print("Input Directory: {INPUT_DIR}")
        print("Output Directory: {OUTPUT_DIR}")


        print(f"Initial RAM usage: {get_ram_usage():.2f} MB")
        
        # print("=" * 60)
        # print("IMAGE PREPROCESSING FOR SAGEMAKER TRAINING")
        # print("=" * 60)


        # print("\n1. Processing TRAINING images...")
        
        
        train_exists = folder_exists_and_not_empty(bucket_name, train_folder)
        test_exists = folder_exists_and_not_empty(bucket_name, test_folder)
        
        if train_exists and test_exists:
            print("Train and test folders exist and are not empty")
        else:
            print("Train and test folders do not exist or are empty")
            train_keys = get_all_image_keys(bucket_name, train_folder) # "data/train_features/"
            test_keys = get_all_image_keys(bucket_name, test_folder) # "data/test_features/"
            
            process_images(bucket_name, train_keys, is_test=False)
            process_images(bucket_name, test_keys, is_test=True)
            


        
        # s3_file_exists(bucket_name, "processed/train_features/ZJ000000.jpg")
        
        
        
        


        # process_images(bucket_name, train_keys, is_test=False)
        # process_images(bucket_name, test_keys, is_test=True)
        
        
        
        
        


    if __name__ == "__main__":
        main()


            # if img is None:
            #     print(f"Image {key} not found")
            #     continue

            # # Extract filename
            # filename = key.split("/")[-1]

            # # Get dimensions
            # width, height = img.size

            # # Basic stats
            # img_array = np.array(img)

            # data_dict = {
            #     "filename": filename,
            #     "width": width,
            #     "height": height,
            #     "channels": img_array.shape[2] if len(img_array.shape) == 3 else 1,
            #     "s3_key": key,
            # }
            
            # # Add image numpy array if requested
            # if save_numpy:
            #     data_dict["image"] = img_array
            

        # return pd.DataFrame(data)
        
            
        # print(f"After getting image keys RAM usage: {get_ram_usage():.2f} MB")
        
        # BATCH_SIZE = 500  # Process 500 images at a time
        
        # # Create temp directory for batch files
        # temp_dir = os.path.join(OUTPUT_DIR, 'temp_batches')
        # print(f"Creating temp directory: {temp_dir}")
        # os.makedirs(temp_dir, exist_ok=True)
        
        # print(f"Processing {len(train_keys)} images in batches of {BATCH_SIZE}...")
        # print(f"Saving each batch to disk to conserve memory...")
        
        # batch_files = []
        # for i in range(0, len(train_keys), BATCH_SIZE):
        #     batch_keys = train_keys[i:i+BATCH_SIZE]
        #     batch_num = i//BATCH_SIZE + 1
        #     total_batches = (len(train_keys) - 1) // BATCH_SIZE + 1
            
        #     print(f"\n   Batch {batch_num}/{total_batches} ({len(batch_keys)} images)...")
        #     batch_df = process_images(bucket_name, batch_keys)
            
            
            
            
        #     # Save batch to disk immediately
        #     batch_file = os.path.join(temp_dir, f'train_batch_{batch_num}.pkl')
        #     with open(batch_file, 'wb') as f:
        #         pickle.dump(batch_df, f)
        #     batch_files.append(batch_file)
            
        #     print(f"   ✓ Batch {batch_num} saved to disk")
            
            
        #     # Free memory
        #     del batch_df
        #     gc.collect()
        #     # break
        # print(f"After processing train images RAM usage: {get_ram_usage():.2f} MB")
        # # Now load and combine all batches (one at a time)
        # print("\n   Combining all batches from disk...")
        # train_dfs = []
        # for batch_file in batch_files:
        #     with open(batch_file, 'rb') as f:
        #         batch_df = pickle.load(f)
        #         train_dfs.append(batch_df)
        
        # train_df = pd.concat(train_dfs, ignore_index=True)
        # print(train_df.head())
        # print(f"   ✓ Combined {len(train_df)} training images")
        # print(f"After combining train images RAM usage: {get_ram_usage():.2f} MB")
        
        # # Clean up temp files
        # del train_dfs # Doesn't free memory
        # gc.collect()
        # print(f"After cleaning up train images RAM usage: {get_ram_usage():.2f} MB")
        # for batch_file in batch_files:
        #     os.remove(batch_file)
        # os.rmdir(temp_dir)

        # # Load labels if available
        # print("\n2. Loading training labels...")
        # print(f"After loading training labels RAM usage: {get_ram_usage():.2f} MB")
        # try:
        #     labels_obj = s3_client.get_object(Bucket=bucket_name, Key="data/train_labels.csv")
        #     print("labels_obj loaded successfully")
        #     labels_df = pd.read_csv(labels_obj["Body"])
        #     print(f"Labels loaded: {len(labels_df)} rows")
            
        #     # DEBUG: Check columns before merge
        #     print("\n=== DEBUG INFO ===")
            
        #     print(f"train_df columns: {list(train_df.columns)}")
        #     print(f"train_df shape: {train_df.shape}")
        #     print(f"train_df sample:\n{train_df.head()}") #

        #     print(f"train_df columns: {list[Any](labels_df.columns)}")
        #     print(f"labels_df shape: {labels_df.shape}")
        #     print(f"labels_df sample:\n{labels_df.head()}")
            
        
        #     labels_df["filename"] = labels_df["id"] + ".jpg"
        #     # print(f"labels_df id sample:\n{labels_df['id'].head()}")


        #     # Merge with image data
        #     print("Attempting merge...")
        #     train_df = train_df.merge(labels_df, on="filename", how="left")
        #     print(f"✓ Merged dataset: {len(train_df)} rows")
            
        #     # Check for NaN labels after merge
        #     label_cols = [col for col in train_df.columns if col in ['antelope_duiker', 'bird', 'blank', 'civet_genet', 'hog', 'leopard', 'monkey_prosimian', 'rodent']]
        #     if label_cols:
        #         null_count = train_df[label_cols].isnull().any(axis=1).sum()
        #         print(f"Rows with missing labels: {null_count}")
            
        # except Exception as e:
        #     print(f"❌ ERROR during label loading/merging: {e}")
        #     import traceback
        #     traceback.print_exc()
        #     sys.exit(0)
        # print(f"After merging training labels RAM usage: {get_ram_usage():.2f} MB")
        # # Process test images IN BATCHES (save to disk immediately)
        # print("\n3. Processing TEST images...")
        # test_keys = get_all_image_keys(bucket_name, test_folderEFIX)
        # print(f"After getting test image keys RAM usage: {get_ram_usage():.2f} MB")
        
        # # Create temp directory for test batch files
        # temp_dir_test = os.path.join(OUTPUT_DIR, 'temp_test_batches')
        # os.makedirs(temp_dir_test, exist_ok=True)
        
        # print(f"Processing {len(test_keys)} images in batches of {BATCH_SIZE}...")
        # print(f"Saving each batch to disk to conserve memory...")
        
        # test_batch_files = []
        # for i in range(0, len(test_keys), BATCH_SIZE):
        #     batch_keys = test_keys[i:i+BATCH_SIZE]
        #     batch_num = i//BATCH_SIZE + 1
        #     total_batches = (len(test_keys) - 1) // BATCH_SIZE + 1
            
        #     print(f"\n   Batch {batch_num}/{total_batches} ({len(batch_keys)} images)...")
        #     batch_df = process_images(bucket_name, batch_keys)
            
        #     # Save batch to disk immediately
        #     batch_file = os.path.join(temp_dir_test, f'test_batch_{batch_num}.pkl')
        #     with open(batch_file, 'wb') as f:
        #         pickle.dump(batch_df, f)
        #     test_batch_files.append(batch_file)
            
        #     print(f"   ✓ Batch {batch_num} saved to disk")
        
        #     # Free memory
        #     del batch_df
        #     gc.collect()
        #     # break
        
        # print(f"After processing test images RAM usage: {get_ram_usage():.2f} MB")
        # # Now load and combine all batches (one at a time)
        # print("\n   Combining all batches from disk...")
        # test_dfs = []
        # for batch_file in test_batch_files:
        #     with open(batch_file, 'rb') as f:
        #         batch_df = pickle.load(f)
        #         test_dfs.append(batch_df)
        
        # test_df = pd.concat(test_dfs, ignore_index=True)
        # print(f"   ✓ Combined {len(test_df)} test images")
        # print(f"After combining test images RAM usage: {get_ram_usage():.2f} MB")
        # # Clean up temp files
        # del test_dfs
        # gc.collect()
        # print(f"After cleaning up test images RAM usage: {get_ram_usage():.2f} MB")
        # for batch_file in test_batch_files:
        #     os.remove(batch_file)
        # os.rmdir(temp_dir_test)
        # print(f"After cleaning up test images RAM usage: {get_ram_usage():.2f} MB")
        # # Display statistics
        # print("\n" + "=" * 60)
        # print("PREPROCESSING SUMMARY")
        # print("=" * 60)
        # print(f"\nTraining images: {len(train_df)}")
        # print(f"Test images: {len(test_df)}")

        # print("\nImage dimensions (training):")
        # print(
        #     f"  Width:  {train_df['width'].min()} - {train_df['width'].max()} (mean: {train_df['width'].mean():.1f})"
        # )
        # print(
        #     f"  Height: {train_df['height'].min()} - {train_df['height'].max()} (mean: {train_df['height'].mean():.1f})"
        # )

        # print("\nFirst few rows:")
        # print(train_df.head())

        # # Save to output directory (for SageMaker Processing)
        # print(f"\n4. Saving preprocessed data to {OUTPUT_DIR}...")
        
        # # Save pickle files WITH image data (for training)
        # train_folderkl = os.path.join(OUTPUT_DIR, 'train_data.pkl')
        # test_pkl = os.path.join(OUTPUT_DIR, 'test_data.pkl')
        # print(f"After saving pickle files RAM usage: {get_ram_usage():.2f} MB")
        # print(f"Saving pickle files (with image data)...")
        # with open(train_folderkl, 'wb') as f:
        #     pickle.dump(train_df, f)
        # print(f"✓ Saved {train_folderkl} ({os.path.getsize(train_folderkl) / 1024 / 1024:.1f} MB)")

        # with open(test_pkl, 'wb') as f:
        #     pickle.dump(test_df, f)
        # print(f"✓ Saved {test_pkl} ({os.path.getsize(test_pkl) / 1024 / 1024:.1f} MB)")
        # print(f"After saving test pickle file RAM usage: {get_ram_usage():.2f} MB")
        # # Save CSV files WITHOUT image data (for reference/inspection)
        # train_csv = os.path.join(OUTPUT_DIR, "train_metadata.csv")
        # test_csv = os.path.join(OUTPUT_DIR, "test_metadata.csv")
        
        # print(f"Saving CSV files (metadata only)...")
        # train_df.drop(columns=['image']).to_csv(train_csv, index=False)
        # test_df.drop(columns=['image']).to_csv(test_csv, index=False)
        # print(f"✓ Saved {train_csv}")
        # print(f"✓ Saved {test_csv}")
        # print(f"After saving CSV files RAM usage: {get_ram_usage():.2f} MB")
        # # Also save summary statistics
        # summary = {
        #     "train_count": len(train_df),
        #     "test_count": len(test_df),
        #     "width_min": int(train_df["width"].min()),
        #     "width_max": int(train_df["width"].max()),
        #     "width_mean": float(train_df["width"].mean()),
        #     "height_min": int(train_df["height"].min()),
        #     "height_max": int(train_df["height"].max()),
        #     "height_mean": float(train_df["height"].mean()),
        # }

        # summary_file = os.path.join(OUTPUT_DIR, "preprocessing_summary.txt")
        # with open(summary_file, "w") as f:
        #     for key, value in summary.items():
        #         f.write(f"{key}: {value}\n")
        # print(f"✓ Saved summary to {summary_file}")
        # print(f"After saving summary statistics RAM usage: {get_ram_usage():.2f} MB")
        # print("\n" + "=" * 60)
        # print("PREPROCESSING COMPLETE!")
        # print("=" * 60)
        # print(f"\nOutput files saved to: {OUTPUT_DIR}")
        # print(f"  - train_data.pkl (with image arrays)")
        # print(f"  - test_data.pkl (with image arrays)")
        # print(f"  - train_metadata.csv (metadata only)")
        # print(f"  - test_metadata.csv (metadata only)")
        # print(f"  - preprocessing_summary.txt")
        # print("\nSageMaker will automatically upload these to S3!")
        # print("Ready for training!")