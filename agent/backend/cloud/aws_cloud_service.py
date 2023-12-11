import os

import boto3

from .cloud_service import CloudService

class AWSCloudService(CloudService):
    def download_files_from_bucket(self, bucket_name, destination_folder):
        # make tmp folder for downloading it from bucket
        os.makedirs(destination_folder, exist_ok=True)

        # Boto3 retrieves credentials from various locations. The easiest being having a role provider for services run directly on aws or for local dev having set up the credentials in one of the ways mentioned in the documentation
        # https://boto3.amazonaws.com/v1/documentation/api/latest/guide/credentials.html
        resource = boto3.resource("s3")


        # Get the bucket
        bucket = resource.Bucket(bucket_name)

        # Download each file - folder structure of original bucket is ignored.
        for obj in bucket.objects.all():
            file_name = obj.key.rsplit('/', 1)[-1]
            if file_name == "":
                continue
            
            # Create destination path
            destination_path = os.path.join(destination_folder, file_name)

            # Download the file
            resource.meta.client.download_file(bucket_name, obj.key, destination_path)
            print(f"Downloaded: {obj.key} to {destination_path}")
