import os
import json

import base64

from google.cloud import storage
from google.oauth2 import service_account

from .cloud_service import CloudService

class GoogleCloudService(CloudService):
    def download_files_from_bucket(self, bucket_name, destination_folder):

        #make tmp folder for downloading it from bucket
        os.makedirs(destination_folder, exist_ok=True)

        # Load base64-encoded JSON key from environment variable
        credentials_base64 = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS_BASE64")
        
        # Decode base64 to get the JSON string
        credentials_json = base64.b64decode(credentials_base64).decode("utf-8")
        
        # Convert JSON string to dictionary
        credentials_dict = json.loads(credentials_json)

        # Create a client using the loaded credentials
        credentials = service_account.Credentials.from_service_account_info(credentials_dict)
        client = storage.Client(credentials=credentials)


        # Get the bucket
        bucket = client.bucket(bucket_name)

        # List all files in the bucket
        blobs = bucket.list_blobs()

        # Download each file
        for blob in blobs:
            # Create destination path
            destination_path = os.path.join(destination_folder, blob.name)

            # Download the file
            blob.download_to_filename(destination_path)
            print(f"Downloaded: {blob.name} to {destination_path}")

