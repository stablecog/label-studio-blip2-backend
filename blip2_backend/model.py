from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from label_studio_ml.utils import DATA_UNDEFINED_NAME
import os
from download_model import MODEL_NAME, MODEL_CACHE_DIR
import time
from azure.storage.blob import BlobServiceClient, BlobSasPermissions, generate_blob_sas
from datetime import datetime, timedelta
import urllib.parse
import torch


label_studio_access_token = os.environ.get("LABEL_STUDIO_ACCESS_TOKEN")
azure_connection_string = os.environ.get("AZURE_CONNECTION_STRING")

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
torch_float = torch.float16 if device == "cuda" else torch.float32
processor_pre = None
model_pre = None
model_settings = {
    "max_length": 40,
    "min_length": 10,
    "top_p": 0.9,
    "num_beams": 4,
    "repetition_penalty": 1.5,
    "length_penalty": 1.5,
    "do_sample": False,
}


def load_model():
    global processor_pre, model_pre
    s = time.time()
    print(f"Loading model: {MODEL_NAME}")
    processor_pre = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
    model_pre = Blip2ForConditionalGeneration.from_pretrained(
        MODEL_NAME, torch_dtype=torch_float, cache_dir=MODEL_CACHE_DIR
    )
    print(f"Loaded model in: {round(time.time() - s, 2)} seconds")
    s = time.time()
    print(f"Moving model to: {device}")
    model_pre.to(device)
    print(f"Moved model in: {round(time.time() - s, 2)} seconds")


load_model()


def generate_presigned_url(azure_url, expiry_time=1):
    # Parse the URL to extract the container and blob name
    if azure_url.startswith("azure-blob://"):
        parts = azure_url.replace("azure-blob://", "").split("/", 1)
        if len(parts) < 2:
            raise ValueError("Invalid Azure Blob URL")
        container_name, blob_name = parts
    else:
        raise ValueError("URL must start with azure-blob://")

    # Initialize BlobServiceClient
    blob_service_client = BlobServiceClient.from_connection_string(
        azure_connection_string
    )
    blob_client = blob_service_client.get_blob_client(container_name, blob_name)

    # Generate a SAS token using blob client properties
    sas_token = generate_blob_sas(
        account_name=blob_client.account_name,
        container_name=container_name,
        blob_name=blob_name,
        account_key=blob_service_client.credential.account_key,
        permission=BlobSasPermissions(read=True),
        expiry=datetime.utcnow() + timedelta(hours=expiry_time),
    )

    # Generate the presigned URL
    presigned_url = f"https://{blob_client.account_name}.blob.core.windows.net/{container_name}/{urllib.parse.quote(blob_name)}?{sas_token}"

    return presigned_url


class BLIP2Model(LabelStudioMLBase):
    def __init__(self, project_id, model=MODEL_NAME, **kwargs):
        super(BLIP2Model, self).__init__(**kwargs)
        self.value = "captioning"
        self.hostname = "https://labelstudio.stablecog.com"
        self.model_name = model
        self.access_token = label_studio_access_token
        self.processor = processor_pre
        self.model = model_pre

    def _get_image_url(self, task):
        image_url = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        headers = None
        if image_url.startswith("/"):
            image_url = self.hostname + image_url
            headers = {"Authorization": f"Token {self.access_token}"}
        if image_url.startswith("azure-blob://"):
            image_url = generate_presigned_url(
                image_url,
            )
        return image_url, headers

    def _download_task_image(self, task):
        image_url, headers = self._get_image_url(task)
        req = {"url": image_url}
        if headers:
            req["headers"] = headers
        response = requests.get(**req)
        if response.status_code != 200:
            raise Exception(f"Failed to download image from {image_url}")
        return Image.open(BytesIO(response.content)).convert("RGB")

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        predictions = []
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema["to_name"][0]
        for task in tasks:
            s = time.time()
            image = self._download_task_image(task)
            inputs = self.processor(image, return_tensors="pt").to(device, torch_float)
            generated_ids = self.model.generate(**inputs, **model_settings)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
            e = time.time()
            print(f"🚀 Generated caption in: {round(e - s, 2)} seconds")
            result = [
                {
                    "type": "textarea",
                    "value": {"text": [generated_text]},
                    "to_name": to_name,
                    "from_name": from_name,
                }
            ]
            predictions.append(
                {
                    "result": result,
                }
            )

        """Write your inference logic here
        :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
        :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml.html#Passing-data-to-ML-backend)
        :return predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Raw-JSON-format-of-completed-tasks)
        """
        print(
            f"""\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}"""
        )
        return predictions

    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get("my_data")
        old_model_version = self.get("model_version")
        print(f"Old data: {old_data}")
        print(f"Old model version: {old_model_version}")

        # store new data to the cache
        self.set("my_data", "my_new_data_value")
        self.set("model_version", "my_new_model_version")
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print("fit() completed successfully.")
