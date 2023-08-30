from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests
from io import BytesIO
from label_studio_ml.utils import DATA_UNDEFINED_NAME
import os
from download_model import MODEL_NAME, MODEL_CACHE_DIR

device = "cuda"
access_token = os.environ.get("LS_ACCESS_TOKEN")

processor_pre = AutoProcessor.from_pretrained(MODEL_NAME, cache_dir=MODEL_CACHE_DIR)
model_pre = Blip2ForConditionalGeneration.from_pretrained(
    MODEL_NAME, cache_dir=MODEL_CACHE_DIR
).to(device)


class BLIP2Model(LabelStudioMLBase):
    def __init__(self, project_id, model=MODEL_NAME, **kwargs):
        super(BLIP2Model, self).__init__(**kwargs)
        self.value = "captioning"
        self.hostname = "https://labelstudio.stablecog.com"
        self.model_name = model
        self.access_token = access_token
        self.processor = processor_pre
        self.model = model_pre

    def _get_image_url(self, task):
        image_url_relative = task["data"].get(self.value) or task["data"].get(
            DATA_UNDEFINED_NAME
        )
        image_url = self.hostname + image_url_relative
        """ if image_url.startswith("s3://"):
            # presign s3 url
            r = urlparse(image_url, allow_fragments=False)
            bucket_name = r.netloc
            key = r.path.lstrip("/")
            client = boto3.client("s3", endpoint_url=self.endpoint_url)
            try:
                image_url = client.generate_presigned_url(
                    ClientMethod="get_object",
                    Params={"Bucket": bucket_name, "Key": key},
                )
            except ClientError as exc:
                logger.warning(
                    f"Can't generate presigned URL for {image_url}. Reason: {exc}"
                ) """
        return image_url

    def _download_task_image(self, task):
        image_url = self._get_image_url(task)
        headers = {"Authorization": f"Token {self.access_token}"}
        response = requests.get(image_url, headers=headers)
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
            image = self._download_task_image(task)
            inputs = self.processor(image, return_tensors="pt").to(device)
            generated_ids = self.model.generate(**inputs, max_new_tokens=75)
            generated_text = self.processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip()
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
