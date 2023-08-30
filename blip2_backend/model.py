from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from transformers import AutoProcessor, Blip2ForConditionalGeneration
from PIL import Image
import requests


device = "cpu"


class NewModel(LabelStudioMLBase):
    def __init__(self, project_id, **kwargs):
        super(NewModel, self).__init__(**kwargs)
        self.blip_2_processor = None
        self.blip_2_model = None
        """ self.blip2_processor = AutoProcessor.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        )
        self.blip2_model = Blip2ForConditionalGeneration.from_pretrained(
            "Salesforce/blip2-opt-2.7b"
        ).to(device) """

    def predict(
        self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs
    ) -> List[Dict]:
        predictions = []
        from_name, schema = list(self.parsed_label_config.items())[0]
        to_name = schema["to_name"][0]
        for task in tasks:
            image_url = task["data"]["image"]
            image = Image.open(requests.get(image_url, stream=True).raw)
            """ inputs = self.blip2_processor(image, return_tensors="pt").to(device)
            generated_ids = self.blip2_model.generate(**inputs, max_new_tokens=20)
            generated_text = self.blip2_processor.batch_decode(
                generated_ids, skip_special_tokens=True
            )[0].strip() """
            generated_text = "This is a test"
            prediction = [
                {
                    "type": "textarea",
                    "value": {"text": [generated_text]},
                    "to_name": to_name,
                    "from_name": from_name,
                }
            ]
            predictions.append(prediction)

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
