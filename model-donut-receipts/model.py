from typing import List, Dict, Optional
from label_studio_ml.model import LabelStudioMLBase
from label_studio_ml.response import ModelResponse
import receipts_inference_model
import json
import os


class NewModel(LabelStudioMLBase):
    """Custom ML Backend model
    """
    def setup(self):
        """Configure any parameters of your model here
        """
        print('### SETUP ###')
        self.set("model_version", "0.0.2")

    def predict(self, tasks: List[Dict], context: Optional[Dict] = None, **kwargs) -> ModelResponse:
        print('### PREDICT ###')

        if not hasattr(self, 'model'):
            self.model = receipts_inference_model.InferenceModel('AdamCodd/donut-receipts-extract')

        """ Write your inference logic here
            :param tasks: [Label Studio tasks in JSON format](https://labelstud.io/guide/task_format.html)
            :param context: [Label Studio context in JSON format](https://labelstud.io/guide/ml_create#Implement-prediction-logic)
            :return model_response
                ModelResponse(predictions=predictions) with
                predictions: [Predictions array in JSON format](https://labelstud.io/guide/export.html#Label-Studio-JSON-format-of-annotated-tasks)
        """
        print(f'''\
        Run prediction on {tasks}
        Received context: {context}
        Project ID: {self.project_id}
        Label config: {self.label_config}
        Parsed JSON Label config: {self.parsed_label_config}
        Extra params: {self.extra_params}''')

        from_name, to_name, value = self.label_interface.get_first_tag_occurence('TextArea', 'Image')

        DATA = os.path.expanduser('~') + "/Library/Application Support/label-studio/media/"
        image_path = tasks[0]['data']['image'].replace('/data', DATA)
        print(image_path)
        model_json = self.model.generate_text_from_image(image_path)
        result = {
            'result': [{
                    'from_name': from_name,
                    'to_name': to_name,
                    'type': 'textarea',
                    'value': {'text': json.dumps(model_json, indent=4)}
                }],
                'model_version': self.get('model_version')
            }
        predictions = [result]
        return ModelResponse(predictions=predictions, model_version=self.get('model_version'))
    
    def fit(self, event, data, **kwargs):
        """
        This method is called each time an annotation is created or updated
        You can run your logic here to update the model and persist it to the cache
        It is not recommended to perform long-running operations here, as it will block the main thread
        Instead, consider running a separate process or a thread (like RQ worker) to perform the training
        :param event: event type can be ('ANNOTATION_CREATED', 'ANNOTATION_UPDATED', 'START_TRAINING')
        :param data: the payload received from the event (check [Webhook event reference](https://labelstud.io/guide/webhook_reference.html))
        """

        # use cache to retrieve the data from the previous fit() runs
        old_data = self.get('my_data')
        old_model_version = self.get('model_version')
        print(f'Old data: {old_data}')
        print(f'Old model version: {old_model_version}')

        # store new data to the cache
        self.set('my_data', 'my_new_data_value')
        self.set('model_version', 'my_new_model_version')
        print(f'New data: {self.get("my_data")}')
        print(f'New model version: {self.get("model_version")}')

        print('fit() completed successfully.')

