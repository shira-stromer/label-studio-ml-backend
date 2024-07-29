import PIL.Image
import torch
import re
import PIL
from datasets import load_dataset
import glob
import json
import os
import torch.nn as nn
from transformers import DonutProcessor, VisionEncoderDecoderModel
from transformers import VisionEncoderDecoderConfig


class InferenceModel(nn.Module):
    def __init__(self, hf_donut_model, device=None) -> None:
        super().__init__()
        if not device:
            self.device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
        else:
            self.device = torch.device(device)

        self.config = VisionEncoderDecoderConfig.from_pretrained(hf_donut_model)
        self.task = self.config.task if hasattr(self.config, 'task') else '<s_receipt>'
        self.processor = DonutProcessor.from_pretrained(hf_donut_model)
        self.model = VisionEncoderDecoderModel.from_pretrained(hf_donut_model).to(device)

    def forward(self, image_path:str):
        text = self.generate_text_from_image(image_path)
        return text
     
    def load_and_preprocess_image(self, image_path: str, processor):
        """
        Load an image and preprocess it for the model.
        """
        image = PIL.Image.open(image_path).convert("RGB")
        pixel_values = processor(image, return_tensors="pt").pixel_values
        return pixel_values

    def generate_text_from_image(self, image_path: str):
        """
        Generate text from an image using the trained model.
        """
        # Load and preprocess the image
        pixel_values = self.load_and_preprocess_image(image_path, self.processor)
        pixel_values = pixel_values.to(self.device)

        # Generate output using model
        self.model.eval()
        with torch.no_grad():
            task_prompt = self.task # <s_cord-v2> for v1
            decoder_input_ids = self.processor.tokenizer(task_prompt, add_special_tokens=False, return_tensors="pt").input_ids
            decoder_input_ids = decoder_input_ids.to(self.device)
            generated_outputs = self.model.generate(
                pixel_values,
                decoder_input_ids=decoder_input_ids,
                max_length=self.model.decoder.config.max_position_embeddings,
                early_stopping=True,
                pad_token_id=self.processor.tokenizer.pad_token_id,
                eos_token_id=self.processor.tokenizer.eos_token_id,
                bad_words_ids=[[self.processor.tokenizer.unk_token_id]],
                num_beams=1,
                return_dict_in_generate=True
            )

        # Decode generated output
        decoded_text = self.processor.batch_decode(generated_outputs.sequences)[0]
        decoded_text = decoded_text.replace(self.processor.tokenizer.eos_token, "").replace(self.processor.tokenizer.pad_token, "")
        decoded_text = re.sub(r"<.*?>", "", decoded_text, count=1).strip()  # remove first task start token
        decoded_text = self.processor.token2json(decoded_text)
        return decoded_text

if __name__ == "__main__":
    model = InferenceModel('AdamCodd/donut-receipts-extract')

    i = 1
    for image_path in glob.glob("receipt_images/*.png"):
        receipt = model(image_path)
        json_path = image_path.replace(".png", ".json")
        if os.path.exists(json_path):
            continue
        json.dump(receipt, open(json_path, "x"), indent=4)
        i += 1
        if i == 10:
            exit()
    """
    
    ds = load_dataset('naver-clova-ix/cord-v2')
    images = []
    for t in ['train', 'validation', 'test']:
        images.extend(ds[t])

    i = 1
    for doc in images:
        file_ext = doc['image'].format.lower()
        doc['image'].save(f'receipt_images/{i}.{file_ext}')
        i += 1
    """