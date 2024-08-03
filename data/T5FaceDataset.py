from torch.utils.data import Dataset
from transformers import Blip2Processor, T5EncoderModel
import glob
from PIL import Image
import torch


class T5FaceDataset(Dataset):
    def __init__(
        self, image_folder, processor: Blip2Processor, t5_encoder: T5EncoderModel
    ):
        image_files = glob.glob(image_folder + "**/*")
        items = []
        for image_file in image_files:
            person_name = image_file.split("/")[-1]
            items.append((image_file, person_name))
        self.items = items
        self.processor = processor
        self.t5_encoder = t5_encoder

    def __len__(self):
        return len(self.items)

    @torch.no_grad()
    def __getitem__(self, idx):
        image_file, person_name = self.items[idx]
        image = Image.open(image_file)
        prompt = f"Portrait of {person_name}"
        inputs = self.processor(
            image, prompt, return_tensors="pt", padding="max_length", max_length=10
        )

        y = self.t5_encoder(
            inputs["input_ids"], attention_mask=inputs["attention_mask"]
        )
        return inputs, y.last_hidden_state


if __name__ == "__main__":
    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    t5_encoder = T5EncoderModel.from_pretrained("city96/t5-v1_1-xxl-encoder-bf16")
    dataset = T5FaceDataset("portraits/", processor, t5_encoder)
    x, y = dataset[0]
    print(x)
    print(y)
    print(y.shape)
