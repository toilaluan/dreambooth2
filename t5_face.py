import torch
import torch.nn as nn
import transformers
from typing import Optional


class T5Face(nn.Module):
    def __init__(
        self,
        qformer_id: str,
        t5_encoder_id: str,
        vision_model_id: str,
        num_query_tokens=32,
        num_face_tokens=4,
        torch_dtype=torch.float16,
    ):
        super(T5Face, self).__init__()
        print(transformers.Blip2Config.from_pretrained(qformer_id))
        self.qformer: transformers.Blip2QFormerModel = (
            transformers.Blip2QFormerModel.from_pretrained(
                qformer_id, torch_dtype=torch_dtype
            )
        )
        self.t5_encoder: transformers.T5EncoderModel = (
            transformers.T5EncoderModel.from_pretrained(
                t5_encoder_id, torch_dtype=torch_dtype
            )
        )
        self.language_projection = nn.Linear(
            self.qformer.config.hidden_size,
            self.t5_encoder.config.hidden_size,
            dtype=torch_dtype,
        )
        self.query_tokens = nn.Parameter(
            torch.zeros(
                1, num_query_tokens, self.qformer.config.hidden_size, dtype=torch_dtype
            )
        )
        self.face_tokens = nn.Parameter(
            torch.zeros(
                1,
                num_face_tokens,
                self.t5_encoder.config.hidden_size,
                dtype=torch_dtype,
            )
        )
        self.vision_model: transformers.Blip2VisionModel = (
            transformers.Blip2VisionModel.from_pretrained(
                vision_model_id, torch_dtype=torch_dtype
            )
        )

    def forward(
        self,
        pixel_values: torch.FloatTensor,
        input_ids: torch.FloatTensor,
        attention_mask: Optional[torch.LongTensor] = None,
        decoder_input_ids: Optional[torch.LongTensor] = None,
        decoder_attention_mask: Optional[torch.LongTensor] = None,
        output_attentions: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        labels: Optional[torch.LongTensor] = None,
        return_dict: Optional[bool] = None,
        interpolate_pos_encoding: bool = False,
        n_face_tokens: int = 4,
    ):
        vision_outputs = self.vision_model(
            pixel_values=pixel_values,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            interpolate_pos_encoding=interpolate_pos_encoding,
        )
        image_embeds = vision_outputs[0]

        image_attention_mask = torch.ones(
            image_embeds.size()[:-1], dtype=torch.long, device=image_embeds.device
        )
        query_tokens = self.query_tokens.expand(image_embeds.shape[0], -1, -1)
        query_outputs = self.qformer(
            query_embeds=query_tokens,
            encoder_hidden_states=image_embeds,
            encoder_attention_mask=image_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        query_output = query_outputs[0]
        language_model_inputs = self.language_projection(query_output)
        language_model_attention_mask = torch.ones(
            language_model_inputs.size()[:-1],
            dtype=torch.long,
            device=language_model_inputs.device,
        )
        inputs_embeds = self.t5_encoder.get_input_embeddings()(input_ids)
        face_tokens = self.face_tokens.expand(inputs_embeds.shape[0], -1, -1)
        inputs_embeds = torch.cat(
            [
                language_model_inputs,
                inputs_embeds.to(language_model_inputs.device),
                face_tokens.to(language_model_inputs.device),
            ],
            dim=1,
        )
        face_attention_mask = torch.zeros(
            face_tokens.size()[:-1], dtype=torch.long, device=face_tokens.device
        )
        face_attention_mask[:, :n_face_tokens] = 1

        if attention_mask is None:
            attention_mask = torch.ones_like(input_ids)
        expected_device = language_model_attention_mask.device
        attention_mask = torch.cat(
            [
                language_model_attention_mask,
                attention_mask.to(expected_device),
                face_attention_mask.to(expected_device),
            ],
            dim=1,
        )

        t5_encoder_outputs = self.t5_encoder(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        t5_encoder_outputs = t5_encoder_outputs.last_hidden_state[
            :, -(len(input_ids) + n_face_tokens), :
        ]

        return t5_encoder_outputs


if __name__ == "__main__":
    from transformers import Blip2Processor
    import requests
    from PIL import Image

    processor = Blip2Processor.from_pretrained("Salesforce/blip2-flan-t5-xxl")
    img_url = (
        "https://storage.googleapis.com/sfr-vision-language-research/BLIP/demo.jpg"
    )
    raw_image = Image.open(requests.get(img_url, stream=True).raw).convert("RGB")

    question = "Portrait of "
    inputs = processor(raw_image, question, return_tensors="pt").to(
        "cuda", torch.float16
    )
    model = T5Face(
        qformer_id="toilaluan/qformer",
        t5_encoder_id="city96/t5-v1_1-xxl-encoder-bf16",
        vision_model_id="toilaluan/vision_blip2",
    )
    model.to("cuda")
    model.eval()
    output = model(**inputs)
    print(output)
    # print(model)
    print("Model loaded successfully!")
