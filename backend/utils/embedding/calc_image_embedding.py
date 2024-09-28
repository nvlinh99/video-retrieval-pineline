import math
import torch
import torch.nn as nn
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from timm.models.registry import register_model
import torch.nn.functional as F
import numpy as np
from torchvision import transforms
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from torchvision.datasets.folder import default_loader
import timm
import os

from torchscale.model.BEiT3 import BEiT3
from torchscale.architecture.config import EncoderConfig
from utils.helper import index_pinecone_embedding

from huggingface_hub import hf_hub_download

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
##Global setup
MODE = "large" # "base" of "large", base(embedding size 768, model 0.4G) faster ~5times compare to large (embedding size 1024, model 1.26G)
MODEL = "coco" # "coco" or "f30k"
batch_size = 128

def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


def _get_base_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
        ):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=768, encoder_attention_heads=12,
        encoder_ffn_embed_dim=int(768 * mlp_ratio), encoder_layers=12,
        checkpoint_activations=checkpoint_activations,
    )

def _get_large_config(
        img_size=224, patch_size=16, drop_path_rate=0,
        checkpoint_activations=None, mlp_ratio=4, vocab_size=64010, **kwargs
        ):
    return EncoderConfig(
        img_size=img_size, patch_size=patch_size, vocab_size=vocab_size, multiway=True,
        layernorm_embedding=False, normalize_output=True, no_output_layer=True,
        drop_path_rate=drop_path_rate, encoder_embed_dim=1024, encoder_attention_heads=16,
        encoder_ffn_embed_dim=int(1024 * mlp_ratio), encoder_layers=24,
        checkpoint_activations=checkpoint_activations,
    )


class BEiT3Wrapper(nn.Module):
    def __init__(self, args, **kwargs):
        super().__init__()
        self.args = args
        self.beit3 = BEiT3(args)
        self.apply(self._init_weights)

    def fix_init_weight(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    def get_num_layers(self):
        return self.beit3.encoder.num_layers

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'beit3.encoder.embed_positions.A.weight', 'beit3.vision_embed.cls_token', 'logit_scale'}

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)


class BEiT3ForRetrieval(BEiT3Wrapper):
    def __init__(
            self,
            args,
            **kwargs
    ):
        super(BEiT3ForRetrieval, self).__init__(args=args)
        embed_dim = args.encoder_embed_dim
        self.language_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.vision_head = nn.Linear(embed_dim, embed_dim, bias=False)
        self.language_head.apply(self._init_weights)
        self.vision_head.apply(self._init_weights)
        self.criterion = None
        self.logit_scale = nn.Parameter(torch.ones([]) * np.log(1 / 0.07))

    def forward(self, image=None, text_description=None, padding_mask=None, only_infer=False, **kwargs):
        if image is not None:
            outputs = self.beit3(
                textual_tokens=None,
                visual_tokens=image,
                text_padding_position=None,
            )
            x = outputs["encoder_out"]
            vision_cls = self.vision_head(x[:, 0, :])
            vision_cls = F.normalize(vision_cls, dim=-1)
        else:
            vision_cls = None

        if text_description is not None:
            outputs = self.beit3(
                textual_tokens=text_description,
                visual_tokens=None,
                text_padding_position=padding_mask,
            )
            x = outputs["encoder_out"]
            language_cls = self.language_head(x[:, 0, :])
            language_cls = F.normalize(language_cls, dim=-1)
        else:
            language_cls = None

        if only_infer:
            return vision_cls, language_cls
        else:
            loss, logits_per_image, logits_per_text = self.criterion(
                vision_cls, language_cls, self.logit_scale.exp())
            return loss, vision_cls, language_cls


@register_model
def beit3_large_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_large_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

@register_model
def beit3_base_patch16_384_retrieval(pretrained=False, **kwargs):
    args = _get_base_config(img_size=384, **kwargs)
    model = BEiT3ForRetrieval(args, **kwargs)
    return model

def build_transform(input_size):
    custom_transform = transforms.Compose([
        transforms.Resize((input_size, input_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=IMAGENET_DEFAULT_MEAN, std=IMAGENET_DEFAULT_STD),
    ])
    return custom_transform

def to_image_tokens(image_paths):
    batch = []
    for image_path in image_paths:
        transform = build_transform(384)
        image = default_loader(image_path)
        image = transform(image)
        batch.append(image)
    return batch


def get_sentencepiece_model_for_beit3(model_path):
    from transformers import XLMRobertaTokenizer
    model_path = os.path.join(os.path.dirname(__file__), "beit3.spm")
    return XLMRobertaTokenizer(model_path)

if MODE == "base":
    model_name = "beit3_base_patch16_384_retrieval"
    if MODEL == "coco":
        ckpt_path = os.path.join(os.path.dirname(__file__), "beit3_base_patch16_384_coco_retrieval.pth")
    else:
        ckpt_path = os.path.join(os.path.dirname(__file__), "beit3_base_patch16_384_f30k_retrieval.pth")

else:
    model_name = "beit3_large_patch16_384_retrieval"
    if MODEL == "coco":
        ckpt_path = os.path.join(os.path.dirname(__file__), "beit3_large_patch16_384_coco_retrieval.pth")

    else:
        ckpt_path = os.path.join(os.path.dirname(__file__), "beit3_large_patch16_384_f30k_retrieval.pth")

# model = timm.models.create_model(model_name, pretrained=False)
# torch.cuda.empty_cache()
# model = model.to(device)
# checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)
# # Load the model weights from the checkpoint
# model.load_state_dict(checkpoint['model'])


# Download the checkpoint from Hugging Face
ckpt_path = hf_hub_download(repo_id="nvlinh99/beit3_large_patch16_384_coco_retrieval", filename="beit3_large_patch16_384_coco_retrieval.pth")

# Create the model using timm and load the checkpoint
model = timm.models.create_model(model_name, pretrained=False)
model = model.to(device)

# Load the model weights from the downloaded checkpoint
checkpoint = torch.load(ckpt_path, map_location=device, weights_only=True)

# If the checkpoint contains a 'model' key (state_dict)
model.load_state_dict(checkpoint['model'], strict=False)


def calc_img_embedding(image_paths):
    output_dict = dict()
    img_tokens = to_image_tokens(image_paths)
    img_tokens_stacked = torch.stack(img_tokens, dim=0)
    img_tokens_stacked = img_tokens_stacked.to(device)
    with torch.no_grad():
        img_embedding = model(image=img_tokens_stacked, only_infer=True)

    for idx, embed in enumerate(img_embedding[0]):
        output_dict[image_paths[idx]] = embed

    del img_tokens
    del img_tokens_stacked
    return output_dict

def image_ebedding(image_path, top_k=5, object_input=None):
    index = index_pinecone_embedding()
    img_embedding = calc_img_embedding(image_path)
    img_embedding_list = img_embedding[image_path[0]].cpu().detach().numpy().tolist()
    
    if object_input is not None:    
        filter={
            "object_labels": { "$in": object_input["object_name"] },
            "object_count": { "$in": object_input["object_count"] },
            "object_colors": { "$in": object_input["object_color"] },
            "ojbect_locations": { "$in": object_input["ojbect_location"] }
        }
        query_results = index.query(
            vector=img_embedding_list,
            top_k=top_k, 
            filter=filter,
            include_metadata=True,
            include_values=True
        )
    else:
        query_results = index.query(
            vector=img_embedding_list,
            top_k=top_k,
            include_metadata=True,
            include_values=True
        )
    return query_results


if __name__ == '__main__':
    image_paths = ["https://link.to/image.jpg"]
    image_embedding = calc_img_embedding(image_paths)
    print(image_embedding)
    print("Done")