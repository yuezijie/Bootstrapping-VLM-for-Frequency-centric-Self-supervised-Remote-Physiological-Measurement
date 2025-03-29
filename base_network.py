import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from modules import Transformer

class Vision_encoder(nn.Module):
    def __init__(self, input_resolution, patch_size, width, layers, heads, output_dim):
        super().__init__()
        self.conv1 = nn.Conv2d(3, width, kernel_size=patch_size, stride=patch_size, bias=False)
        self.class_embedding = nn.Parameter(torch.zeros(width))
        n_patches = (input_resolution // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.zeros(n_patches + 1, width))
        self.transformer = Transformer(layers, width, heads)
        self.ln_post = nn.LayerNorm(width)
        self.proj = nn.Parameter(torch.empty(width, output_dim))
        self.initialize_parameters()
    def initialize_parameters(self):
        nn.init.normal_(self.conv1.weight, std=self.conv1.in_channels ** -0.5)
        nn.init.normal_(self.class_embedding, std=0.01)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.proj, std=self.proj.shape[0] ** -0.5)
    def forward(self, x):
        x = self.conv1(x)  # -> (batch, width, H/patch, W/patch)
        x = x.reshape(x.shape[0], x.shape[1], -1)  # -> (batch, width, n_patches)
        x = x.permute(0, 2, 1)  # -> (batch, n_patches, width)
        cls_tokens = self.class_embedding.unsqueeze(0).unsqueeze(1).expand(x.shape[0], 1, -1)
        x = torch.cat([cls_tokens, x], dim=1)  # (batch, n_patches+1, width)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x_cls = self.ln_post(x[:, 0, :])
        x_proj = x_cls @ self.proj
        return x_proj


class Text_encoder(nn.Module):
    def __init__(self, vocab_size, context_length, width, layers, heads, output_dim):
        super().__init__()
        self.token_embedding = nn.Embedding(vocab_size, width)
        self.positional_embedding = nn.Parameter(torch.empty(context_length, width))
        self.transformer = Transformer(layers, width, heads)
        self.ln_final = nn.LayerNorm(width)
        self.text_projection = nn.Parameter(torch.empty(width, output_dim))
        self.context_length = context_length
        self.initialize_parameters()
    def initialize_parameters(self):
        nn.init.normal_(self.token_embedding.weight, std=0.02)
        nn.init.normal_(self.positional_embedding, std=0.01)
        nn.init.normal_(self.text_projection, std=self.text_projection.shape[0] ** -0.5)
    def forward(self, tokens):
        x = self.token_embedding(tokens)  # (batch, context_length, width)
        x = x + self.positional_embedding
        x = self.transformer(x)
        x = self.ln_final(x)
        x = x[:, -1, :]
        x_proj = x @ self.text_projection
        return x_proj


def load_clip_state_dict(model_visual, model_text, checkpoint_path):

    state_dict = torch.load(checkpoint_path, map_location="cpu")
    mapped_visual = {}
    mapped_text = {}

    for key, value in state_dict.items():
        if key.startswith("vision_model.embeddings."):
            sub_key = key[len("vision_model.embeddings."):]
            if sub_key == "patch_embedding.weight":
                mapped_visual["conv1.weight"] = value
            elif sub_key == "class_embedding":
                mapped_visual["class_embedding"] = value
            elif sub_key == "position_embedding.weight":
                mapped_visual["positional_embedding"] = value

    if "vision_model.post_layernorm.weight" in state_dict:
        mapped_visual["ln_post.weight"] = state_dict["vision_model.post_layernorm.weight"]
    if "vision_model.post_layernorm.bias" in state_dict:
        mapped_visual["ln_post.bias"] = state_dict["vision_model.post_layernorm.bias"]

    temp_blocks = {}
    for key, value in state_dict.items():
        if key.startswith("vision_model.encoder.layers."):
            rest = key[len("vision_model.encoder.layers."):]
            block_idx, sub_key = rest.split(".", 1)
            if block_idx not in temp_blocks:
                temp_blocks[block_idx] = {}
            temp_blocks[block_idx][sub_key] = value

    for block_idx, block_dict in temp_blocks.items():
        prefix_new = f"transformer.resblocks.{block_idx}."
        # layer norm
        if "layer_norm1.weight" in block_dict:
            mapped_visual[prefix_new + "ln_1.weight"] = block_dict["layer_norm1.weight"]
        if "layer_norm1.bias" in block_dict:
            mapped_visual[prefix_new + "ln_1.bias"] = block_dict["layer_norm1.bias"]
        if "layer_norm2.weight" in block_dict:
            mapped_visual[prefix_new + "ln_2.weight"] = block_dict["layer_norm2.weight"]
        if "layer_norm2.bias" in block_dict:
            mapped_visual[prefix_new + "ln_2.bias"] = block_dict["layer_norm2.bias"]
        q_w = block_dict.get("self_attn.q_proj.weight", None)
        k_w = block_dict.get("self_attn.k_proj.weight", None)
        v_w = block_dict.get("self_attn.v_proj.weight", None)
        if q_w is not None and k_w is not None and v_w is not None:
            mapped_visual[prefix_new + "attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        q_b = block_dict.get("self_attn.q_proj.bias", None)
        k_b = block_dict.get("self_attn.k_proj.bias", None)
        v_b = block_dict.get("self_attn.v_proj.bias", None)
        if q_b is not None and k_b is not None and v_b is not None:
            mapped_visual[prefix_new + "attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        if "self_attn.out_proj.weight" in block_dict:
            mapped_visual[prefix_new + "attn.out_proj.weight"] = block_dict["self_attn.out_proj.weight"]
        if "self_attn.out_proj.bias" in block_dict:
            mapped_visual[prefix_new + "attn.out_proj.bias"] = block_dict["self_attn.out_proj.bias"]
        if "mlp.fc1.weight" in block_dict:
            mapped_visual[prefix_new + "mlp.fc1.weight"] = block_dict["mlp.fc1.weight"]
        if "mlp.fc1.bias" in block_dict:
            mapped_visual[prefix_new + "mlp.fc1.bias"] = block_dict["mlp.fc1.bias"]
        if "mlp.fc2.weight" in block_dict:
            mapped_visual[prefix_new + "mlp.fc2.weight"] = block_dict["mlp.fc2.weight"]
        if "mlp.fc2.bias" in block_dict:
            mapped_visual[prefix_new + "mlp.fc2.bias"] = block_dict["mlp.fc2.bias"]

    if "visual_projection.weight" in state_dict:
        mapped_visual["proj"] = state_dict["visual_projection.weight"].t()

    for key, value in state_dict.items():
        if key.startswith("text_model.embeddings."):
            sub_key = key[len("text_model.embeddings."):]
            if sub_key == "token_embedding.weight":
                mapped_text["token_embedding.weight"] = value
            elif sub_key == "position_embedding.weight":
                mapped_text["positional_embedding"] = value

    temp_blocks_text = {}
    for key, value in state_dict.items():
        if key.startswith("text_model.encoder.layers."):
            rest = key[len("text_model.encoder.layers."):]
            block_idx, sub_key = rest.split(".", 1)
            if block_idx not in temp_blocks_text:
                temp_blocks_text[block_idx] = {}
            temp_blocks_text[block_idx][sub_key] = value

    for block_idx, block_dict in temp_blocks_text.items():
        prefix_new = f"transformer.resblocks.{block_idx}."
        if "layer_norm1.weight" in block_dict:
            mapped_text[prefix_new + "ln_1.weight"] = block_dict["layer_norm1.weight"]
        if "layer_norm1.bias" in block_dict:
            mapped_text[prefix_new + "ln_1.bias"] = block_dict["layer_norm1.bias"]
        if "layer_norm2.weight" in block_dict:
            mapped_text[prefix_new + "ln_2.weight"] = block_dict["layer_norm2.weight"]
        if "layer_norm2.bias" in block_dict:
            mapped_text[prefix_new + "ln_2.bias"] = block_dict["layer_norm2.bias"]
        q_w = block_dict.get("self_attn.q_proj.weight", None)
        k_w = block_dict.get("self_attn.k_proj.weight", None)
        v_w = block_dict.get("self_attn.v_proj.weight", None)
        if q_w is not None and k_w is not None and v_w is not None:
            mapped_text[prefix_new + "attn.in_proj_weight"] = torch.cat([q_w, k_w, v_w], dim=0)
        q_b = block_dict.get("self_attn.q_proj.bias", None)
        k_b = block_dict.get("self_attn.k_proj.bias", None)
        v_b = block_dict.get("self_attn.v_proj.bias", None)
        if q_b is not None and k_b is not None and v_b is not None:
            mapped_text[prefix_new + "attn.in_proj_bias"] = torch.cat([q_b, k_b, v_b], dim=0)
        if "self_attn.out_proj.weight" in block_dict:
            mapped_text[prefix_new + "attn.out_proj.weight"] = block_dict["self_attn.out_proj.weight"]
        if "self_attn.out_proj.bias" in block_dict:
            mapped_text[prefix_new + "attn.out_proj.bias"] = block_dict["self_attn.out_proj.bias"]
        if "mlp.fc1.weight" in block_dict:
            mapped_text[prefix_new + "mlp.fc1.weight"] = block_dict["mlp.fc1.weight"]
        if "mlp.fc1.bias" in block_dict:
            mapped_text[prefix_new + "mlp.fc1.bias"] = block_dict["mlp.fc1.bias"]
        if "mlp.fc2.weight" in block_dict:
            mapped_text[prefix_new + "mlp.fc2.weight"] = block_dict["mlp.fc2.weight"]
        if "mlp.fc2.bias" in block_dict:
            mapped_text[prefix_new + "mlp.fc2.bias"] = block_dict["mlp.fc2.bias"]

    if "text_model.final_layer_norm.weight" in state_dict:
        mapped_text["ln_final.weight"] = state_dict["text_model.final_layer_norm.weight"]
    if "text_model.final_layer_norm.bias" in state_dict:
        mapped_text["ln_final.bias"] = state_dict["text_model.final_layer_norm.bias"]
    if "text_projection.weight" in state_dict:
        mapped_text["text_projection"] = state_dict["text_projection.weight"].t()

    model_visual.load_state_dict(mapped_visual, strict=True)
    model_text.load_state_dict(mapped_text, strict=True)

