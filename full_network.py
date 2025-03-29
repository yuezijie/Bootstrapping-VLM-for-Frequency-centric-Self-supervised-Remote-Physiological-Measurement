import torch
import torch.nn as nn
import torch.nn.functional as F
import clip
from base_network import Vision_encoder, Text_encoder, load_clip_state_dict
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class ReconstructionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, ffn_hidden_dim):
        super().__init__()
        self.ln1 = nn.LayerNorm(embed_dim)
        self.self_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)

        self.ln2 = nn.LayerNorm(embed_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim, num_heads=num_heads)

        self.ln3 = nn.LayerNorm(embed_dim)
        self.ffn = nn.Sequential(
            nn.Linear(embed_dim, ffn_hidden_dim),
            nn.ReLU(),
            nn.Linear(ffn_hidden_dim, embed_dim)
        )

    def forward(self, x, text_features):
        x_norm = self.ln1(x)
        self_attn_out, _ = self.self_attn(x_norm, x_norm, x_norm)
        x = x + self_attn_out
        x_norm = self.ln2(x)
        text_features_unsq = text_features.unsqueeze(0)
        cross_attn_out, _ = self.cross_attn(x_norm, text_features_unsq, text_features_unsq)
        x = x + cross_attn_out
        x_norm = self.ln3(x)
        x_trans = x_norm.transpose(0, 1)
        ffn_out = self.ffn(x_trans)
        ffn_out = ffn_out.transpose(0, 1)
        x = x + ffn_out

        return x


class ReconstructionHead(nn.Module):
    def __init__(self, embed_dim, out_channels, scale_factor):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(embed_dim, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(embed_dim, out_channels, kernel_size=3, padding=1)
        )
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')

    def forward(self, x):
        x = self.conv(x)
        x = self.upsample(x)
        return x


class VisualReconstructionModule(nn.Module):
    def __init__(self, patch_size=16, in_channels=3, embed_dim=512, num_heads=8, num_blocks=3, ffn_hidden_dim=2048):
        super().__init__()
        self.patch_size = patch_size
        self.patch_dim = in_channels * patch_size * patch_size  # 例如 3*16*16 = 768
        self.embed_dim = embed_dim
        self.in_channels = in_channels
        self.patch_embed = nn.Linear(self.patch_dim, embed_dim)
        self.mask_token = nn.Parameter(torch.randn(embed_dim))
        self.blocks = nn.ModuleList([
            ReconstructionBlock(embed_dim, num_heads, ffn_hidden_dim)
            for _ in range(num_blocks)
        ])
        self.reconstruction_head = ReconstructionHead(embed_dim, out_channels=in_channels, scale_factor=patch_size)

    def extract_patches(self, images):

        B, C, H, W = images.shape
        patches = images.unfold(2, self.patch_size, self.patch_size).unfold(3, self.patch_size, self.patch_size)
        patches = patches.permute(0, 2, 3, 1, 4, 5).contiguous()
        B, num_h, num_w, C, ph, pw = patches.shape
        num_patches = num_h * num_w
        patches = patches.view(B, num_patches, C * ph * pw)
        return patches

    def forward(self, images, text_features):

        patches = self.extract_patches(images)  # (B, num_patches, patch_dim)
        gt_patches = patches
        patch_tokens = self.patch_embed(patches)  # (B, num_patches, embed_dim)
        B, num_patches, _ = patch_tokens.shape
        mask = (torch.rand(B, num_patches, device=patch_tokens.device) < 0.6)
        mask_token_expanded = self.mask_token.unsqueeze(0).unsqueeze(0).expand(B, num_patches, self.embed_dim)
        patch_tokens = torch.where(mask.unsqueeze(-1), mask_token_expanded, patch_tokens)
        H_patch = images.shape[2] // self.patch_size
        W_patch = images.shape[3] // self.patch_size
        x = patch_tokens.view(B, H_patch, W_patch, self.embed_dim).permute(0, 3, 1, 2)
        L = H_patch * W_patch
        x_seq = x.reshape(B, self.embed_dim, L).permute(2, 0, 1)  # (L, B, embed_dim)
        for block in self.blocks:
            x_seq = block(x_seq, text_features)
        x = x_seq.permute(1, 2, 0).reshape(B, self.embed_dim, H_patch, W_patch)
        rec_image = self.reconstruction_head(x)
        rec_patches = self.extract_patches(rec_image)
        return rec_patches, mask, gt_patches


class VL_phys(nn.Module):
    def __init__(self, device, signal_length=224):

        super().__init__()
        self.device = device
        input_resolution = 224
        patch_size = 32
        visual_width = 768
        visual_layers = 12
        visual_heads = 12
        visual_output_dim = 512

        vocab_size = 49408
        context_length = 77
        text_width = 512
        text_layers = 12
        text_heads = 8
        text_output_dim = 512

        self.vision_encoder = Vision_encoder(input_resolution, patch_size, visual_width, visual_layers, visual_heads, visual_output_dim)
        self.text_encoder = Text_encoder(vocab_size, context_length, text_width, text_layers, text_heads, text_output_dim)
        load_clip_state_dict(self.vision_encoder, self.text_encoder, "./model/pre_model.bin")

        self.projection = nn.Linear(visual_output_dim, signal_length)
        self.visual_reconstructor = VisualReconstructionModule(patch_size=16, in_channels=3, embed_dim=512)

    def forward(self, images, texts):

        B, N, C, H, W = images.shape
        images_flat = images.view(B * N, C, H, W)
        img_features_all_flat = self.vision_encoder(images_flat)
        img_features_all = img_features_all_flat.view(B, N, -1)

        signals = self.projection(img_features_all[:, :5, :])
        img_features_text = img_features_all[:, 5:, :]


        B_text, num_texts, token_length = texts.shape
        texts_flat = texts.view(B_text * num_texts, token_length)
        text_features_flat = self.text_encoder(texts_flat)
        text_features = text_features_flat.view(B_text, num_texts, -1)

        rec_patches_list = []
        mask_list = []
        gt_patches_list = []
        for i in range(B):
            images_for_rec = images[i, 5:, :, :, :]
            text_features_sample = text_features[i]
            rec_patches_sample, mask_sample, gt_patches_sample = self.visual_reconstructor(images_for_rec, text_features_sample)
            rec_patches_list.append(rec_patches_sample)
            mask_list.append(mask_sample)
            gt_patches_list.append(gt_patches_sample)
        rec_patches = torch.stack(rec_patches_list, dim=0)
        mask = torch.stack(mask_list, dim=0)
        gt_patches = torch.stack(gt_patches_list, dim=0)

        return signals, img_features_text, text_features, rec_patches, mask, gt_patches