import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import clip
from full_network import VL_phys
from vlloss import info_nce_psd_loss, info_nce_contrastive_loss
import os
from torch.utils.data import Dataset, DataLoader
from PIL import Image
from torchvision import transforms

os.environ["CUDA_VISIBLE_DEVICES"] = "2"


class VLDataset(Dataset):
    def __init__(self, root_dir, transform=None, tokenizer=clip.tokenize):

        self.root_dir = root_dir
        self.sample_dirs = [os.path.join(root_dir, d) for d in os.listdir(root_dir)
                            if os.path.isdir(os.path.join(root_dir, d))]
        self.transform = transform
        self.tokenizer = tokenizer
        self.nums = [0.25, 0.5, 0.75, 1.25, 1.5, 1.75]
        self.text_template = ("the frequency of the horizontal color variation on the left side is {} "
                              "times of that on the right side of the image")

    def __len__(self):
        return len(self.sample_dirs)

    def __getitem__(self, idx):
        sample_dir = self.sample_dirs[idx]
        images = []
        for i in range(1, 12):
            image_path = os.path.join(sample_dir, f"img{i}.jpg")
            image = Image.open(image_path).convert("RGB")
            if self.transform is not None:
                image = self.transform(image)
            images.append(image)
        images = torch.stack(images, dim=0)

        texts = [self.text_template.format(num) for num in self.nums]
        tokenized_texts = [self.tokenizer(text) for text in texts]
        tokenized_texts = torch.cat(tokenized_texts, dim=0)
        front_five = images[:5]
        last_six = images[5:]
        perm = torch.randperm(6)
        last_six_shuffled = last_six[perm]
        tokenized_texts_shuffled = tokenized_texts[perm]
        images = torch.cat([front_five, last_six_shuffled], dim=0)

        return images, tokenized_texts_shuffled


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((0.48145466, 0.4578275, 0.40821073),
                         (0.26862954, 0.26130258, 0.27577711))
])

traindataset = VLDataset(root_dir='./train', transform=transform, tokenizer=clip.tokenize)
traindataloader = DataLoader(traindataset, batch_size=4, shuffle=True)

testdataset = VLDataset(root_dir='./test', transform=transform, tokenizer=clip.tokenize)
testdataloader = DataLoader(testdataset, batch_size=4, shuffle=False)


def train(num_epochs=10, lr=1e-4):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VL_phys(device=device).to(device)
    optimizer = optim.Adam(model.parameters(), lr=lr)

    for epoch in range(num_epochs):
        model.train()
        for batch_idx, (images, texts) in enumerate(traindataloader):
            images = images.to(device)
            texts = texts.to(device)
            optimizer.zero_grad()
            signals, img_features_text, text_features, rec_patches, mask, gt_patches = model(images, texts)
            psd_loss = info_nce_psd_loss(signals)
            contrastive_loss = info_nce_contrastive_loss(img_features_text, text_features)
            if mask.sum() > 0:
                rec_loss = F.mse_loss(rec_patches[mask], gt_patches[mask])
            else:
                rec_loss = torch.tensor(0.0, device=device)
            total_loss = psd_loss + contrastive_loss + rec_loss
            total_loss.backward()
            optimizer.step()
            print(f"Epoch [{epoch + 1}/{num_epochs}], Batch [{batch_idx + 1}/{len(dataloader)}], Total Loss: {total_loss.item():.4f}")

def test():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = VL_phys(device=device).to(device)
    model.eval()
    total_batches = 0
    with torch.no_grad():
        for images, texts in testdataloader:
            images = images.to(device)
            texts = texts.to(device)
            signals, img_features_text, text_features, rec_patches, mask, gt_patches = model(images, texts)
            print(f"Batch {total_batches+1}:")
            print("  Signals shape:", signals.shape)
            print("  Image features text shape:", img_features_text.shape)
            print("  Text features shape:", text_features.shape)
            print("  Reconstructed patches shape:", rec_patches.shape)
            print("  Mask shape:", mask.shape)
            print("  GT patches shape:", gt_patches.shape)
            total_batches += 1
    print("Total test batches:", total_batches)

if __name__ == "__main__":
    train()
    test()