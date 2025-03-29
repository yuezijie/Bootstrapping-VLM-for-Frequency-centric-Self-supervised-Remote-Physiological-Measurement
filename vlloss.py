import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim


def compute_psd(signal):
    fft_result = torch.fft.rfft(signal)
    psd = torch.abs(fft_result) ** 2
    return psd


def info_nce_psd_loss(signals, temperature=0.08):

    batch_size = signals.size(0)
    num_signals = signals.size(1)
    if num_signals < 3:
        raise ValueError("每个样本至少需要3个信号用于计算损失")
    loss_total = 0.0
    for b in range(batch_size):
        sample_signals = signals[b]

        psd_list = [compute_psd(sample_signals[i]) for i in range(num_signals)]
        def cosine_sim(a, b):
            return F.cosine_similarity(a.unsqueeze(0), b.unsqueeze(0))
        anchor = psd_list[0]
        positive = psd_list[1]
        negatives = [psd_list[i] for i in range(2, num_signals)]
        pos_sim = cosine_sim(anchor, positive)
        neg_sims = torch.stack([cosine_sim(anchor, neg) for neg in negatives])
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        logits = logits / temperature
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        loss0 = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))
        anchor = psd_list[1]
        positive = psd_list[0]
        negatives = [psd_list[i] for i in range(2, num_signals)]
        pos_sim = cosine_sim(anchor, positive)
        neg_sims = torch.stack([cosine_sim(anchor, neg) for neg in negatives])
        logits = torch.cat([pos_sim.unsqueeze(0), neg_sims])
        logits = logits / temperature
        labels = torch.zeros(1, dtype=torch.long, device=logits.device)
        loss1 = F.cross_entropy(logits.unsqueeze(0), labels.unsqueeze(0))

        loss_total += (loss0 + loss1) / 2

    return loss_total / batch_size


def info_nce_contrastive_loss(image_features, text_features, temperature=0.08):

    batch_size = image_features.size(0)
    loss_total = 0.0

    for b in range(batch_size):
        img = F.normalize(image_features[b], dim=1)
        txt = F.normalize(text_features[b], dim=1)
        logits = torch.matmul(img, txt.t()) / temperature
        targets = torch.arange(logits.size(0), device=logits.device)
        loss_i = F.cross_entropy(logits, targets)
        loss_t = F.cross_entropy(logits.t(), targets)
        loss_total += (loss_i + loss_t) / 2

    return loss_total / batch_size