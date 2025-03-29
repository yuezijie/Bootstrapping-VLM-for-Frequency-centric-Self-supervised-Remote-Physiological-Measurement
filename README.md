# Bootstrapping-VLM-for-Frequency-centric-Self-supervised-Remote-Physiological-Measurement

This is the code base for paper [Bootstrapping Vision-language Models for Frequency-centric Self-supervised Remote Physiological Measurement](https://link.springer.com/article/10.1007/s11263-025-02388-5) (IJCV, 2025)

Abstract:

Facial video-based remote physiological measurement is a promising research area for detecting human vital signs (e.g., heart rate, respiration frequency) in a non-contact way. Conventional approaches are mostly supervised learning, requiring extensive collections of facial videos and synchronously recorded photoplethysmography (PPG) signals. To tackle it, self-supervised learning has recently gained attentions; due to the lack of ground truth PPG signals, its performance is however limited. In this paper, we propose a novel frequency-centric self-supervised framework that successfully integrates the popular vision-language models (VLMs) into the remote physiological measurement task. Given a facial video, we first augment its positive and negative video samples with varying rPPG signal frequencies. Next, we introduce a frequency-oriented vision-text pair generation method by carefully creating contrastive spatio-temporal maps from positive and negative samples and designing proper text prompts to describe their relative ratios of signal frequencies. A pre-trained VLM is employed to extract features for these formed vision-text pairs and estimate rPPG signals thereafter. We develop a series of frequency-related generative and contrastive learning mechanisms to optimize the VLM, including the text-guided visual reconstruction task, the vision-text contrastive learning task, and the frequency contrastive and ranking task. Overall, our method for the first time adapts VLMs to digest and align the frequency-related knowledge in vision and text modalities. Extensive experiments on four benchmark datasets demonstrate that it significantly outperforms state of the art self-supervised methods. 

# Install and compile the prerequisites
- Python 3.8
- PyTorch >= 1.8
- NVIDIA GPU + CUDA
- Python packages: numpy,opencv-python
# Pretrained model
Please download the [pretrained model](https://drive.google.com/file/d/1CtAJeG-KHWbLbPd2Cl0OXxzRJdg3o4Wu/view?usp=sharing), and put it under model/

# Main experiment

1. Modify the data path in main.py to your own data path.
2. Run [python main.py].


# Citation
```
@article{yue2025bootstrapping,
  title={Bootstrapping Vision-Language Models for Frequency-Centric Self-Supervised Remote Physiological Measurement},
  author={Yue, Zijie and Shi, Miaojing and Wang, Hanli and Ding, Shuai and Chen, Qijun and Yang, Shanlin},
  journal={International Journal of Computer Vision},
  year={2025},
  publisher={Springer}
}
```
