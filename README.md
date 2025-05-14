# Image Retrieval by Caption with GUI

## 1. Introduction

Retrieving images using textual queries is a fundamental task in computer vision and natural language processing. In this project, we explore one way of solving this task using deep learning methods. The project culminates in an interactive application that allows users to explore the model's behavior.

The core idea is to generate vector embeddings for both images and their captions using two separate encoders. These encoders are trained so that embeddings of corresponding image‑caption pairs are close in the shared vector space. At inference time, given a textual prompt, the text encoder generates an embedding, and the model retrieves the most semantically similar images.

The interactive application is implemented with **Streamlit**, located in the `app` folder. It can be launched using:

```
streamlit run main.py
```

---

## 2. Dataset

The dataset used in this project is **Flickr8K**, which contains 8,091 images, each annotated with five unique captions. The dataset is split into training, validation, and testing sets in an 80/10/10 ratio.

Each subset is represented as a PyTorch dataset. The images undergo the following preprocessing steps:
- Resize to 256×256
- Center-crop to 224×224
- Normalize using ImageNet statistics

All preprocessed images are stored in memory as tensors. For training, each call to `__getitem__` returns one image tensor and one randomly selected caption (of the five) corresponding to that image index, which improves model generalization.

---

## 3. Model Setup and Training

### 3.1 Model Architecture

#### 3.1.1 Image Encoder

The image encoder is a pre-trained **EfficientNet‑B0** model with its classification head removed. The output feature vector is passed through a linear projection layer to obtain a 256‑dimensional embedding.

#### 3.1.2 Text Encoder

The text encoder is a pre-trained **BERT** model. Its pooled output is similarly mapped into a 256‑dimensional embedding space via a linear projection layer.

### 3.2 Training Process

The model is trained using the **NT‑Xent** (Normalized Temperature‑scaled Cross Entropy) loss, with a temperature hyperparameter of **0.1**.

#### Training Schedule

1. **Stage 1** – Freeze both backbones; train only the projection layers for 20 epochs at learning rate **0.01**.  
2. **Stage 2** – Continue training projection layers for 10 more epochs at learning rate **0.001**.  
3. **Stage 3** – Unfreeze all backbone parameters; train entire network for 20 epochs at learning rate **0.00001**.  
4. **Stage 4** – Additional 20 epochs showed diminishing returns (overfitting).  

The best checkpoint was saved after **epoch 50**. Experiments with partial unfreezing of the backbones did not improve performance.

For full training logs and code, see `notebooks/model_training.ipynb`.

---

## 4. Testing and Evaluating Performance

### 4.1 Recall@K

Recall@K measures the fraction of relevant images found in the top **K** retrieved results. Higher Recall@K indicates better coverage of relevant items.

### 4.2 Precision@K

Precision@K measures the proportion of the top **K** retrieved results that are relevant. Higher Precision@K indicates fewer irrelevant items in the top K.

### 4.3 Results

The model’s performance on both metrics is below acceptable thresholds. The small gap between training and testing scores indicates limited overfitting. Qualitative testing via the GUI also showed that retrieved images often do not align semantically with the query.

Evaluation plots for Recall@K and Precision@K are available in `notebooks/model_evaluation.ipynb`.

---

## 5. Conclusion

Although the current model’s performance is suboptimal, there are several clear avenues for improvement:

- **Advanced Backbones**: Fine‑tune larger pre-trained models (e.g., EfficientNet‑B3, RoBERTa).  
- **Data Augmentation**: Employ random crops, color jitter, and other augmentations to enrich training.  
- **Loss Variants**: Explore margin‑based or contrastive losses beyond NT‑Xent.  
- **User‑uploaded Images**: Extend the GUI to accept user images and generate candidate captions.  

Implementing these ideas could substantially improve both quantitative metrics and user experience.

---

## 6. Citations

- Devlin, J., Chang, M. W., Lee, K., & Toutanova, K. (2019). BERT: Pre‑training of Deep Bidirectional Transformers for Language Understanding. *NAACL‑HLT*, 4171–4186.  
- Ågren, W. (2022). The NT‑Xent Loss Upper Bound. *arXiv preprint arXiv:2205.03169*.  
- Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. *ICML*, 6105–6114.  
- Hodosh, M., Young, P., & Hockenmaier, J. (2013). Framing Image Description as a Ranking Task: Data, Models and Evaluation Metrics. *Journal of Artificial Intelligence Research*, 47, 853–899.
