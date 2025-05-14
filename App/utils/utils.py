import torch
import torch.nn as nn
import numpy as np

from transformers import BertTokenizer, BertModel
from torch.nn.utils.rnn import pad_sequence
from sklearn.metrics.pairwise import cosine_similarity


class TextEncoder(nn.Module):
    def __init__(self, embed_size):
        super().__init__()
        self.tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
        self.bert = BertModel.from_pretrained("bert-base-uncased")
        self.fc1 = nn.Linear(768, embed_size)

    def forward(self, captions):
        tokens = self.tokenizer(captions, padding=True, truncation=True, return_tensors="pt")
        input_ids = tokens["input_ids"]
        attention_mask = tokens["attention_mask"]

        outputs = self.bert(input_ids, attention_mask=attention_mask)
        last_hidden_state = outputs.last_hidden_state
        embeds = self.fc1(last_hidden_state[:, 0, :])
        return embeds
    

def retrieve_images(model, captions_df, query, k):
    query_embed = model(query).detach().numpy().reshape(1,-1)
    embeddings = np.vstack([arr for arr in captions_df["embedding"]])
    similarities = cosine_similarity(X=query_embed, Y=embeddings)
    captions_df["similaritiy"] = similarities[0]

    top_k = captions_df.sort_values(by="similaritiy", ascending=False).head(k)

    return top_k["image_name"]