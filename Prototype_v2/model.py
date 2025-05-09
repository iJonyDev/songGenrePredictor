from transformers import BertModel
import torch.nn as nn

class GenreClassifier(nn.Module):
    def __init__(self, num_genres, dropout_prob=0.3): # Añadir dropout_prob
        super(GenreClassifier, self).__init__()
        self.bert = BertModel.from_pretrained('bert-base-uncased')
        # --- Añadir capa Dropout ---
        self.dropout = nn.Dropout(dropout_prob)
        # --------------------------
        self.classifier = nn.Linear(self.bert.config.hidden_size, num_genres)

    def forward(self, input_ids, attention_mask):
        # Pass inputs through BERT
        outputs = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        # Use the [CLS] token representation for classification
        cls_output = outputs.last_hidden_state[:, 0, :]
        # --- Aplicar Dropout ---
        dropped_output = self.dropout(cls_output)
        # -----------------------
        # Pass through the classifier
        return self.classifier(dropped_output) # Usar salida con dropout
