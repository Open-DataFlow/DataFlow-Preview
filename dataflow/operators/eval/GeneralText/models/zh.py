from transformers import BertModel, BertConfig, PreTrainedModel, AutoTokenizer
import torch.nn as nn
import torch

class BertForRegression(PreTrainedModel):
    config_class = BertConfig

    def __init__(self, config):
        super().__init__(config)

        self.bert = BertModel(config)
        self.regression = nn.Sequential(
            nn.Linear(config.hidden_size, 256),
            nn.ReLU(),
            nn.Linear(256, 1)
        )

        self.post_init()

    def forward(self, input_ids=None, attention_mask=None, token_type_ids=None, **kwargs):
        outputs = self.bert(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids
        )
        pooled = outputs.pooler_output  # [batch_size, hidden_size]
        score = self.regression(pooled)  # [batch_size, 1]
        return pooled, score


model = "zks2856/PairQual-Scorer-zh"
config = BertConfig.from_pretrained(model)
model = BertForRegression.from_pretrained(model, config=config, trust_remote_code=True)
tokenizer = AutoTokenizer.from_pretrained(model, trust_remote_code=True)


