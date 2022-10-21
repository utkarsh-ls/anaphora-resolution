import torch
import torch.nn as nn
from transformers import BertModel, AutoModel
import configs


class MentionModel(nn.Module):
    def __init__(self) -> None:
        super(MentionModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            configs.transformer_model
        )
        self.classifier = nn.Linear(in_features=768, out_features=1)

    def forward(self, ids, mask):

        with torch.no_grad():
            word_embed = self.bert_model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True
            ).last_hidden_state
            # x has shape N , L , 768

        mention_logits = self.classifier(word_embed)

        # mention_logits has shape N, L, 1
        return mention_logits.squeeze(-1), word_embed


class PairScoreModel(nn.Module):
    def __init__(self) -> None:
        super(PairScoreModel, self).__init__()
        self.relu = nn.Tanh()
        self.fc = nn.Linear(in_features=2 * 768, out_features=1)
        self.hidden1 = nn.Linear(in_features=2 * 768, out_features=1024)
        self.hidden2 = nn.Linear(in_features=1024, out_features=1024)
        self.hidden3 = nn.Linear(in_features=1024, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word1_embeds, word2_embeds):
        pair_embed = torch.cat([word1_embeds, word2_embeds], dim=-1)
        assert pair_embed.size() == (word1_embeds.size(0), 2 * word1_embeds.size(1))
        scores = self.fc(pair_embed)
        # scores = self.hidden1(pair_embed)
        # scores = self.relu(scores)
        # scores = self.hidden2(scores)
        # scores = self.relu(scores)
        # scores = self.hidden3(scores)
        # scores = self.sigmoid(scores)
        return scores.squeeze(-1)
