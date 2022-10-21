import torch
import torch.nn as nn
from transformers import BertModel, AutoModel


class TorchGRUIntent(nn.Module):
    def __init__(self, hidden_size, seq_len) -> None:
        super(TorchGRUIntent, self).__init__()

        self.bert_model = AutoModel.from_pretrained(
            # "google/muril-base-cased"
            "bert-base-uncased"
        )

        self.GRU = nn.GRU(
            input_size=768,
            hidden_size=hidden_size,
            num_layers=2,
            bidirectional=True,
            batch_first=True,
        )
        self.classifier = nn.Linear(in_features=2 * hidden_size, out_features=seq_len)
        # self.query = nn.Parameter(data = torch.empty(1, 2 * hidden_size))
        # nn.init.uniform_(self.query, -1, 1)
        # self.mha = nn.MultiheadAttention(embed_dim=2 * hidden_size ,num_heads=2, batch_first=True)

    def forward(self, ids, mask):

        with torch.no_grad():
            x = self.bert_model(
                input_ids=ids, attention_mask=mask, output_hidden_states=True
            ).last_hidden_state
            # x has shape N , L , 768

            # x = torch.transpose(x, 0 , 1)
            # # x has shape L, N , 768

        out, h = self.GRU(x)
        # out has size  N(batch), L(seq_len), D∗H_out(hidden)
        # h has shape D*num_layers, N, Hout(hidden size)​
        # h = torch.transpose(h, 0, 1)
        # # h has shape N, D*num_layers, Hout(hidden size)​
        # h = h.reshape(h.size()[0], -1)
        # outputs = self.classifier(torch.mean(out, dim=0))
        # batched_query = torch.stack([self.query] * out.size()[0])
        # # N, 1, DHout
        # # print(batched_query.size())
        # # print(out.size())
        # # print((mask==False).size())
        # attn_out, _ = self.mha(query=batched_query, key=out, value=out, key_padding_mask=(mask==False))
        outputs = self.classifier(out)

        # outputs has shape N, L, vocab
        return outputs


class MentionModel(nn.Module):
    def __init__(self) -> None:
        super(MentionModel, self).__init__()

        self.bert_model = BertModel.from_pretrained(
            # "google/muril-base-cased"
            "bert-base-uncased"
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
        self.relu = nn.ReLU()
        self.hidden1 = nn.Linear(in_features=2 * 768, out_features=1024)
        self.hidden2 = nn.Linear(in_features=1024, out_features=1024)
        self.hidden3 = nn.Linear(in_features=1024, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, word1_embeds, word2_embeds):
        pair_embed = torch.cat([word1_embeds, word2_embeds], dim=-1)
        assert pair_embed.size() == (word1_embeds.size(0), 2*word1_embeds.size(1))
        scores = self.hidden1(torch.concat((word1_embeds, word2_embeds), dim=-1))
        scores = self.relu(scores)
        scores = self.hidden2(scores)
        scores = self.relu(scores)
        scores = self.hidden3(scores)
        # scores = self.sigmoid(scores)
        return scores.squeeze(-1)
