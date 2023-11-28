import torch
import torch.nn as nn
from models.alphafold2 import Evoformer


class LayerNorm(nn.Module):
    "Construct a layernorm module (See citation for details)."

    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class RelationalAttentionBertClassifier(nn.Module):
    def __init__(self, bert, opt):
        super().__init__()
        self.opt = opt
        self.bert = bert
        self.attention_heads = opt.attention_heads
        self.hidden_dim = opt.bert_dim // 2
        self.bert_dim = opt.bert_dim
        self.bert_layernorm = LayerNorm(opt.bert_dim)
        self.bert_drop = nn.Dropout(opt.bert_dropout)
        self.pooled_drop = nn.Dropout(opt.bert_dropout)
        self.edge_emb = torch.load(opt.amr_edge_pt) \
            if opt.edge == "normal" or opt.edge == "same" else nn.Embedding(56000, 1024)
        self.edge_emb_layernorm = nn.LayerNorm(opt.amr_edge_dim)
        self.edge_emb_drop = nn.Dropout(opt.edge_dropout)
        self.edge_dim_change = nn.Linear(opt.amr_edge_dim, self.hidden_dim, bias=False)
        self.evoformer = Evoformer(depth=opt.num_layers,
                                   dim=opt.bert_dim,  # 输入维度
                                   pair_dim=self.hidden_dim,
                                   seq_len=opt.max_length,
                                   heads=opt.attention_heads,
                                   dim_head=opt.dim_heads,  # 每个头的隐藏维度
                                   attn_dropout=opt.attn_dropout,
                                   ff_dropout=0.)
        self.final_dropout = nn.Dropout(opt.final_dropout)
        self.final_layernorm = LayerNorm(opt.bert_dim)
        if opt.feature_type == "1+2+A":
            self.classifier = nn.Linear(opt.bert_dim * 3, opt.polarities_dim)
        elif opt.feature_type == "1+2+A/3":
            self.classifier = nn.Sequential(
                nn.Linear(opt.bert_dim * 3, opt.bert_dim),
                nn.ReLU(),
                nn.Linear(opt.bert_dim, opt.polarities_dim)
            )
        elif opt.feature_type == "1+A" or opt.feature_type == "2+A":
            self.classifier = nn.Linear(opt.bert_dim * 2, opt.polarities_dim)
        else:
            raise Exception("Wrong feature type!")

    def forward(self, inputs):
        text_bert_indices, bert_segments_ids, attention_mask, asp_start, asp_end, adj_dep, edge_adj, src_mask, aspect_mask = inputs

        sequence_output, pooled_output = self.bert(text_bert_indices, attention_mask=attention_mask, token_type_ids=bert_segments_ids)
        sequence_output = self.bert_layernorm(sequence_output)
        token_inputs = self.bert_drop(sequence_output)
        pooled_output = self.pooled_drop(pooled_output)

        batch_size, max_length, _ = edge_adj.size()
        edge_adj = self.edge_emb(edge_adj)
        edge_adj = self.edge_emb_layernorm(edge_adj)
        edge_adj = self.edge_emb_drop(edge_adj)
        edge_adj = self.edge_dim_change(edge_adj)

        x = edge_adj  # b, i, j, d
        m = token_inputs.unsqueeze(1)  # b, n, d -> b, 1, n, d
        token_mask = src_mask.unsqueeze(1)  # b, n -> b, 1, n
        edge_mask = src_mask.unsqueeze(1) * src_mask.unsqueeze(2)  # b, i * b, j -> b, i, j
        edges, tokens = self.evoformer(x=x, m=m, mask=edge_mask.bool(), msa_mask=token_mask.bool())
        tokens = tokens.squeeze(1)

        asp_wn = aspect_mask.sum(dim=1).unsqueeze(-1)
        aspect_mask_repeat = aspect_mask.unsqueeze(-1)
        outputs1 = (tokens * aspect_mask_repeat).sum(dim=1) / asp_wn
        outputs2 = (token_inputs * aspect_mask_repeat).sum(dim=1) / asp_wn

        if self.opt.feature_type in ["1+2+A", "1+2+A/3"]:
            final_outputs = torch.cat((outputs1, outputs2, pooled_output), dim=-1)
        elif self.opt.feature_type == "2+A":
            final_outputs = torch.cat((outputs2, pooled_output), dim=-1)
        elif self.opt.feature_type == "1+A":
            final_outputs = torch.cat((outputs1, pooled_output), dim=-1)
        else:
            raise Exception("Wrong feature type!")

        logits = self.classifier(final_outputs)

        return logits, None