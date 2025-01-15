import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset
import warnings

class EmbedPhenoDataset(Dataset):
    def __init__(self, data, genes, control_gene='negative', n_gene_per_pert=2, pheno_shuffle=True):
        super().__init__()
        self.data = data

        self.genes = genes
        self.control_gene = control_gene
        self.gene_to_idx = {g: i for i, g in enumerate(self.genes)}
        self.idx_to_gene = {i: g for i, g in enumerate(self.genes)}

        self.pheno_shuffle = pheno_shuffle
        
        all_gene_idx = []
        for i in range(n_gene_per_pert):
            gene_i = 'gene' + str(i+1)
            gene_names = self.data[gene_i].values
            gene_idx = [self.gene_to_idx[g] for g in gene_names]
            gene_idx = torch.tensor(gene_idx)
            all_gene_idx.append(gene_idx)

        self.gene_idx = torch.stack(all_gene_idx, dim=-1)
        self.targets = torch.tensor(data['comb_score'].values)
        
        pheno_cols = [f'g{i}_score' for i in range(1, n_gene_per_pert + 1)]
        self.phenos = torch.tensor(data[pheno_cols].values)

        if self.pheno_shuffle:
            self.generator = torch.Generator().manual_seed(0)
            row_idx = torch.argsort(torch.rand(self.phenos.shape, generator=self.generator), dim=-1)
            self.phenos = self.phenos[torch.arange(len(self.phenos)).unsqueeze(-1), row_idx]
        
        if data.columns.str.contains('rank').any():
            rank_var = data.columns[data.columns.str.contains('rank')]
            if len(rank_var) > 1:
                raise ValueError('Multiple rank variables found in data')
            self.rank =  torch.tensor(data[rank_var].values.squeeze())

    def __len__(self):
        return self.data.shape[0]
    
    def __getitem__(self, i):
        gene_idx = self.gene_idx[i]
        target = self.targets[i]
        phenos = self.phenos[i]
        if hasattr(self, 'rank'):
            rank = self.rank[i]
            return gene_idx, target, phenos, rank
        else:
            return gene_idx, target, phenos

class NAIADEmbedPheno(nn.Module):
    def __init__(self, n_genes, 
                 d_embed = 16, 
                 d_pheno_hid = 512, 
                 d_out = 1, 
                 p_dropout = 0.1, 
                 model_type = 'both', 
                 embed_model = 'MLP',
                 train_embed = True, 
                 n_gene_per_pert = 2):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_embed = d_embed
        self.d_out = d_out
        self.p_dropout = p_dropout
        self.train_embed = train_embed
        self.d_embed = d_embed
        self.d_pheno_hid = d_pheno_hid
        self.model_type = model_type.lower()
        self.n_gene_per_pert = n_gene_per_pert
        self.embed_model = embed_model
        n_heads = 1

        if self.embed_model not in ['MLP', 'multihead', 'transformer', 'transformer-cls']:
            raise ValueError('embed_model must be one of the following: "MLP", "multihead", "transformer", "transformer-cls"')


        if self.model_type not in ['pheno', 'embed', 'both']:
            raise ValueError('model_type must be one of the following: "pheno", "embed", "both"')  

        if self.embed_model == 'MLP':
            self.embedding = nn.Embedding(self.n_genes, self.d_embed)
            self.embedding.weight.requires_grad = train_embed
            self.embedding_ffn = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                                nn.Dropout(self.p_dropout),
                                                nn.ReLU(),
                                                nn.Linear(self.d_embed, self.d_embed))
        else:
            n_heads = int(np.log2(d_embed))
            n_heads = max(2, n_heads)
            self.d_embed = self.d_embed * n_heads
            self.embedding = nn.Embedding(self.n_genes, self.d_embed)
            self.embedding.weight.requires_grad = train_embed

            if self.embed_model == 'multihead':
                self.embedding_ffn = TwoLayerMultiHeadAttention(d_embed=self.d_embed, n_heads=n_heads, p_dropout=self.p_dropout)
            elif self.embed_model == 'transformer':
                self.embedding_ffn = TwoLayerTransformer(d_embed=self.d_embed, n_heads=n_heads, p_dropout=self.p_dropout, use_cls_token=False)
            elif self.embed_model == 'transformer-cls':
                self.embedding_ffn = TwoLayerTransformer(d_embed=self.d_embed, n_heads=n_heads, p_dropout=self.p_dropout, use_cls_token=True)

        
        self.embedding_comb = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                                nn.Dropout(self.p_dropout),
                                                nn.ReLU(),
                                                nn.Linear(self.d_embed, 1))
            
        self.pheno_ffn = nn.Sequential(nn.Linear(self.n_gene_per_pert, d_pheno_hid, bias=False),
                                        nn.Dropout(self.p_dropout),
                                        nn.GELU(),
                                        nn.Linear(self.d_pheno_hid, self.d_pheno_hid, bias=False),
                                        nn.Dropout(self.p_dropout),
                                        nn.GELU(),
                                        nn.Linear(self.d_pheno_hid, 1))
        
        
        self.rank_ffn = nn.Sequential(nn.Linear(self.d_embed + 1, self.d_embed // n_heads),
                                        nn.Dropout(self.p_dropout),
                                        nn.GELU(),
                                        nn.Linear(self.d_embed // n_heads, 1))
         
        
    def extract_embedding(self, x, comb=True):
        x = self.embedding(x)                    
        if comb:
            x = x.reshape(x.shape[0]*self.n_gene_per_pert, x.shape[2])
            x = self.embedding_ffn(x)
            x = x.reshape(-1, self.n_gene_per_pert, x.shape[1])
            x = torch.sum(x, axis=1)
        return x

    def forward_embed(self, x):
        x = self.extract_embedding(x, comb=True)     
        x = self.embedding_comb(x)
        return x
    
    def forward_embed_transformer(self, x):
        x = self.extract_embedding(x, comb=False)     
        x = x.permute(1, 0, 2)
        # enter transformer block
        x, _ = self.embedding_ffn(x)
        x = self.embedding_comb(x)
        return x
    
    def get_attention_weights(self, x):
        x = self.extract_embedding(x, comb=False)     
        x = x.permute(1, 0, 2)
        if self.embed_model in ['transformer','transformer-cls']:
            _, attn_weights = self.embedding_ffn(x)
            return attn_weights
        else:
            warnings.warn('Attention weights only available for transformer models')
            return None


    def forward_rank_predictor(self, x, phenos):

        x = self.extract_embedding(x, comb=False)     
        x = x.permute(1, 0, 2)
        # enter transformer block
        _, _, x = self.embedding_ffn(x, return_all_embeddings=True)
        # remove cls token
        x = x[1:, :, :].mean(axis=0)
        phenos = self.pheno_ffn(phenos)
        x = torch.cat([phenos, x], axis=1)
        # sampling score
        x = self.rank_ffn(x)
        return x

    def forward(self, x, phenos=None, return_intermediate=False):
        if self.model_type == 'pheno':
            phenos = self.pheno_ffn(phenos)
            return phenos.squeeze()

        if self.model_type == 'embed':
            x = self.forward_embed(x)
            return x.squeeze()

        elif self.model_type == 'both':
            if self.embed_model in ['transformer', 'transformer-cls']:
                x = self.forward_embed_transformer(x)
            else:
                x = self.forward_embed(x)
            phenos = self.pheno_ffn(phenos)
            out_x = x + phenos
            if return_intermediate:
                return out_x.squeeze(), x.squeeze(), phenos.squeeze()
            return out_x.squeeze()
                
        return None

class BilinearMLPEmbedPheno(NAIADEmbedPheno):
    def __init__(self, n_genes, 
                 d_embed = 16, 
                 d_pheno_hid = 512, 
                 d_out = 1, 
                 p_dropout = 0.1, 
                 model_type = 'both', 
                 train_embed = True, 
                 n_gene_per_pert = 2):
        super().__init__(n_genes,
                         d_embed,
                         d_pheno_hid,
                         d_out,
                         p_dropout,
                         model_type,
                         train_embed,
                         n_gene_per_pert)
                
        self.bilinear_weights = nn.Parameter(
            1 / 100 * torch.randn((self.d_embed, self.d_embed, self.d_embed))
            + torch.cat([torch.eye(self.d_embed)[None, :, :]] * self.d_embed, dim=0)
        )
        self.bilinear_offsets = nn.Parameter(1 / 100 * torch.randn((self.d_embed)))

        self.allow_neg_eigval = False
        if self.allow_neg_eigval:
            self.bilinear_diag = nn.Parameter(1 / 100 * torch.randn((self.d_embed, self.d_embed)) + 1)

    def forward_embed(self, x):
        x = self.embedding(x)

        x = x.reshape(x.shape[0]*self.n_gene_per_pert, x.shape[2])
        x = self.embedding_ffn(x)
        x = x.reshape(-1, self.n_gene_per_pert, x.shape[1])

        x0 = self.bilinear_weights.matmul(x[:, 0, :].T).T # TODO: Check these dimensions
        x1 = self.bilinear_weights.matmul(x[:, 1, :].T).T

        if self.allow_neg_eigval:
            x1 *= self.bilinear_diag

        x1 = x1.permute(0, 2, 1)
        x = (x0 * x1).sum(1)
        
        x += self.bilinear_offsets
        x = self.embedding_comb(x)

        return x
    
class TwoLayerMultiHeadAttention(nn.Module):
    def __init__(self, d_embed, n_heads, p_dropout):
        super(TwoLayerMultiHeadAttention, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.p_dropout = p_dropout


        self.attention_1 = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, dropout=p_dropout)
        self.ffn_1 = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_embed, d_embed),
        )
        self.norm_1 = nn.LayerNorm(d_embed)

        self.attention_2 = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, dropout=p_dropout)
        self.ffn_2 = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.ReLU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_embed, d_embed),
        )
        self.norm_2 = nn.LayerNorm(d_embed)

    def forward(self, x):
        # First transformer block
        attn_output_1, _ = self.attention_1(x, x, x)
        x = x + attn_output_1
        x = self.norm_1(x)
        x = x + self.ffn_1(x)

        # Second transformer block
        attn_output_2, _ = self.attention_2(x, x, x)
        x = x + attn_output_2
        x = self.norm_2(x)
        x = x + self.ffn_2(x) 

        return x
    

class TwoLayerTransformer(nn.Module):
    def __init__(self, d_embed, n_heads, p_dropout, use_cls_token=False):
        super(TwoLayerTransformer, self).__init__()
        self.d_embed = d_embed
        self.n_heads = n_heads
        self.p_dropout = p_dropout
        self.use_cls_token = use_cls_token
        if self.use_cls_token:
            self.cls_token = nn.Parameter(torch.randn(1, 1, d_embed))

        self.attention_1 = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, dropout=p_dropout)
        self.ffn_1 = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_embed, d_embed),
        )
        self.norm_1 = nn.LayerNorm(d_embed)

        self.attention_2 = nn.MultiheadAttention(embed_dim=d_embed, num_heads=n_heads, dropout=p_dropout)
        self.ffn_2 = nn.Sequential(
            nn.Linear(d_embed, d_embed),
            nn.GELU(),
            nn.Dropout(p_dropout),
            nn.Linear(d_embed, d_embed),
        )
        self.norm_2 = nn.LayerNorm(d_embed)

    def forward(self, x, return_all_embeddings=False):
        # First transformer block
        # input dimensions: seq_len, batch_size, embed_dim
        batch_size = x.shape[1]
        if self.use_cls_token:
            cls_token_expanded = self.cls_token.expand(-1, batch_size, -1)  # [1, batch_size, embed_dim]
            x = torch.cat([cls_token_expanded, x], dim=0) 
        attn_output_1, attn_weights_1 = self.attention_1(x, x, x)
        x = x + attn_output_1
        x = self.norm_1(x)
        x = x + self.ffn_1(x)

        # Second transformer block
        attn_output_2, attn_weights_2 = self.attention_2(x, x, x)
        x = x + attn_output_2
        x = self.norm_2(x)
        x = x + self.ffn_2(x) 

        if self.use_cls_token:
           output = x[0, :, :]
        else:
            output = torch.mean(x, dim=0)

        attn_output = torch.stack((attn_weights_1, attn_weights_2), dim=0)

        if return_all_embeddings:
            return output, attn_output, x
        else:
            return output, attn_output
    
 
    

    

class VarModel(nn.Module):
    def __init__(self, base_model):
        super().__init__()
        self.pred_model = base_model
        self.var_model = copy.deepcopy(base_model) 
        
    def forward(self, x, phenos=None, return_intermediate=False):
        preds = self.pred_model(x, phenos, return_intermediate)
        var = self.var_model(x, phenos, return_intermediate)
        return tuple(*preds, *var)
        
def model_loader(n_genes, 
                 d_embed = 16, 
                 d_pheno_hid = 512, 
                 d_out = 1, 
                 p_dropout = 0.1, 
                 model_type = 'both', 
                 embed_model = 'MLP',
                 train_embed = True, 
                 n_gene_per_pert = 2,
                 bilinear_comb = False, 
                 var_pred = False):

    model_type = model_type.lower()
    
    if model_type not in ['pheno', 'embed', 'both']:
        raise ValueError('model_type must be one of the following: "pheno", "embed", "both"')
    
    if not bilinear_comb:
        model = NAIADEmbedPheno(n_genes,
                              d_embed,
                              d_pheno_hid,
                              d_out,
                              p_dropout,
                              model_type,
                              embed_model,
                              train_embed,
                              n_gene_per_pert)
    else:
        model = BilinearMLPEmbedPheno(n_genes,
                                        d_embed,
                                        d_pheno_hid,
                                        d_out,
                                        p_dropout,
                                        model_type,
                                        train_embed,
                                        n_gene_per_pert)

    if var_pred:
        model = VarModel(model)

    return model
    
class RecoverModel(nn.Module):
    def __init__(self, n_genes, 
                 d_embed = 16, 
                 d_out = 1, 
                 p_dropout = 0.1, 
                 n_gene_per_pert = 2):
        super().__init__()
        
        self.n_genes = n_genes
        self.d_embed = d_embed
        self.d_out = d_out
        self.p_dropout = p_dropout
        self.n_gene_per_pert = n_gene_per_pert

        self.embedding = nn.Embedding(self.n_genes, self.d_embed)

        self.embedding_ffn = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                            nn.Dropout(self.p_dropout),
                                            nn.ReLU(),
                                            nn.Linear(self.d_embed, self.d_embed),
                                            nn.Dropout(self.p_dropout),
                                            nn.ReLU(), 
                                            nn.Linear(self.d_embed, self.d_embed))
        
        # Code adapted from RECOVER package: https://github.com/RECOVERcoalition/Recover/blob/master/recover/models/predictors.py
        self.bilinear_weights = nn.Parameter(
            1 / 100 * torch.randn((self.d_embed, self.d_embed, self.d_embed))
            + torch.cat([torch.eye(self.d_embed)[None, :, :]] * self.d_embed, dim=0)
        )
        self.bilinear_offsets = nn.Parameter(1 / 100 * torch.randn((self.d_embed)))

        self.allow_neg_eigval = True
        if self.allow_neg_eigval:
            self.bilinear_diag = nn.Parameter(1 / 100 * torch.randn((self.d_embed, self.d_embed)) + 1)

        self.embedding_comb = nn.Sequential(nn.Linear(self.d_embed, self.d_embed),
                                            nn.Dropout(self.p_dropout),
                                            nn.ReLU(),
                                            nn.Linear(self.d_embed, 1))

    def forward(self, x, phenos=None):
        x = self.embedding(x)
        
        x = x.reshape(x.shape[0]*self.n_gene_per_pert, x.shape[2])
        x = self.embedding_ffn(x)
        x = x.reshape(-1, self.n_gene_per_pert, x.shape[1])

        ## Code below follows RECOVER repo but I suspect it isn't correct...
        # I think the last transpose should be an axis permutation to preserve relative order of dimensions
        x0 = self.bilinear_weights.matmul(x[:, 0, :].T).T
        x1 = self.bilinear_weights.matmul(x[:, 1, :].T).T

        # x0 = self.bilinear_weights.matmul(x[:, 0, :].T).permute(2, 0, 1)
        # x1 = self.bilinear_weights.matmul(x[:, 1, :].T).permute(2, 0, 1)

        if self.allow_neg_eigval:
            x1 *= self.bilinear_diag

        # I think this permutation isn't correct to calculate the inner product between two matrices
        x1 = x1.permute(0, 2, 1)
        x = (x0 * x1).sum(1)
        
        x += self.bilinear_offsets

        x = self.embedding_comb(x)
        x = torch.sum(phenos, axis=1) + x.squeeze()
        return x