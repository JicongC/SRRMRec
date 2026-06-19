# coding: utf-8
import os
import torch
import numpy as np
import torch.nn as nn
import scipy.sparse as sp
import torch.nn.functional as F
from common.abstract_recommender import GeneralRecommender


class SRRMRec(GeneralRecommender):
    def __init__(self, config, dataset):
        super(SRRMRec, self).__init__(config, dataset)
        self.embedding_dim = config['embedding_size']
        self.feat_embed_dim = config['feat_embed_dim']
        self.n_layers = config['n_mm_layers']
        self.n_ui_layers = config['n_ui_layers']
        self.reg_weight = config['reg_weight']
        self.ssl_temp = config['ssl_temp']
        self.knn_k = config['knn_k']
        self.mm_image_weight = config['mm_image_weight']
        self.ssl_weight = config['ssl_weight']
        self.beta = config['beta']
        self.reliability_metric = config['reliability_metric']
        self.n_nodes = self.n_users + self.n_items
        self.interaction_matrix = dataset.inter_matrix(form='coo').astype(np.float32)
        self.norm_adj = self.get_norm_adj_mat().to(self.device)
        self.user_text = nn.Embedding(self.n_users, self.embedding_dim)
        self.user_image = nn.Embedding(self.n_users, self.embedding_dim)
        nn.init.xavier_uniform_(self.user_image.weight)
        nn.init.xavier_uniform_(self.user_text.weight)
        if self.v_feat is not None:
            self.image_embedding = nn.Embedding.from_pretrained(self.v_feat, freeze=False)
            self.image_trs = nn.Linear(self.v_feat.shape[1], self.feat_embed_dim)
        if self.t_feat is not None:
            self.text_embedding = nn.Embedding.from_pretrained(self.t_feat, freeze=False)
            self.text_trs = nn.Linear(self.t_feat.shape[1], self.feat_embed_dim)
        dataset_path = os.path.abspath(config['data_path'] + config['dataset'])
        file_name = f'mm_adj_b{self.beta}_{self.reliability_metric}.pt'
        mm_adj_file = os.path.join(dataset_path, file_name)
        if os.path.exists(mm_adj_file):
            self.mm_adj = torch.load(mm_adj_file)
        else:
            if self.v_feat is not None:
                _, image_adj = self.get_knn_adj_mat(self.image_embedding.weight.detach())
                self.mm_adj = image_adj
            if self.t_feat is not None:
                _, text_adj = self.get_knn_adj_mat(self.text_embedding.weight.detach())
                self.mm_adj = text_adj
            if self.v_feat is not None and self.t_feat is not None:
                self.mm_adj = self.mm_image_weight * image_adj + (1.0 - self.mm_image_weight) * text_adj
                del text_adj, image_adj
                self.mm_adj = self.structural_reliability_reweight(self.mm_adj, beta=self.beta)
            torch.save(self.mm_adj, mm_adj_file)

    def get_knn_adj_mat(self, mm_embeddings):
        context_norm = mm_embeddings.div(torch.norm(mm_embeddings, p=2, dim=-1, keepdim=True))
        sim = torch.mm(context_norm, context_norm.transpose(1, 0))
        _, knn_ind = torch.topk(sim, self.knn_k, dim=-1)
        adj_size = sim.size()
        del sim
        indices0 = torch.arange(knn_ind.shape[0]).to(self.device).unsqueeze(1).expand(-1, self.knn_k)
        indices = torch.stack((torch.flatten(indices0), torch.flatten(knn_ind)), 0)

        return indices, self.compute_normalized_laplacian(indices, adj_size)

    def compute_normalized_laplacian(self, indices, adj_size):
        adj = torch.sparse.FloatTensor(indices, torch.ones_like(indices[0]), adj_size)
        row_sum = 1e-7 + torch.sparse.sum(adj, -1).to_dense()
        r_inv_sqrt = torch.pow(row_sum, -0.5)
        rows_inv_sqrt = r_inv_sqrt[indices[0]]
        cols_inv_sqrt = r_inv_sqrt[indices[1]]
        values = rows_inv_sqrt * cols_inv_sqrt

        return torch.sparse.FloatTensor(indices, values, adj_size)

    def get_norm_adj_mat(self):
        A = sp.dok_matrix((self.n_nodes, self.n_nodes), dtype=np.float32)
        inter_M = self.interaction_matrix
        inter_M_t = self.interaction_matrix.transpose()
        data_dict = dict(zip(zip(inter_M.row, inter_M.col + self.n_users), [1] * inter_M.nnz))
        data_dict.update(dict(zip(zip(inter_M_t.row + self.n_users, inter_M_t.col), [1] * inter_M_t.nnz)))
        A._update(data_dict)
        sumArr = (A > 0).sum(axis=1)
        diag = np.array(sumArr.flatten())[0] + 1e-7
        diag = np.power(diag, -0.5)
        D = sp.diags(diag)
        L = D * A * D
        L = sp.coo_matrix(L)
        i = torch.LongTensor(np.array([L.row, L.col]))
        data = torch.FloatTensor(L.data)

        return torch.sparse.FloatTensor(i, data, torch.Size((self.n_nodes, self.n_nodes)))

    def structural_reliability_reweight(self, torch_sparse_adj, beta):
        torch_sparse_adj = torch_sparse_adj.coalesce()
        indices = torch_sparse_adj.indices().cpu().numpy()
        values = torch_sparse_adj.values().cpu().numpy()
        shape = torch_sparse_adj.shape
        binary_adj = sp.coo_matrix((np.ones_like(values), (indices[0], indices[1])), shape=shape).tocsr()
        binary_adj.setdiag(0)
        binary_adj.eliminate_zeros()
        metric = self.reliability_metric
        degrees = np.asarray(binary_adj.sum(axis=1)).flatten()
        if metric == 'AA':
            inv_log_deg = np.zeros_like(degrees, dtype=np.float32)
            valid_mask = degrees > 1
            inv_log_deg[valid_mask] = 1.0 / np.log(degrees[valid_mask])
            D_aa = sp.diags(inv_log_deg)
            score_matrix = (binary_adj @ D_aa @ binary_adj.T).tocsr()
            scores = np.asarray(score_matrix[indices[0], indices[1]]).flatten()
        elif metric == 'RA':
            inv_deg = np.zeros_like(degrees, dtype=np.float32)
            valid_mask = degrees > 0
            inv_deg[valid_mask] = 1.0 / degrees[valid_mask]
            D_ra = sp.diags(inv_deg)
            score_matrix = (binary_adj @ D_ra @ binary_adj.T).tocsr()
            scores = np.asarray(score_matrix[indices[0], indices[1]]).flatten()
        elif metric == 'LP':
            A = binary_adj
            A2 = (A @ A).tocsr()
            A3 = (A2 @ A).tocsr()
            eps = 0.01
            score_matrix = A2 + eps * A3
            scores = np.asarray(score_matrix[indices[0], indices[1]]).flatten()
        elif metric == 'Jaccard':
            cn_matrix = (binary_adj @ binary_adj.T).tocsr()
            intersection = np.asarray(cn_matrix[indices[0], indices[1]]).flatten()
            row_deg = degrees[indices[0]]
            col_deg = degrees[indices[1]]
            union = row_deg + col_deg - intersection + 1e-8
            scores = intersection / union
        else:
            raise ValueError(f'Unknown reliability metric: {metric}')
        max_score = scores.max()
        if max_score > 0:
            scores = scores / max_score
        score_mean = scores.mean()
        reliability = np.exp(beta * (scores - score_mean))
        refined_values = values * reliability
        refined_adj = sp.coo_matrix((refined_values, (indices[0], indices[1])), shape=shape)
        refined_adj.eliminate_zeros()
        final_indices = np.vstack((refined_adj.row, refined_adj.col))
        final_values = refined_adj.data
        row_sum = np.asarray(refined_adj.sum(axis=1)).flatten() + 1e-7
        col_sum = np.asarray(refined_adj.sum(axis=0)).flatten() + 1e-7
        r_inv_sqrt = np.power(row_sum, -0.5)
        c_inv_sqrt = np.power(col_sum, -0.5)
        normalized_values = final_values * r_inv_sqrt[final_indices[0]] * c_inv_sqrt[final_indices[1]]
        new_indices = torch.LongTensor(final_indices)
        new_values = torch.FloatTensor(normalized_values)
        refined_sparse_adj = torch.sparse.FloatTensor(new_indices, new_values, shape).coalesce().to(self.device)

        return refined_sparse_adj

    def forward(self):
        image_feats, text_feats = None, None
        if self.v_feat is not None:
            image_feats = self.image_trs(self.image_embedding.weight)
            image_feats = F.normalize(image_feats) * self.mm_image_weight
        if self.t_feat is not None:
            text_feats = self.text_trs(self.text_embedding.weight)
            text_feats = F.normalize(text_feats) * (1.0 - self.mm_image_weight)
        item_embeds = torch.cat([f for f in [image_feats, text_feats] if f is not None], dim=1)
        u_img_embeds, u_txt_embeds = None, None
        if self.v_feat is not None:
            u_img_embeds = self.user_image.weight * self.mm_image_weight
        if self.t_feat is not None:
            u_txt_embeds = self.user_text.weight * (1.0 - self.mm_image_weight)
        user_embeds = torch.cat([u for u in [u_img_embeds, u_txt_embeds] if u is not None], dim=1)
        h = item_embeds
        for _ in range(self.n_layers):
            h = torch.sparse.mm(self.mm_adj, h)
        ego_embeddings = torch.cat((user_embeds, item_embeds), dim=0)
        all_embeddings = [ego_embeddings]
        for _ in range(self.n_ui_layers):
            ego_embeddings = torch.sparse.mm(self.norm_adj, ego_embeddings)
            all_embeddings.append(ego_embeddings)
        all_embeddings = torch.stack(all_embeddings, dim=1).mean(dim=1, keepdim=False)
        u_g_embeddings, i_g_embeddings = torch.split(all_embeddings, [self.n_users, self.n_items], dim=0)

        return u_g_embeddings, i_g_embeddings, h

    def bpr_loss(self, users, pos_items, neg_items):
        pos_scores = torch.sum(torch.mul(users, pos_items), dim=1)
        neg_scores = torch.sum(torch.mul(users, neg_items), dim=1)
        maxi = F.logsigmoid(pos_scores - neg_scores)

        return -torch.mean(maxi)

    def ssl_loss(self, view1, view2):
        view1 = F.normalize(view1, p=2, dim=1)
        view2 = F.normalize(view2, p=2, dim=1)
        logits = torch.matmul(view1, view2.transpose(0, 1)) / self.ssl_temp
        labels = torch.arange(logits.shape[0]).to(self.device)

        return F.cross_entropy(logits, labels)

    def calculate_loss(self, interaction):
        users = interaction[0]
        pos_items = interaction[1]
        neg_items = interaction[2]
        u_g_embeddings, i_g_embeddings, h = self.forward()
        ia_embeddings = i_g_embeddings + h
        u_g_batch = u_g_embeddings[users]
        pos_i_g_batch = ia_embeddings[pos_items]
        neg_i_g_batch = ia_embeddings[neg_items]
        batch_mf_loss = self.bpr_loss(u_g_batch, pos_i_g_batch, neg_i_g_batch)
        reg_loss = (1 / 2) * (u_g_batch.norm(2).pow(2) +
                              pos_i_g_batch.norm(2).pow(2) +
                              neg_i_g_batch.norm(2).pow(2)) / float(len(users))
        unique_items = torch.unique(pos_items)
        view1 = i_g_embeddings[unique_items]
        view2 = h[unique_items]
        ssl_loss = self.ssl_loss(view1, view2)

        return batch_mf_loss + self.reg_weight * reg_loss + self.ssl_weight * ssl_loss

    def full_sort_predict(self, interaction):
        user = interaction[0]
        u_embeddings, i_g_embeddings, h = self.forward()
        restore_user_e = u_embeddings
        restore_item_e = i_g_embeddings + h
        u_batch_embeddings = restore_user_e[user]
        scores = torch.matmul(u_batch_embeddings, restore_item_e.transpose(0, 1))

        return scores
