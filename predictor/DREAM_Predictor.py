import argparse
import time
import torch
import torch.nn.functional as F
import numpy as np
from copy import deepcopy
import nni
import os
import matplotlib.pyplot as plt

from predictor.Base_Predictor import Predictor
from predictor.module.GNNs import EmbedGCN, SimilarityMLP
from torch_geometric.utils import dropout_edge, k_hop_subgraph

INVALID_SIM_VALUE = -2.0


@torch.no_grad()
def precompute_k_hop_neighbors(num_nodes, edge_index, h, device):
    if h <= 0:
        return None
    neighbor_map = {}
    edge_index_dev = edge_index.to(device)
    for node_idx in range(num_nodes):
        subset, _, _, _ = k_hop_subgraph(node_idx=node_idx, num_hops=h,
                                         edge_index=edge_index_dev, relabel_nodes=False,
                                         num_nodes=num_nodes, flow='source_to_target')
        neighbors = subset[subset != node_idx].unique()
        neighbor_map[node_idx] = neighbors
    return neighbor_map


@torch.no_grad()
def create_topology_adjacency_mask(num_nodes, topology_neighbors_map, device):
    if topology_neighbors_map is None:
        return torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    mask = torch.zeros((num_nodes, num_nodes), dtype=torch.bool, device=device)
    for i, neighbors in topology_neighbors_map.items():
        if neighbors.numel() > 0:
            valid_neighbor_indices = neighbors[neighbors < num_nodes]
            if valid_neighbor_indices.numel() > 0:
                mask[i, valid_neighbor_indices] = True
    return mask


class dream_Predictor(Predictor):
    def __init__(self, conf, data, device='cuda:0'):
        conf.model.setdefault('k_topology', 10)
        conf.model.setdefault('k_proximity', 15)
        conf.model.setdefault('temperature', 0.04)
        conf.model.setdefault('d_max', 4)
        conf.training.setdefault('warmup_epochs', 3)

        conf.training.setdefault('lambda_cl', 0.02)
        conf.training.setdefault('cl_temperature', 0.5)
        conf.training.setdefault('warmup_epochs_cl', 70)
        conf.training.setdefault('drop_edge_rate1', 0.2)
        conf.training.setdefault('mask_feat_rate1', 0.3)
        conf.training.setdefault('drop_edge_rate2', 0.3)
        conf.training.setdefault('mask_feat_rate2', 0.2)

        ablation_settings = getattr(conf, 'ablation', argparse.Namespace())

        self.ablation_no_homogeneity_score = getattr(ablation_settings, 'disable_homogeneity_reweighting', False)
        if self.ablation_no_homogeneity_score:
            print("ABLATION STUDY: Homogeneity Score disabled (all weights are 1).")

        self.ablation_mlp_sim = getattr(ablation_settings, 'mlp_sim', False)
        if self.ablation_mlp_sim:
            print("ABLATION STUDY: Using a trainable MLP for similarity.")
            self.sim_mlp = None

        super().__init__(conf, data, device)

        if self.noisy_label is None:
            raise ValueError("This predictor requires noisy_label.")
        self.noisy_label = self.noisy_label.to(self.device)
        self.current_epoch = 0

        self.plot_homogeneity_scores = self.conf.training.get('plot_homogeneity_scores', False)
        self._last_train_homogeneity_scores = None

        if self.plot_homogeneity_scores:
            self.avg_clean_homogeneity_history = []
            self.avg_noisy_homogeneity_history = []

            if self.train_mask is not None and len(self.train_mask) > 0:
                train_nodes_clean_labels = self.clean_label[self.train_mask]
                train_nodes_noisy_labels = self.noisy_label[self.train_mask]

                self._is_clean_among_train_nodes = (train_nodes_clean_labels == train_nodes_noisy_labels).to(self.device)
            else:
                self._is_clean_among_train_nodes = torch.empty(0, dtype=torch.bool, device=self.device)

        self.d_max = conf.model['d_max']
        self.topology_neighbors_map = None
        self.topology_adjacency_mask = None

        if self.d_max > 0:
            if self.edge_index is None:
                raise ValueError("Edge index not initialized.")
            self.topology_neighbors_map = precompute_k_hop_neighbors(self.n_nodes, self.edge_index, self.d_max, self.device)
            self.topology_adjacency_mask = create_topology_adjacency_mask(self.n_nodes, self.topology_neighbors_map, self.device)
        else:
            self.topology_adjacency_mask = torch.zeros((self.n_nodes, self.n_nodes), dtype=torch.bool, device=self.device)

    def method_init(self, conf, data):
        self.model = EmbedGCN(in_channels=conf.model['n_feat'],
                              hidden_channels=conf.model['n_hidden'],
                              out_channels=conf.model['n_classes'],
                              n_layers=conf.model['n_layer'],
                              dropout=conf.model['dropout'],
                              norm_info=conf.model.get('norm_info'),
                              act=conf.model['act'],
                              input_layer=conf.model.get('input_layer', False),
                              output_layer=conf.model.get('output_layer', False)).to(self.device)

        if self.ablation_mlp_sim:
            mlp_input_dim = conf.model['n_hidden'] * 2
            self.sim_mlp = SimilarityMLP(input_dim=mlp_input_dim).to(self.device)
            from itertools import chain
            self.optim = torch.optim.Adam(chain(self.model.parameters(), self.sim_mlp.parameters()),
                                          lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])
        else:
            self.optim = torch.optim.Adam(self.model.parameters(),
                                          lr=self.conf.training['lr'],
                                          weight_decay=self.conf.training['weight_decay'])

        ablation_settings = getattr(self.conf, 'ablation', argparse.Namespace())

        self.k_topology_original = self.conf.model.get('k_topology', 10)
        self.k_topology = self.k_topology_original
        if getattr(ablation_settings, 'disable_topology_aware_anchors', False):
            print("ABLATION STUDY: k_topology set to 0 (disable_topology_aware_anchors).")
            self.k_topology = 0

        self.k_proximity_original = self.conf.model.get('k_proximity', 15)
        self.k_proximity = self.k_proximity_original
        if getattr(ablation_settings, 'disable_proximity_aware_anchors', False):
            print("ABLATION STUDY: k_proximity set to 0 (disable_proximity_aware_anchors).")
            self.k_proximity = 0

        self.ablation_topk_from_all = getattr(ablation_settings, 'topk_from_all', False)
        if self.ablation_topk_from_all:
            print("ABLATION STUDY: Selecting TopK from ALL other nodes.")

        self.ablation_combine_topk = getattr(ablation_settings, 'combine_topk', False)
        if self.ablation_combine_topk:
            print("ABLATION STUDY: Combining local and same-class neighbors before TopK.")

        self.temperature_original = self.conf.model.get('temperature', 0.04)
        self.temperature = self.temperature_original
        if getattr(ablation_settings, 'no_temp_scale', False):
            print("ABLATION STUDY: Temperature scaling disabled (temperature = 1.0).")
            self.temperature = 1.0

        self.lambda_cl_original = self.conf.training.get('lambda_cl', 0.02)
        self.lambda_cl = self.lambda_cl_original
        if getattr(ablation_settings, 'no_cl', False):
            print("ABLATION STUDY: Contrastive Learning disabled (lambda_cl = 0).")
            self.lambda_cl = 0.0

        self.warmup_epochs = self.conf.training.get('warmup_epochs', 3)
        self.cl_temperature = self.conf.training.get('cl_temperature', 0.5)
        self.warmup_epochs_cl = conf.training['warmup_epochs_cl']
        self.drop_edge_rate1 = conf.training['drop_edge_rate1']
        self.mask_feat_rate1 = conf.training['mask_feat_rate1']
        self.drop_edge_rate2 = conf.training['drop_edge_rate2']
        self.mask_feat_rate2 = conf.training['mask_feat_rate2']

    def get_homogeneity_scores(self, embeddings, current_mask):
        n_nodes = embeddings.shape[0]
        k_topology = self.k_topology
        k_proximity = self.k_proximity

        target_indices = torch.where(current_mask)[0]
        n_target = len(target_indices)

        should_calculate_sim = True
        if self.ablation_topk_from_all:
            if (self.k_topology_original + self.k_proximity_original <= 0) or n_target == 0:
                should_calculate_sim = False
        elif self.ablation_combine_topk:
            if (self.k_topology_original + self.k_proximity_original <= 0) or n_target == 0:
                should_calculate_sim = False
        else:
            if (k_topology <= 0 and k_proximity <= 0) or n_target == 0:
                should_calculate_sim = False

        if not should_calculate_sim:
            default_scores = torch.ones(n_target, device=self.device)
            raw_sims = torch.full((n_target,), -1.0, device=self.device)
            return default_scores, raw_sims

        if self.ablation_mlp_sim:
            target_embs = embeddings[target_indices]

            repeated_targets = target_embs.repeat_interleave(n_nodes, dim=0)
            tiled_all = embeddings.repeat(n_target, 1)

            mlp_input = torch.cat([repeated_targets, tiled_all], dim=1)

            sim_scores = self.sim_mlp(mlp_input).squeeze()
            sims_for_targets = sim_scores.view(n_target, n_nodes)
        else:
            emb_norm = F.normalize(embeddings, p=2, dim=1)
            cosine_sim_matrix = torch.matmul(emb_norm, emb_norm.t())
            sims_for_targets = cosine_sim_matrix[target_indices, :]

        pred_labels = self.noisy_label
        same_class_mask = pred_labels.unsqueeze(1) == pred_labels.unsqueeze(0)
        topology_mask = self.topology_adjacency_mask
        identity_mask = ~torch.eye(n_nodes, dtype=torch.bool, device=self.device)

        if self.ablation_topk_from_all:
            k_total_for_all = self.k_topology_original + self.k_proximity_original
            if k_total_for_all <= 0:
                final_avg_similarities = torch.full((n_target,), -1.0, device=self.device)
            else:
                masked_sims_all = sims_for_targets.clone()

                for i, target_node_idx in enumerate(target_indices):
                    masked_sims_all[i, target_node_idx] = INVALID_SIM_VALUE

                actual_k_total = min(k_total_for_all, n_nodes - 1 if n_nodes > 1 else 1)
                final_avg_similarities = torch.full((n_target,), INVALID_SIM_VALUE, device=self.device)

                if actual_k_total > 0:
                    topk_sims_all, _ = torch.topk(masked_sims_all, k=actual_k_total, dim=1)

                    valid_topk_all_mask = topk_sims_all > (INVALID_SIM_VALUE + 1e-5)
                    topk_sims_all_valid = topk_sims_all * valid_topk_all_mask

                    sum_valid_sims_all = torch.sum(topk_sims_all_valid, dim=1)
                    count_valid_neighbors_all = torch.sum(valid_topk_all_mask, dim=1)

                    has_valid_all_neighbors_mask = count_valid_neighbors_all > 0
                    final_avg_similarities[has_valid_all_neighbors_mask] = sum_valid_sims_all[has_valid_all_neighbors_mask] / count_valid_neighbors_all[has_valid_all_neighbors_mask]

                final_avg_similarities[final_avg_similarities <= (INVALID_SIM_VALUE + 1e-5)] = -1.0
        elif self.ablation_combine_topk:
            k_combined = self.k_topology_original + self.k_proximity_original
            if k_combined <= 0:
                final_avg_similarities = torch.full((n_target,), -1.0, device=self.device)
            else:
                combined_candidate_mask_for_targets = torch.zeros((n_target, n_nodes), dtype=torch.bool, device=self.device)

                if self.k_topology_original > 0 and self.d_max > 0:
                    local_candidate_base = topology_mask & identity_mask
                    combined_candidate_mask_for_targets |= local_candidate_base[target_indices, :]

                if self.k_proximity_original > 0:
                    same_class_candidate_base = same_class_mask & identity_mask
                    combined_candidate_mask_for_targets |= same_class_candidate_base[target_indices, :]

                masked_sims_combined = sims_for_targets.clone()
                masked_sims_combined[~combined_candidate_mask_for_targets] = INVALID_SIM_VALUE

                actual_k_combined = min(k_combined, n_nodes - 1 if n_nodes > 1 else 1)

                final_avg_similarities = torch.full((n_target,), INVALID_SIM_VALUE, device=self.device)

                if actual_k_combined > 0 and torch.any(combined_candidate_mask_for_targets):
                    topk_sims_combined, _ = torch.topk(masked_sims_combined, k=actual_k_combined, dim=1)

                    valid_topk_combined_mask = topk_sims_combined > (INVALID_SIM_VALUE + 1e-5)
                    topk_sims_combined_valid = topk_sims_combined * valid_topk_combined_mask

                    sum_valid_sims_combined = torch.sum(topk_sims_combined_valid, dim=1)
                    count_valid_neighbors_combined = torch.sum(valid_topk_combined_mask, dim=1)

                    has_valid_combined_neighbors_mask = count_valid_neighbors_combined > 0
                    final_avg_similarities[has_valid_combined_neighbors_mask] = sum_valid_sims_combined[has_valid_combined_neighbors_mask] / count_valid_neighbors_combined[has_valid_combined_neighbors_mask]
                final_avg_similarities[final_avg_similarities <= (INVALID_SIM_VALUE + 1e-5)] = -1.0
        else:
            avg_sims_topology = torch.full((n_target,), INVALID_SIM_VALUE, device=self.device)
            sum_valid_sims_topology = torch.zeros(n_target, device=self.device)
            count_valid_neighbors_topology = torch.zeros(n_target, device=self.device, dtype=torch.long)

            if k_topology > 0 and self.d_max > 0:
                topology_candidate_mask = topology_mask & identity_mask
                topology_candidate_mask_for_targets = topology_candidate_mask[target_indices, :]

                masked_sims_topology = sims_for_targets.clone()
                masked_sims_topology[~topology_candidate_mask_for_targets] = INVALID_SIM_VALUE

                actual_k_topology = min(k_topology, n_nodes - 1 if n_nodes > 1 else 1)
                if actual_k_topology > 0 and torch.any(topology_candidate_mask_for_targets):
                    topk_sims_topology, _ = torch.topk(masked_sims_topology, k=actual_k_topology, dim=1)

                    valid_topk_topology_mask = topk_sims_topology > (INVALID_SIM_VALUE + 1e-5)
                    topk_sims_topology_valid = topk_sims_topology * valid_topk_topology_mask
                    sum_valid_sims_topology = torch.sum(topk_sims_topology_valid, dim=1)
                    count_valid_neighbors_topology = torch.sum(valid_topk_topology_mask, dim=1)

                    has_valid_topology_neighbors_mask = count_valid_neighbors_topology > 0
                    avg_sims_topology[has_valid_topology_neighbors_mask] = sum_valid_sims_topology[has_valid_topology_neighbors_mask] / count_valid_neighbors_topology[has_valid_topology_neighbors_mask]

            avg_sims_proximity = torch.full((n_target,), INVALID_SIM_VALUE, device=self.device)
            sum_valid_sims_proximity = torch.zeros(n_target, device=self.device)
            count_valid_neighbors_proximity = torch.zeros(n_target, device=self.device, dtype=torch.long)

            if k_proximity > 0:
                proximity_candidate_mask = same_class_mask & identity_mask
                proximity_candidate_mask_for_targets = proximity_candidate_mask[target_indices, :]

                masked_sims_proximity = sims_for_targets.clone()
                masked_sims_proximity[~proximity_candidate_mask_for_targets] = INVALID_SIM_VALUE

                actual_k_proximity = min(k_proximity, n_nodes - 1 if n_nodes > 1 else 1)
                if actual_k_proximity > 0 and torch.any(proximity_candidate_mask_for_targets):
                    topk_sims_proximity, _ = torch.topk(masked_sims_proximity, k=actual_k_proximity, dim=1)

                    valid_topk_proximity_mask = topk_sims_proximity > (INVALID_SIM_VALUE + 1e-5)
                    topk_sims_proximity_valid = topk_sims_proximity * valid_topk_proximity_mask
                    sum_valid_sims_proximity = torch.sum(topk_sims_proximity_valid, dim=1)
                    count_valid_neighbors_proximity = torch.sum(valid_topk_proximity_mask, dim=1)

                    has_valid_same_class_neighbors_mask = count_valid_neighbors_proximity > 0
                    avg_sims_proximity[has_valid_same_class_neighbors_mask] = sum_valid_sims_proximity[has_valid_same_class_neighbors_mask] / count_valid_neighbors_proximity[has_valid_same_class_neighbors_mask]

            total_sum_valid_sims = sum_valid_sims_topology + sum_valid_sims_proximity
            total_count_valid_neighbors = count_valid_neighbors_topology + count_valid_neighbors_proximity

            final_avg_similarities = torch.full_like(total_sum_valid_sims, -1.0)
            has_any_valid_neighbors_mask = total_count_valid_neighbors > 0
            final_avg_similarities[has_any_valid_neighbors_mask] = total_sum_valid_sims[has_any_valid_neighbors_mask] / total_count_valid_neighbors[has_any_valid_neighbors_mask]

        avg_anchor_similarity = (final_avg_similarities.clone() + 1.0) / 2.0
        avg_anchor_similarity = torch.clamp(avg_anchor_similarity, min=0.0, max=1.0)

        unscaled_homogeneity_scores = (final_avg_similarities + 1.0) / 2.0
        unscaled_homogeneity_scores = torch.clamp(unscaled_homogeneity_scores, min=1e-6, max=1.0)

        homogeneity_scores = unscaled_homogeneity_scores.clone()
        if self.temperature != 1.0 and self.temperature > 0:
            homogeneity_scores = torch.pow(homogeneity_scores, 1.0 / self.temperature)
        homogeneity_scores = torch.clamp(homogeneity_scores, min=0.0, max=1.0)

        if torch.isnan(homogeneity_scores).any():
            print(f"Warning: NaN detected in SCALED weights @ epoch {self.current_epoch}. Replacing with 1.0.")
            homogeneity_scores = torch.nan_to_num(homogeneity_scores, nan=1.0)

        if torch.isnan(avg_anchor_similarity).any():
            avg_anchor_similarity = torch.nan_to_num(avg_anchor_similarity, nan=-1.0)

        return homogeneity_scores, avg_anchor_similarity

    def get_prediction(self, features, edge_index, label=None, mask=None, calculate_homogeneity_scores=False):
        output, hidden_embeds = self.model(features, edge_index)
        loss, acc = None, None

        current_homogeneity_scores = None
        current_raw_similarities = None

        if (label is not None) and (mask is not None):
            if isinstance(mask, (np.ndarray, list)):
                mask_np = np.asarray(mask)
                if mask_np.size == 0:
                    return output, torch.tensor(0.0, device=self.device), 0.0, None, None
                boolean_mask = torch.zeros(output.shape[0], dtype=torch.bool, device=self.device)
                if len(mask_np) > 0:
                    boolean_mask[mask_np] = True
            elif isinstance(mask, torch.Tensor):
                if mask.dtype == torch.bool:
                    boolean_mask = mask
                else:
                    if mask.numel() == 0:
                        return output, torch.tensor(0.0, device=self.device), 0.0, None, None
                    boolean_mask = torch.zeros(output.shape[0], dtype=torch.bool, device=self.device)
                    if mask.numel() > 0:
                        boolean_mask[mask] = True
            else:
                raise TypeError("Mask must be a list, numpy array, or torch tensor.")

            boolean_mask = mask

            if not boolean_mask.any():
                return output, torch.tensor(0.0, device=self.device), 0.0, None, None

            masked_output = output[boolean_mask]
            masked_label = label[boolean_mask]

            if self.loss_fn == F.cross_entropy and masked_label.dtype != torch.long:
                masked_label = masked_label.long()
            elif self.loss_fn == F.binary_cross_entropy_with_logits:
                if masked_label.dtype != torch.float:
                    masked_label = masked_label.float()
                if masked_output.ndim == 2 and masked_output.shape[1] == 1:
                    masked_output = masked_output.squeeze(1)
                if masked_label.ndim == 2 and masked_label.shape[1] == 1:
                    masked_label = masked_label.squeeze(1)

            try:
                raw_loss = self.loss_fn(masked_output, masked_label, reduction='none')
            except Exception as e:
                print(f"Error calculating raw loss @ epoch {self.current_epoch}: {e}")
                return output, torch.tensor(0.0, device=self.device, requires_grad=True), 0.0

            if calculate_homogeneity_scores:
                homogeneity_scores, raw_similarities_local = None, None
                if self.ablation_no_homogeneity_score or self.current_epoch < self.warmup_epochs:
                    homogeneity_scores = torch.ones_like(raw_loss, device=self.device)
                    raw_similarities_local = torch.zeros_like(raw_loss, device=self.device)
                else:
                    if self.ablation_mlp_sim and self.model.training:
                        homogeneity_scores, raw_similarities_local = self.get_homogeneity_scores(hidden_embeds, boolean_mask)
                    else:
                        with torch.no_grad():
                            homogeneity_scores, raw_similarities_local = self.get_homogeneity_scores(hidden_embeds.detach(), boolean_mask)

                current_homogeneity_scores = homogeneity_scores
                current_raw_similarities = raw_similarities_local

                if raw_loss.shape == homogeneity_scores.shape:
                    valid_mask = ~torch.isnan(raw_loss) & ~torch.isnan(homogeneity_scores)
                    weighted_loss = raw_loss[valid_mask] * homogeneity_scores[valid_mask]
                    loss = torch.mean(weighted_loss) if weighted_loss.numel() > 0 else torch.tensor(0.0, device=self.device)
                else:
                    print(f"Warning: Shape mismatch raw_loss {raw_loss.shape} / homogeneity_scores {homogeneity_scores.shape}. Using mean raw loss.")
                    loss = torch.mean(raw_loss[~torch.isnan(raw_loss)])

                if torch.isnan(loss):
                    print(f"Warning: Final weighted loss is NaN @ epoch {self.current_epoch}. Setting to 0.")
                    loss = torch.tensor(0.0, device=self.device)
            else:
                loss = torch.mean(raw_loss[~torch.isnan(raw_loss)])
                if torch.isnan(loss):
                    loss = torch.tensor(0.0, device=self.device)

            try:
                acc = self.metric(masked_label.cpu().numpy(), masked_output.detach().cpu().numpy())
            except Exception as e:
                print(f"Error calculating metric: {e}")
                acc = 0.0

        if loss is None:
            loss = torch.tensor(0.0, device=self.device)
        if acc is None:
            acc = 0.0

        return output, loss, acc, current_homogeneity_scores, current_raw_similarities

    def augment_graph(self, features, edge_index, drop_edge_rate, mask_feat_rate):
        aug_edge_index, _ = dropout_edge(edge_index, p=drop_edge_rate,
                                         force_undirected=True, training=self.model.training)
        num_nodes, num_feats = features.shape
        mask = torch.empty((num_nodes, num_feats), dtype=torch.float32,
                           device=self.device).uniform_(0, 1) < mask_feat_rate
        aug_features = features.clone()
        aug_features[mask] = 0.0
        return aug_features, aug_edge_index

    def contrastive_loss(self, z1, z2, temperature):
        z1_norm = F.normalize(z1, p=2, dim=1)
        z2_norm = F.normalize(z2, p=2, dim=1)
        sim_matrix = torch.mm(z1_norm, z2_norm.t()) / temperature
        n = sim_matrix.shape[0]
        labels = torch.arange(n, device=self.device)
        loss = F.cross_entropy(sim_matrix, labels)
        return loss

    def train(self):
        train_bool_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        val_bool_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        test_bool_mask = torch.zeros(self.n_nodes, dtype=torch.bool, device=self.device)
        if self.train_mask is not None and len(self.train_mask) > 0:
            train_bool_mask[self.train_mask] = True
        if self.val_mask is not None and len(self.val_mask) > 0:
            val_bool_mask[self.val_mask] = True
        if self.test_mask is not None and len(self.test_mask) > 0:
            test_bool_mask[self.test_mask] = True

        if not train_bool_mask.any():
            print("Error: Training mask is empty. Cannot train.")
            return {'train': -1, 'valid': -1, 'test': -1}

        for epoch in range(self.conf.training['n_epochs']):
            self.current_epoch = epoch
            improve = ''
            t0 = time.time()

            self.model.train()
            self.optim.zero_grad()
            features, edge_index = self.feats, self.edge_index

            loss_cl = torch.tensor(0.0, device=self.device)
            if self.lambda_cl > 0 and epoch >= self.warmup_epochs_cl:
                try:
                    features1, edge_index1 = self.augment_graph(features, edge_index, self.drop_edge_rate1, self.mask_feat_rate1)
                    features2, edge_index2 = self.augment_graph(features, edge_index, self.drop_edge_rate2, self.mask_feat_rate2)
                    _, embeds1 = self.model(features1, edge_index1)
                    _, embeds2 = self.model(features2, edge_index2)
                    loss_cl = self.contrastive_loss(embeds1, embeds2, self.cl_temperature)
                    if torch.isnan(loss_cl) or torch.isinf(loss_cl):
                        loss_cl = torch.tensor(0.0, device=self.device)
                except Exception as e:
                    print(f"Error CL @ epoch {epoch + 1}: {e}")
                    loss_cl = torch.tensor(0.0, device=self.device)

            try:
                output, loss_train_weighted_sup, acc_train, current_train_homogeneity_scores, current_train_raw_similarities = self.get_prediction(
                    features, edge_index, self.noisy_label, train_bool_mask, calculate_homogeneity_scores=True)

                if self.plot_homogeneity_scores and epoch >= self.warmup_epochs:
                    if current_train_homogeneity_scores is not None and \
                            current_train_homogeneity_scores.numel() == self._is_clean_among_train_nodes.numel() and \
                            current_train_homogeneity_scores.numel() > 0:

                        clean_mask_for_vals = self._is_clean_among_train_nodes
                        noisy_mask_for_vals = ~self._is_clean_among_train_nodes

                        clean_node_homogeneity_scores = current_train_homogeneity_scores[clean_mask_for_vals]
                        noisy_node_homogeneity_scores = current_train_homogeneity_scores[noisy_mask_for_vals]
                        avg_clean_score = clean_node_homogeneity_scores.mean().item() if clean_node_homogeneity_scores.numel() > 0 else np.nan
                        avg_noisy_score = noisy_node_homogeneity_scores.mean().item() if noisy_node_homogeneity_scores.numel() > 0 else np.nan
                        self.avg_clean_homogeneity_history.append(avg_clean_score)
                        self.avg_noisy_homogeneity_history.append(avg_noisy_score)
                    else:
                        self.avg_clean_homogeneity_history.append(np.nan)
                        self.avg_noisy_homogeneity_history.append(np.nan)
                elif self.plot_homogeneity_scores and epoch < self.warmup_epochs:
                    self.avg_clean_homogeneity_history.append(np.nan)
                    self.avg_noisy_homogeneity_history.append(np.nan)

                if loss_train_weighted_sup is None:
                    loss_train_weighted_sup = torch.tensor(0.0, device=self.device)
                    acc_train = 0.0
                elif torch.isnan(loss_train_weighted_sup) or torch.isinf(loss_train_weighted_sup):
                    loss_train_weighted_sup = torch.tensor(0.0, device=self.device)
            except Exception as e:
                print(f"Error Sup @ epoch {epoch + 1}: {e}")
                import traceback
                traceback.print_exc()
                loss_train_weighted_sup = torch.tensor(0.0, device=self.device)
                acc_train = 0.0

            total_loss = loss_train_weighted_sup + self.lambda_cl * loss_cl
            if not torch.isnan(total_loss) and not torch.isinf(total_loss) and total_loss.requires_grad:
                total_loss.backward()
                self.optim.step()

            if val_bool_mask.any():
                loss_val, acc_val = self.evaluate(self.noisy_label, val_bool_mask)
                if loss_val is None or acc_val is None:
                    print(f"Warning: Val results None @ {epoch + 1}. Stopping.")
                    break
            else:
                loss_val, acc_val = torch.tensor(float('inf')), 0.0

            flag, flag_earlystop = self.recoder.add(loss_val, acc_val)
            if flag and val_bool_mask.any():
                improve = '*'
                self.total_time = time.time() - self.start_time
                self.best_val_loss = loss_val
                self.result['valid'] = acc_val
                self.result['train'] = acc_train
                self.weights = deepcopy(self.model.state_dict())
            elif flag_earlystop and val_bool_mask.any():
                if self.conf.training['debug']:
                    print(f"Early stopping @ epoch {epoch + 1}.")
                break

            if self.conf.training['debug']:
                acc_test_clean = -1.0
                if test_bool_mask.any():
                    _, acc_test_clean_eval = self.evaluate(self.clean_label, test_bool_mask)
                    if acc_test_clean_eval is not None:
                        acc_test_clean = acc_test_clean_eval

                log_str = f"E {epoch + 1:04d} | T {time.time() - t0:.2f} | Ls(Sup) {loss_train_weighted_sup.item():.4f} | "
                if self.lambda_cl > 0 and epoch >= self.warmup_epochs_cl:
                    log_str += f"Ls(CL) {loss_cl.item():.4f} | "
                log_str += f"Acc(Tr) {acc_train:.4f} | Ls(Vl) {loss_val.item():.4f} | Acc(Vl) {acc_val:.4f} | "
                log_str += f"Acc(Te_Cl) {acc_test_clean:.4f} | {improve}"
                print(log_str)
                if 'NNI_OUTPUT_DIR' in os.environ:
                    nni.report_intermediate_result(acc_val)

        if test_bool_mask.any():
            loss_test, acc_test = self.test(test_bool_mask)
            self.result['test'] = acc_test if acc_test is not None else -1.0
            loss_test_item = loss_test.item() if loss_test is not None else -1.0
        else:
            print("Skipping final test: Test mask empty.")
            self.result['test'] = -1.0
            loss_test_item = -1.0

        if self.conf.training['debug']:
            print('Optimization Finished!')
            final_time = self.total_time if self.total_time > 0 else time.time() - self.start_time
            print(f'Time(s): {final_time:.4f}')
            best_val_acc = self.recoder.best_metric if self.recoder.best_metric is not None else -1.0
            print(f"Best Acc(val): {best_val_acc:.4f}")
            print(f"Loss(test) {loss_test_item:.4f} | Acc(test) {self.result['test']:.4f}")
            if 'NNI_OUTPUT_DIR' in os.environ:
                nni.report_final_result(self.result['test'])

        if self.plot_homogeneity_scores and self.avg_clean_homogeneity_history:
            try:
                import pandas as pd
                import seaborn as sns

                sns.reset_defaults()
                plt.rcParams['mathtext.fontset'] = 'cm'
                plt.rcParams['font.family'] = 'Georgia'
                plt.rcParams['text.color'] = 'black'
                plt.rcParams['font.weight'] = 'normal'
                plt.rcParams['font.size'] = 21
                plt.rcParams['text.usetex'] = False

                num_recorded_epochs = len(self.avg_clean_homogeneity_history)
                if num_recorded_epochs == 0:
                    print("Warning: No Homogeneity Score history to plot.")
                    return self.result

                epochs_to_plot = np.arange(1, len(self.avg_clean_homogeneity_history) + 1)

                window_size = 10
                clean_scores = pd.Series(self.avg_clean_homogeneity_history)
                noisy_scores = pd.Series(self.avg_noisy_homogeneity_history)

                clean_smooth = clean_scores.rolling(window=window_size, min_periods=1).mean()
                noisy_smooth = noisy_scores.rolling(window=window_size, min_periods=1).mean()

                fig, ax = plt.subplots(figsize=(5.5, 4.5))

                colors = sns.color_palette("deep", 2)

                ax.plot(epochs_to_plot, clean_scores, color=colors[0], alpha=0.25)
                ax.plot(epochs_to_plot, noisy_scores, color=colors[1], alpha=0.25)

                ax.plot(epochs_to_plot, clean_smooth, color=colors[0], linestyle='-', linewidth=2.5, label='Clean')
                ax.plot(epochs_to_plot, noisy_smooth, color=colors[1], linestyle='--', linewidth=2.5, label='Noisy')

                ax.set_xlabel('Epoch')
                ax.set_ylabel('Avg. H-Score')

                ax.set_xlim(0, len(epochs_to_plot) + 1)
                max_epoch = len(epochs_to_plot)
                ax.set_xticks(np.linspace(0, max_epoch, num=3, dtype=int))
                all_scores = self.avg_clean_homogeneity_history + self.avg_noisy_homogeneity_history

                valid_scores = [s for s in all_scores if pd.notna(s)]

                if valid_scores:
                    y_max_limit = np.nanmax(all_scores) * 1.1
                else:
                    y_max_limit = 0.07

                ax.set_ylim(0, max(y_max_limit, 0.07))
                ax.grid(False)

                ax.legend(loc='upper left', frameon=True, shadow=False, fontsize=18)
                ax.spines['right'].set_visible(True)
                ax.spines['left'].set_visible(True)
                ax.spines['bottom'].set_visible(True)
                ax.spines['top'].set_visible(True)

                plt.tight_layout()

                plot_dir = './plots'
                if not os.path.exists(plot_dir):
                    os.makedirs(plot_dir)

                dataset_name_str = getattr(self.conf, 'dataset_name', 'UnknownDataset')
                noise_rate = getattr(self.conf, 'noise_rate', '0.3')
                seed_str = getattr(self.conf, 'seed_conf', 'NA')

                filename_pdf = f"{plot_dir}/homogeneity_score_{dataset_name_str}_{noise_rate}_seed{seed_str}.pdf"
                plt.savefig(
                    filename_pdf,
                    bbox_inches='tight',
                    pad_inches=0.01
                )

                print(f"Homogeneity Score evolution plot saved to {filename_pdf}")
                if hasattr(self.conf, 'training') and hasattr(self.conf.training, 'debug') and self.conf.training.debug:
                    plt.show()
                plt.close(fig)

            except Exception as e:
                print(f"Error during plotting: {e}")
                import traceback
                traceback.print_exc()

        return self.result

    def evaluate(self, label, mask):
        self.model.eval()
        features, adj = self.feats, self.adj
        edge_idx_to_use = self.edge_index
        with torch.no_grad():
            _, loss, acc, _, _ = self.get_prediction(features, edge_idx_to_use, label, mask, calculate_homogeneity_scores=False)
        return loss, acc
