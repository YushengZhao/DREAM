import argparse
import warnings
import numpy as np
from utils.labelnoise import label_process
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed, get_neighbors
from utils.logger import MultiExpRecorder, ResultLogger
from predictor.NRGNN_Predictor import nrgnn_Predictor
from predictor.CP_Predictor import cp_Predictor
from predictor.Smodel_Predictor import smodel_Predictor
from predictor.Coteaching_Predictor import coteaching_Predictor
from predictor.GCN_Predictor import gcn_Predictor
from predictor.RTGNN_Predictor import rtgnn_Predictor
from predictor.CLNode_Predictor import clnode_Predictor
from predictor.RNCGLN_Predictor import rncgln_Predictor
from predictor.PIGNN_Predictor import pignn_Predictor
from predictor.GIN_Predictor import gin_Predictor
from predictor.DGNN_Predictor import dgnn_Predictor
from predictor.UnionNET_Predictor import unionnet_Predictor
from predictor.CGNN_Predictor import cgnn_Predictor
from predictor.JoCoR_Predictor import jocor_Predictor
from predictor.CRGNN_Predictor import crgnn_Predictor
from predictor.APL_Predictor import apl_Predictor
from predictor.SCE_Predictor import sce_Predictor
from predictor.Forward_Predictor import forward_Predictor
from predictor.Backward_Predictor import backward_Predictor
from predictor.MLP_Predictor import mlp_Predictor
from predictor.R2LP_Predictor import r2lp_Predictor
from predictor.TSS_Predictor import tss_Predictor
from predictor.DREAM_Predictor import dream_Predictor


def run_single_exp(dataset, method_name, seed, noise_type, noise_rate, device, debug=True, plot_homogeneity_scores_arg=False, ablation_args=None, k_topology_override=None, k_proximity_override=None):
    setup_seed(seed)
    model_conf = load_conf(None, method_name, dataset.name)
    dataset.noisy_label, modified_mask = label_process(labels=dataset.labels, n_classes=dataset.n_classes,
                                                       noise_type=noise_type, noise_rate=noise_rate,
                                                       random_seed=seed, debug=debug)
    incorrect_labeled_train_mask = dataset.train_masks[np.in1d(dataset.train_masks, modified_mask)]
    correct_labeled_train_mask = dataset.train_masks[~ np.in1d(dataset.train_masks, modified_mask)]
    supervised_mask = get_neighbors(dataset.adj, dataset.train_masks)
    incorrect_supervised_mask = get_neighbors(dataset.adj, incorrect_labeled_train_mask)
    correct_supervised_mask = get_neighbors(dataset.adj, correct_labeled_train_mask)
    unlabeled_incorrect_supervised_mask = dataset.test_masks[np.in1d(dataset.test_masks, incorrect_supervised_mask)]
    unlabeled_correct_supervised_mask = dataset.test_masks[np.in1d(dataset.test_masks, correct_supervised_mask)]
    unlabeled_unsupervised_mask = dataset.test_masks[~np.in1d(dataset.test_masks, supervised_mask)]

    model_conf.model['n_feat'] = dataset.dim_feats
    model_conf.model['n_classes'] = dataset.n_classes
    model_conf.training['debug'] = debug
    model_conf.training['plot_homogeneity_scores'] = plot_homogeneity_scores_arg
    model_conf.dataset_name = dataset.name
    model_conf.method_name_conf = method_name
    model_conf.seed_conf = seed
    model_conf.noise_rate = noise_rate
    if k_topology_override is not None:
        print(f"Overriding k_topology with command-line value: {k_topology_override}")
        model_conf.model['k_topology'] = k_topology_override
    if k_proximity_override is not None:
        print(f"Overriding k_proximity with command-line value: {k_proximity_override}")
        model_conf.model['k_proximity'] = k_proximity_override

    model_conf.ablation = argparse.Namespace()

    if ablation_args:
        model_conf.ablation.disable_topology_aware_anchors = getattr(ablation_args, 'ablation_disable_topology_aware_anchors', False)
        model_conf.ablation.disable_proximity_aware_anchors = getattr(ablation_args, 'ablation_disable_proximity_aware_anchors', False)
        model_conf.ablation.disable_homogeneity_reweighting = getattr(ablation_args, 'ablation_no_homogeneity_score', False)
        model_conf.ablation.no_cl = getattr(ablation_args, 'ablation_no_cl', False)
        model_conf.ablation.no_temp_scale = getattr(ablation_args, 'ablation_no_temp_scale', False)
        model_conf.ablation.combine_topk = getattr(ablation_args, 'ablation_combine_topk', False)
        model_conf.ablation.topk_from_all = getattr(ablation_args, 'ablation_topk_from_all', False)
        model_conf.ablation.mlp_sim = getattr(ablation_args, 'ablation_mlp_sim', False)
    else:
        model_conf.ablation.disable_topology_aware_anchors = False
        model_conf.ablation.disable_proximity_aware_anchors = False
        model_conf.ablation.disable_homogeneity_reweighting = False
        model_conf.ablation.no_cl = False
        model_conf.ablation.no_temp_scale = False
        model_conf.ablation.combine_topk = False
        model_conf.ablation.topk_from_all = False
        model_conf.ablation.mlp_sim = False

    predictor = eval(method_name + '_Predictor')(model_conf, dataset, device)

    original_result = predictor.train()
    extended_result = original_result.copy()

    _, correct_labeled_train_accuracy = predictor.test(correct_labeled_train_mask)
    _, incorrect_labeled_train_accuracy = predictor.test(incorrect_labeled_train_mask)
    _, incorrect_labeled_mislead_train_accuracy = predictor.evaluate(
        predictor.noisy_label, incorrect_labeled_train_mask)
    _, unlabeled_unsupervised_accuracy = predictor.test(unlabeled_unsupervised_mask)
    _, unlabeled_correct_supervised_accuracy = predictor.test(unlabeled_correct_supervised_mask)
    _, unlabeled_incorrect_supervised_accuracy = predictor.test(unlabeled_incorrect_supervised_mask)

    extended_result[
        'correct_labeled_train_accuracy'] = correct_labeled_train_accuracy if correct_labeled_train_accuracy is not None else 0.0
    extended_result[
        'incorrect_labeled_train_accuracy'] = incorrect_labeled_train_accuracy if incorrect_labeled_train_accuracy is not None else 0.0
    extended_result[
        'incorrect_labeled_mislead_train_accuracy'] = incorrect_labeled_mislead_train_accuracy if incorrect_labeled_mislead_train_accuracy is not None else 0.0
    extended_result[
        'unlabeled_correct_supervised_accuracy'] = unlabeled_correct_supervised_accuracy if unlabeled_correct_supervised_accuracy is not None else 0.0
    extended_result[
        'unlabeled_unsupervised_accuracy'] = unlabeled_unsupervised_accuracy if unlabeled_unsupervised_accuracy is not None else 0.0
    extended_result[
        'unlabeled_incorrect_supervised_accuracy'] = unlabeled_incorrect_supervised_accuracy if unlabeled_incorrect_supervised_accuracy is not None else 0.0
    extended_result['total_time'] = predictor.total_time if hasattr(predictor, 'total_time') else -1.0

    return original_result, extended_result


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int,
                        default=10,
                        help="Number of experiments for each combination of method and data")
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['gcn', 'smodel', 'coteaching', 'jocor', 'apl', 'sce', 'forward', 'backward'],
                        choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching',
                                 'apl', 'sce', 'forward', 'backward', 'mlp',
                                 'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'tss',
                                 'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn', 'r2lp',
                                 'dream'],
                        help='Select methods')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                                 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                        choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                                 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                        help='Select datasets')
    parser.add_argument('--noise_type', type=str, nargs='+',
                        default=['clean', 'pair', 'uniform'],
                        choices=['clean', 'pair', 'uniform', 'random'], help='Noise type')
    parser.add_argument('--noise_rate', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Noise rate')
    parser.add_argument('--device', type=str,
                        default='cuda:0',
                        help='Device')
    parser.add_argument('--seed', type=int,
                        default=3000, help="Random Seed")

    parser.add_argument('--plot', action='store_true',
                        help="Plot Homogeneity Score evolution for clean/noisy training nodes (only for dream).")

    parser.add_argument('--k_topology_override', type=int, default=None,
                        help="Override k_topology for DERAM predictor. Takes precedence over config.")
    parser.add_argument('--k_proximity_override', type=int, default=None,
                        help="Override k_proximity for DREAM predictor. Takes precedence over config.")

    parser.add_argument('--ablation_disable_topology_aware_anchors', action='store_true',
                        help="Ablation: Set k_topology to 0 (disable local similarity component).")
    parser.add_argument('--ablation_disable_proximity_aware_anchors', action='store_true',
                        help="Ablation: Set k_proximity to 0 (disable same-category similarity component).")
    parser.add_argument('--ablation_no_homogeneity_score', action='store_true',
                        help="Ablation: Disable Homogeneity Score mechanism (all weights are 1).")
    parser.add_argument('--ablation_no_cl', action='store_true',
                        help="Ablation: Disable Contrastive Learning component (lambda_cl = 0).")
    parser.add_argument('--ablation_no_temp_scale', action='store_true',
                        help="Ablation: Disable temperature scaling in Homogeneity Score (temperature = 1.0).")
    parser.add_argument('--ablation_combine_topk', action='store_true',
                        help="Ablation: Combine local and same-class neighbors before taking top-k.")
    parser.add_argument('--ablation_topk_from_all', action='store_true',
                        help="Ablation: Select top-K from all other nodes (K = k_topology + k_proximity).")
    parser.add_argument('--ablation_mlp_sim', action='store_true',
                        help="Ablation: Replace cosine similarity with a trainable MLP.")
    args = parser.parse_args()

    print(args)
    warnings.filterwarnings("ignore")
    data_path = './data/'
    method_list = args.methods
    data_list = args.datasets
    noise_type_list = args.noise_type
    noise_rate_list = args.noise_rate

    noise_list = []
    for noise_type in noise_type_list:
        if noise_type == 'clean':
            noise_list.append([0.0, noise_type])
            continue
        for noise_rate in noise_rate_list:
            noise_list.append([noise_rate, noise_type])

    result_recorder = ResultLogger(method_list, data_list, noise_list, args.runs)
    for noise_rate, noise_type in noise_list:
        for data_name in data_list:
            setup_seed(args.seed)
            data_conf = load_conf('./config/_dataset/' + data_name + '.yaml')
            data = Dataset(data_name, path=data_path,
                           feat_norm=data_conf.norm['feat_norm'], adj_norm=data_conf.norm['adj_norm'],
                           train_size=data_conf.split['train_size'],
                           val_size=data_conf.split['val_size'],
                           test_size=data_conf.split['test_size'],
                           train_percent=data_conf.split['train_percent'],
                           val_percent=data_conf.split['val_percent'],
                           test_percent=data_conf.split['test_percent'],
                           train_examples_per_class=data_conf.split['train_examples_per_class'],
                           val_examples_per_class=data_conf.split['val_examples_per_class'],
                           test_examples_per_class=data_conf.split['test_examples_per_class'],
                           add_self_loop=data_conf.modify['add_self_loop'],
                           from_npz=data_conf.modify['from_npz_largest_component'],
                           device=args.device,
                           split_type=data_conf.split['split_type'])

            for method_name in method_list:
                logger = MultiExpRecorder(runs=args.runs)
                for run in range(args.runs):
                    setup_seed(args.seed + run)
                    ablation_settings = argparse.Namespace(
                        ablation_disable_topology_aware_anchors=args.ablation_disable_topology_aware_anchors,
                        ablation_disable_proximity_aware_anchors=args.ablation_disable_proximity_aware_anchors,
                        ablation_no_homogeneity_score=args.ablation_no_homogeneity_score,
                        ablation_no_cl=args.ablation_no_cl,
                        ablation_no_temp_scale=args.ablation_no_temp_scale,
                        ablation_combine_topk=args.ablation_combine_topk,
                        ablation_topk_from_all=args.ablation_topk_from_all,
                        ablation_mlp_sim=args.ablation_mlp_sim,
                    )
                    simple_result, total_results = run_single_exp(data, method_name, noise_type=noise_type,
                                                                  noise_rate=noise_rate,
                                                                  seed=args.seed + run, device=args.device, debug=False,
                                                                  plot_homogeneity_scores_arg=args.plot,
                                                                  ablation_args=ablation_settings,
                                                                  k_topology_override=args.k_topology_override,
                                                                  k_proximity_override=args.k_proximity_override)
                    logger.add_result(run, total_results)
                total_results = logger.get_statistics()
                result_recorder.dump_record(method_name, data_name, noise_type, noise_rate, total_results)
