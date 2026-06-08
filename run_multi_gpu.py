import argparse
import queue
import warnings
import time
import multiprocessing
import os
import torch  # Import torch early in main process if needed, but env var matters for children

# Import necessary functions and classes from your original script
from total_exp import run_single_exp  # Assuming your original code is in total_exp.py
from utils.dataloader import Dataset
from utils.tools import load_conf, setup_seed
from utils.logger import MultiExpRecorder, ResultLogger


def run_worker_persistent(gpu_id, task_queue, result_queue):
    warnings.filterwarnings("ignore", category=UserWarning)
    device = torch.device(f'cuda:{gpu_id}')
    os.environ['CUDA_VISIBLE_DEVICES'] = str(gpu_id)

    current_data = None
    current_data_key = None

    while True:
        try:
            task = task_queue.get()
            if task is None:
                break

            run_id = task['run_id']
            seed = task['seed']
            dataset_args = task['dataset_args']
            method_name = task['method_name']
            noise_type = task['noise_type']
            noise_rate = task['noise_rate']
            debug_flag = task['debug']
            plot_homogeneity_scores_flag = task.get('plot_homogeneity_scores', False)
            ablation_args_from_task = task.get('ablation_args', None)
            k_topology_override_from_task = task.get('k_topology_override', None)
            k_proximity_override_from_task = task.get('k_proximity_override', None)

            data_key = (frozenset(dataset_args.items()), method_name, noise_type, noise_rate)

            if data_key != current_data_key:
                try:
                    args_for_dataset_call = dict(dataset_args)
                    data_name_arg = args_for_dataset_call.pop('name')
                    current_data = Dataset(data_name_arg, **args_for_dataset_call, device=device)
                    current_data_key = data_key
                except Exception as load_e:
                    print(f"[Worker {os.getpid()} on GPU {gpu_id}] Error loading data: {load_e}")
                    result_queue.put({'run_id': run_id, 'error': f'Data loading failed: {load_e}'})
                    current_data = None
                    current_data_key = None
                    continue

            if current_data is None:
                print(f"[Worker {os.getpid()} on GPU {gpu_id}] Error: Data is None for run {run_id}. Skipping.")
                result_queue.put({'run_id': run_id, 'error': 'Data was None'})
                continue

            ablation_namespace_for_exp = None
            if ablation_args_from_task:
                ablation_namespace_for_exp = argparse.Namespace(**ablation_args_from_task)

            setup_seed(seed)
            try:
                _, extended_result = run_single_exp(
                    dataset=current_data,
                    method_name=method_name,
                    seed=seed,
                    noise_type=noise_type,
                    noise_rate=noise_rate,
                    device=device,
                    debug=debug_flag,
                    plot_homogeneity_scores_arg=plot_homogeneity_scores_flag,
                    ablation_args=ablation_namespace_for_exp,
                    k_topology_override=k_topology_override_from_task,
                    k_proximity_override=k_proximity_override_from_task
                )
                result_queue.put({'run_id': run_id, 'result': extended_result})
            except Exception as run_e:
                print(f"[Worker {os.getpid()} on GPU {gpu_id}] Error during run {run_id}: {run_e}")
                import traceback
                traceback.print_exc()
                result_queue.put({'run_id': run_id, 'error': str(run_e)})

        except queue.Empty:
            time.sleep(0.1)
        except Exception as worker_e:
            print(f"[Worker {os.getpid()} on GPU {gpu_id}] Unhandled exception: {worker_e}")


if __name__ == '__main__':
    try:
        multiprocessing.set_start_method('spawn', force=True)
        print("Multiprocessing start method set to 'spawn'.")
    except RuntimeError:
        print("Multiprocessing start method already set or could not be set.")
        pass

    parser = argparse.ArgumentParser()
    parser.add_argument('--runs', type=int,
                        default=10,
                        help="Number of experiments (total runs) for each combination of method and data")
    parser.add_argument('--methods', type=str, nargs='+',
                        default=['dream'],
                        choices=['gcn', 'gin', 'smodel', 'jocor', 'coteaching',
                                 'apl', 'sce', 'forward', 'backward', 'lcat', 'mlp',
                                 'nrgnn', 'rtgnn', 'cp', 'unionnet', 'cgnn', 'tss',
                                 'crgnn', 'clnode', 'rncgln', 'pignn', 'dgnn', 'r2lp',
                                 'dream'],
                        help='Select methods')
    parser.add_argument('--datasets', type=str, nargs='+',
                        default=['cora'],
                        choices=['cora', 'citeseer', 'pubmed', 'amazoncom', 'amazonpho',
                                 'dblp', 'blogcatalog', 'flickr', 'amazon-ratings', 'roman-empire'],
                        help='Select datasets')
    parser.add_argument('--noise_type', type=str, nargs='+',
                        default=['uniform'],
                        choices=['clean', 'pair', 'uniform', 'random'], help='Noise type')
    parser.add_argument('--noise_rate', type=float, nargs='+',
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help='Noise rate')
    parser.add_argument('--seed', type=int,
                        default=3000, help="Base Random Seed (run seeds will be base_seed + run_index)")

    parser.add_argument('--gpu_ids', type=int, nargs='+',
                        default=[0],
                        help='List of GPU IDs to use (e.g., 0 1 2)')
    parser.add_argument('--procs_per_gpu', type=int,
                        default=1,
                        help='Maximum number of processes to run concurrently on each GPU')

    parser.add_argument('--plot_homogeneity_scores', action='store_true',
                        help="Plot Homogeneity Score evolution for clean/noisy training nodes (only for dream).")

    parser.add_argument('--k_topology_override', type=int, default=None,
                        help="Override k_topology for DREAM predictor. Takes precedence over config.")
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
    warnings.filterwarnings("ignore", category=UserWarning)

    data_path = './data/'
    method_list = args.methods
    data_list = args.datasets
    noise_type_list = args.noise_type
    noise_rate_list = args.noise_rate
    available_gpu_ids = args.gpu_ids
    max_procs_per_gpu = args.procs_per_gpu
    total_worker_slots = len(available_gpu_ids) * max_procs_per_gpu

    manager = multiprocessing.Manager()
    task_queue = manager.Queue()
    results_queue = manager.Queue()
    worker_pool = []

    worker_idx = 0
    for gpu_id in available_gpu_ids:
        for _ in range(max_procs_per_gpu):
            p = multiprocessing.Process(
                target=run_worker_persistent,
                args=(gpu_id, task_queue, results_queue),
                daemon=True
            )
            p.start()
            worker_pool.append(p)
            worker_idx += 1

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
            dataset_args = dict(
                verbose=False,
                name=data_name, path=data_path,
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
                split_type=data_conf.split['split_type']
            )

            for method_name in method_list:
                tasks_submitted_count = 0
                for run_id in range(args.runs):
                    current_seed = args.seed + run_id

                    ablation_settings_for_task = {
                        'ablation_disable_topology_aware_anchors': args.ablation_disable_topology_aware_anchors,
                        'ablation_disable_proximity_aware_anchors': args.ablation_disable_proximity_aware_anchors,
                        'ablation_no_homogeneity_score': args.ablation_no_homogeneity_score,
                        'ablation_no_cl': args.ablation_no_cl,
                        'ablation_no_temp_scale': args.ablation_no_temp_scale,
                        'ablation_combine_topk': args.ablation_combine_topk,
                        'ablation_topk_from_all': args.ablation_topk_from_all,
                        'ablation_mlp_sim': args.ablation_mlp_sim,
                        'k_topology_override': args.k_topology_override,
                        'k_proximity_override': args.k_proximity_override,
                    }

                    task_definition = {
                        'run_id': run_id,
                        'seed': current_seed,
                        'dataset_args': dataset_args,
                        'method_name': method_name,
                        'noise_type': noise_type,
                        'noise_rate': noise_rate,
                        'debug': False,
                        'plot_homogeneity_scores': args.plot_homogeneity_scores,
                        'ablation_args': ablation_settings_for_task,
                    }
                    task_queue.put(task_definition)
                    tasks_submitted_count += 1

                print(f"--- Collecting {args.runs} results for: {data_name}/{method_name}/{noise_type}@{noise_rate} ---")
                results_collected_dict = {}
                runs_processed_count = 0
                while runs_processed_count < args.runs:
                    try:
                        result_data = results_queue.get(timeout=300)
                        run_id = result_data['run_id']
                        results_collected_dict[run_id] = result_data
                        runs_processed_count += 1
                        if runs_processed_count % max(1, args.runs // 10) == 0:
                            print(f"  Collected {runs_processed_count}/{args.runs} results...")

                    except queue.Empty:
                        print(f"Warning: Timeout waiting for results ({runs_processed_count}/{args.runs} collected). Checking worker status...")
                        alive_workers = sum(1 for p in worker_pool if p.is_alive())
                        print(f"  Alive workers: {alive_workers}/{total_worker_slots}")
                        if alive_workers < total_worker_slots:
                            print("Error: Some workers may have died.")
                            break

                print(f"--- Aggregating results for {data_name}/{method_name}/{noise_type}@{noise_rate} ---")
                logger = MultiExpRecorder(runs=args.runs)
                successful_runs = 0
                for run_idx in range(args.runs):
                    result_data = results_collected_dict.get(run_idx)
                    if result_data and 'result' in result_data:
                        logger.add_result(run_idx, result_data['result'])
                        successful_runs += 1
                    elif result_data and 'error' in result_data:
                        print(f"  Run {run_idx} failed: {result_data['error']}")
                    else:
                        print(f"  Run {run_idx} has no result or missing data.")

                if successful_runs > 0:
                    total_results = logger.get_statistics()
                    print(f"  Aggregation complete. Successful runs: {successful_runs}/{args.runs}")
                    result_recorder.dump_record(method_name, data_name, noise_type, noise_rate, total_results)
                else:
                    print(f"  No successful runs for this combo. Skipping dump.")

    print("\n--- All experiments finished. Signaling workers to stop ---")
    for _ in range(total_worker_slots):
        task_queue.put(None)

    for i, p in enumerate(worker_pool):
        p.join(timeout=60)
        if p.is_alive():
            print(f"Warning: Worker {i} (PID: {p.pid}) did not terminate after 60s. Forcing.")
            p.terminate()

    print("--- Main process finished ---")
