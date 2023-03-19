import os

import json
import torch
import util
import torch_geometric.utils as torch_utils
import networkx as nx
import random

from tqdm import tqdm
import argparse
import importance

import numpy as np
from data import load_dataset
from gnn import load_trained_gnn, load_trained_prediction

from proposal import Proposal_Random
from util import MolFromGraphs


VALIDCHECK=True
ROOT="results/test5w"

graph_map = {}  # graph_hash -> {edge_index, x}
graph_index_map = {}  # graph hash -> index in counterfactual_graphs
counterfactual_candidates = []  # [{frequency: int, graph_hash: str, importance_parts: tuple, input_graphs_covering_indexes: set}]
input_graphs_covered = []  # [int] with of number of input graphs
covering_graphs = set()  # dictionary graph hash which is in first #number input graph counterfactual list (i.e., contributing input_graph_covered)
transitions = {}  # graph_hash -> {transitions ([hashes], [actions], [importance_parts], tensor(input_graph_covering_for_all_neighbours))}

MAX_COUNTERFACTUAL_SIZE = 0
starting_step = 1

traversed_hashes = []  # list of traversed graph hashes
node_mapping = {}


def get_args():
    parser = argparse.ArgumentParser(description='Graph Global Counterfactual Summary')
    parser.add_argument('--dataset', type=str, default='mutagenicity', choices=['mutagenicity', 'aids', 'nci1', 'proteins'])
    parser.add_argument('--alpha', type=float, default=0.5, help='alpha value to balance individual and cumulative coverage')
    parser.add_argument('--theta', type=float, default=0.05, help='distance threshold value during training.')
    parser.add_argument('--teleport', type=float, default=0.1, help='teleport probability to input graphs')
    parser.add_argument('--max_steps', type=int, default=50000, help='random walk step size')
    parser.add_argument('--k', type=int, default=100000, help='number of graphs will be selected from counterfactuals')
    parser.add_argument('--device1', type=str, help='Cuda device or cpu for gnn model', default='0')
    parser.add_argument('--device2', type=str, help='Cuda device or cpu for neurosed model', default='0')
    parser.add_argument('--sample_size', type=int, help='Sample count for neighbour graphs', default=10000)
    parser.add_argument('--sample', action='store_true')
    return parser.parse_args()


def calculate_hash(graph_embedding):
    if isinstance(graph_embedding, (np.ndarray,)):
        return hash(graph_embedding.tobytes())
    else:
        raise Exception('graph_embedding should be ndarray')


def is_counterfactual_array_full():
    return len(counterfactual_candidates) >= MAX_COUNTERFACTUAL_SIZE


def get_minimum_frequency():
    return counterfactual_candidates[-1]['frequency']


def is_graph_counterfactual(graph_hash):
    return counterfactual_candidates[graph_index_map[graph_hash]]['importance_parts'][0] >= 0.5


def reorder_counterfactual_candidates(start_idx):
    """
        sort the candidates by their frequency, from large to small
        start_idx: the idx of candidate that just increase its frequency, so the reorder only affect [0, start_idx]
    """
    swap_idx = start_idx - 1
    while swap_idx >= 0 and counterfactual_candidates[start_idx]['frequency'] > counterfactual_candidates[swap_idx]['frequency']:
        swap_idx -= 1
    swap_idx += 1
    if swap_idx < start_idx:
        graph_index_map[counterfactual_candidates[start_idx]['graph_hash']] = swap_idx
        graph_index_map[counterfactual_candidates[swap_idx]['graph_hash']] = start_idx
        counterfactual_candidates[start_idx], counterfactual_candidates[swap_idx] = counterfactual_candidates[swap_idx], counterfactual_candidates[start_idx]
    return swap_idx


def update_input_graphs_covered(add_graph_covering_list=None, remove_graph_covering_list=None):
    global input_graphs_covered
    if add_graph_covering_list is not None:
        input_graphs_covered += add_graph_covering_list
    if remove_graph_covering_list is not None:
        input_graphs_covered -= remove_graph_covering_list

# TODO: add domain constraint here
def check_reinforcement_condition(graph_hash):
    if VALIDCHECK:
        return is_graph_counterfactual(graph_hash) and domain_check(graph_hash)
    else:
        return is_graph_counterfactual(graph_hash)

def domain_check(graph_hash):
    graph_can = graph_map[graph_hash]
    return util.valid_checking(graph_can, node_mapping)


def populate_counterfactual_candidates(graph_hash, importance_parts, input_graphs_covering_list):
    is_new_graph = False
    if graph_hash in graph_index_map:
        graph_idx = graph_index_map[graph_hash]
        condition = check_reinforcement_condition(graph_hash)
        # only that satisfied condition will update the frequency
        if condition:
            # update the visit to candidate N(v)
            counterfactual_candidates[graph_idx]['frequency'] += 1
            swap_idx = reorder_counterfactual_candidates(graph_idx)
        else:
            swap_idx = graph_idx
    else:
        is_new_graph = True
        # new graph will be added to counterfactual_candidates anyway
        if is_counterfactual_array_full():
            deleting_graph_hash = counterfactual_candidates[-1]['graph_hash']
            del graph_index_map[deleting_graph_hash]
            del graph_map[deleting_graph_hash]
            if deleting_graph_hash in transitions:
                del transitions[deleting_graph_hash]
            counterfactual_candidates[-1] = {
                "frequency": get_minimum_frequency() + 1,
                "graph_hash": graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            }
        else:
            counterfactual_candidates.append({
                'frequency': 2,         # each candidate will at least has frequency 2
                'graph_hash': graph_hash,
                "importance_parts": importance_parts,
                "input_graphs_covering_list": input_graphs_covering_list
            })
        graph_idx = len(counterfactual_candidates) - 1
        graph_index_map[graph_hash] = graph_idx
        swap_idx = reorder_counterfactual_candidates(graph_idx)

    # updating input_graphs_covered entries
    if swap_idx == graph_idx:  # no swap
        if is_new_graph and graph_idx < len(input_graphs_covered) and check_reinforcement_condition(graph_hash):
            update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
            covering_graphs.add(graph_hash)
    else:  # swapped graph_idx position has swapped graph now
        swapped_graph = counterfactual_candidates[graph_idx]
        if check_reinforcement_condition(swapped_graph['graph_hash']) and graph_idx >= len(input_graphs_covered) > swap_idx:
            update_input_graphs_covered(remove_graph_covering_list=swapped_graph['input_graphs_covering_list'])
            covering_graphs.remove(swapped_graph['graph_hash'])
        if is_new_graph:
            if check_reinforcement_condition(graph_hash) and swap_idx < len(input_graphs_covered):
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)
        else:
            if check_reinforcement_condition(graph_hash) and swap_idx < len(input_graphs_covered) <= graph_idx:
                update_input_graphs_covered(add_graph_covering_list=input_graphs_covering_list)
                covering_graphs.add(graph_hash)


def calculate_importance(hashes, importances, coverage_matrices, importance_args):
    cum_coverage = None
    ind_coverage = None
    if importance_args['alpha'] < 1:
        # cumulative coverage
        belong = torch.Tensor([hash_ in covering_graphs for hash_ in hashes])
        support = coverage_matrices.to_dense() + (coverage_matrices.to_dense().T * belong).T - input_graphs_covered
        cum_selected = torch.maximum(torch.zeros(input_graphs_covered.shape), support)
        cum_coverage = cum_selected.sum(dim=1) / input_graphs_covered.shape[0]
        cum_coverage = cum_coverage.numpy()
        importances[:, 1] = cum_coverage
    if importance_args['alpha'] > 0:
        # individual coverage
        ind_coverage = coverage_matrices.to_dense().sum(dim=1) / input_graphs_covered.shape[0]
        ind_coverage = ind_coverage.numpy()
        importances[:, 1] = ind_coverage

    importances[:, 1] = alpha * ind_coverage + (1 - alpha) * cum_coverage
    if importances[:, 1].sum() == 0:  # all coverage values are zero, we will only use prediction as importance
        importance_values = importances[:, 0]
    else:
        importance_values = np.prod(importances, axis=1)
    return importance_values


def move_from_known_graph(hashes, importances, coverage_matrices, importance_args):
    probabilities = []
    importance_values = calculate_importance(hashes, np.array(importances), coverage_matrices, importance_args)
    for i, hash_i in enumerate(hashes):
        importance_value = importance_values[i]
        if hash_i in graph_index_map:  # and is_graph_counterfactual(hash_i):  # reinforcing only seen counterfactuals
            frequency = counterfactual_candidates[graph_index_map[hash_i]]['frequency']
        else:
            frequency = get_minimum_frequency() if is_counterfactual_array_full() else 1
        probabilities.append(importance_value * frequency)

    if sum(probabilities) == 0:  # if probability values are all 0, we assign equal probs to all transitions
        probabilities = np.ones(len(probabilities)) / len(probabilities)
    else:
        probabilities = np.array(probabilities) / sum(probabilities)
    selected_hash_idx = random.choices(range(len(hashes)), weights=probabilities)[0]
    return selected_hash_idx


def move_to_next_graph(graph_hash, editor, importance_args, teleport_probability):
    graph = graph_map[graph_hash]
    not_teleport = False
    if random.uniform(0, 1) < teleport_probability:  # teleport to start
        return None, not not_teleport
    else:
        if graph_hash in transitions:
            target_graphs_hashes, target_graphs_actions, target_graphs_importance_parts, target_graphs_coverage_matrix = transitions[graph_hash]
            selected_hash_idx = move_from_known_graph(target_graphs_hashes, target_graphs_importance_parts, target_graphs_coverage_matrix, importance_args)

        else:  # uncalculated transitions for a graph
            new_mols, neighbor_graphs_actions = editor.enumerate_from_graph(graph)
            neighbor_graphs = editor.convert_to_geo(new_mols)

            if len(neighbor_graphs) > sample_size and is_sample:
                samples = random.sample(range(len(neighbor_graphs)), sample_size)
                neighbor_graphs_actions = [neighbor_graphs_actions[sample] for sample in samples]
                neighbor_graphs = [neighbor_graphs[sample] for sample in samples]

            if len(neighbor_graphs) == 0:
                return graph_hash, not_teleport
            neighbor_graphs_importance_parts, neighbor_graphs_embeddings, neighbor_graphs_coverage_matrix = importance.call(neighbor_graphs, importance_args)
            target_graphs_coverage_matrix = torch.cat([counterfactual_candidates[graph_index_map[graph_hash]]['input_graphs_covering_list'].unsqueeze(0), neighbor_graphs_coverage_matrix.to_sparse()])

            target_graphs_set = {graph_hash}
            target_graphs_hashes = [graph_hash]
            target_graphs_actions = [(None, None, None, None)]
            target_graphs_importance_parts = [counterfactual_candidates[graph_index_map[graph_hash]]['importance_parts']]
            needed_i = []
            for i in range(len(neighbor_graphs_embeddings)):
                graph_neighbour_hash = calculate_hash(neighbor_graphs_embeddings[i])
                if graph_neighbour_hash not in target_graphs_set:
                    needed_i.append(i)
                    target_graphs_importance_parts.append(neighbor_graphs_importance_parts[i])
                    target_graphs_hashes.append(graph_neighbour_hash)
                    target_graphs_set.add(graph_neighbour_hash)
                    target_graphs_actions.append(neighbor_graphs_actions[i])
            target_graphs_coverage_matrix = torch.cat([counterfactual_candidates[graph_index_map[graph_hash]]['input_graphs_covering_list'].unsqueeze(0), neighbor_graphs_coverage_matrix[needed_i].to_sparse()])

            selected_hash_idx = move_from_known_graph(target_graphs_hashes, target_graphs_importance_parts, target_graphs_coverage_matrix, importance_args)

            # update transition part of cur_graph
            transitions[graph_hash] = (target_graphs_hashes, target_graphs_actions, target_graphs_importance_parts, target_graphs_coverage_matrix)


        selected_hash = target_graphs_hashes[selected_hash_idx]
        selected_action = target_graphs_actions[selected_hash_idx]
        selected_importance_parts = target_graphs_importance_parts[selected_hash_idx]
        selected_graph = editor.action_on_graph(graph, selected_action)

        if selected_hash not in graph_map:
            selected_input_graphs_covering_list = target_graphs_coverage_matrix[selected_hash_idx]
            graph_map[selected_hash] = selected_graph  # next graph addition to memory
        else:
            selected_input_graphs_covering_list = counterfactual_candidates[graph_index_map[selected_hash]]['input_graphs_covering_list']
        populate_counterfactual_candidates(selected_hash, selected_importance_parts, selected_input_graphs_covering_list)

        return selected_hash, not_teleport


def dynamic_teleportation_probabilities():
    input_graphs_covered_exp = np.exp(input_graphs_covered)
    return (1 / input_graphs_covered_exp) / (1 / input_graphs_covered_exp).sum()


def restart_randomwalk(input_graphs):
    dynamic_probs = dynamic_teleportation_probabilities()
    idx = random.choices(range(dynamic_probs.shape[0]), weights=dynamic_probs)[0]
    graph = input_graphs[idx]
    importance_parts, graph_embeddings, coverage_matrix = importance.call([graph], importance_args)
    input_graphs_covering_list = coverage_matrix[0].to_sparse()
    graph_hash = calculate_hash(graph_embeddings[0])
    if graph_hash not in graph_index_map:
        graph_map[graph_hash] = graph
    populate_counterfactual_candidates(graph_hash, importance_parts[0], input_graphs_covering_list)
    return graph_hash


def counterfactual_summary_with_randomwalk(input_graphs, editor, importance_args, teleport_probability, max_steps):
    cur_graph_hash = restart_randomwalk(input_graphs)
    for step in tqdm(range(starting_step, max_steps + 1)):
        traversed_hashes.append(cur_graph_hash)
        next_graph_hash, is_teleported = move_to_next_graph(graph_hash=cur_graph_hash,
                                                            editor = editor,
                                                            importance_args=importance_args,
                                                            teleport_probability=teleport_probability)

        cur_graph_hash = restart_randomwalk(input_graphs) if is_teleported else next_graph_hash

        # checking if memory is handled well
        assert len(graph_map) == len(graph_index_map) == len(counterfactual_candidates)  # == len(transitions) - len(input_graphs)
        assert set(graph_index_map.keys()) == set(graph_map.keys())

    save_item = {
        'graph_map': graph_map,
        'graph_index_map': graph_index_map,
        'counterfactual_candidates': counterfactual_candidates,
        'MAX_COUNTERFACTUAL_SIZE': MAX_COUNTERFACTUAL_SIZE,
        'traversed_hashes': traversed_hashes,
        'input_graphs_covered': input_graphs_covered,
    }
    if not os.path.exists(f'{ROOT}/{dataset_name}/runs/'):
        os.makedirs(f'{ROOT}/{dataset_name}/runs/')
    torch.save(save_item, f'{ROOT}/{dataset_name}/runs/counterfactuals.pt')


def prepare_devices(device1, device2):
    device1 = 'cuda:' + device1 if torch.cuda.is_available() and device1 in ['0', '1', '2', '3'] else 'cpu'
    device2 = 'cuda:' + device2 if torch.cuda.is_available() and device2 in ['0', '1', '2', '3'] else 'cpu'

    return device1, device2


if __name__ == '__main__':
    np.random.seed(0)
    random.seed(0)
    torch.manual_seed(0)

    args = get_args()

    device1, device2 = prepare_devices(args.device1, args.device2)

    teleport_probability = args.teleport
    max_steps = args.max_steps
    dataset_name = args.dataset
    alpha = args.alpha
    if alpha > 1 or alpha < 0:
        raise Exception('Alpha cannot be bigger than 1, or smaller than 0!')
    sample_size = args.sample_size
    is_sample = args.sample

    # global MAX_COUNTERFACTUAL_SIZE
    MAX_COUNTERFACTUAL_SIZE = args.k

    # Load dataset
    graphs = load_dataset(dataset_name)

    # Load node_mapping
    mapping_info = json.load(open('data/{}/raw/mapping_info.json'.format(dataset_name)))
    node_mapping = mapping_info['keep_node_mapping']

    # Load GNN model for dataset
    gnn_model = load_trained_gnn(dataset_name, device=device1)
    gnn_model.eval()

    # Load prediction based on model
    preds = load_trained_prediction(dataset_name, device=device1)
    preds = preds.cpu().numpy()
    input_graph_indices = np.array(range(len(preds)))[preds == 0]
    input_graphs = graphs[input_graph_indices.tolist()]

    # setting covered graph numbers to 0
    input_graphs_covered = torch.zeros(len(input_graphs), dtype=torch.float)

    importance_args = importance.prepare_and_get(graphs, gnn_model, input_graph_indices, args.alpha, args.theta, device1=device1, device2=device2, dataset_name=dataset_name)

    # setting proposal
    max_size = 40
    max_sample = 10
    data_dir = "/home/danqingwang/workspace/GCFExplainer/"
    vocab_name = "AIDS_1837"
    vocab_size=1000
    random_editor = Proposal_Random(data_dir, vocab_name, vocab_size, max_size, max_sample, node_mapping)

    # graphs with adjacency matrix and feature matrix
    counterfactual_summary_with_randomwalk(input_graphs=input_graphs,
                                           editor=random_editor,
                                           importance_args=importance_args,
                                           teleport_probability=teleport_probability,
                                           max_steps=max_steps)
