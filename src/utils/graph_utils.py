import torch
from tqdm import tqdm
from .dict_utils import iterate_nested_dict

def get_degree_stats(graphs, use_tqdm=False, degree_type='deg'):
    """
    Calculate degree statistics for a list of graphs.

    Parameters:
    graphs (list): List of graph objects.
    use_tqdm (bool): Whether to use tqdm for progress bar.
    degree_type (str): Type of degree to calculate ('deg', 'in', 'out', 'all').

    Returns:
    dict: Dictionary containing degree statistics.
    """
    def _get_stats(degrees):
        degrees = degrees.to(torch.float)
        statistics = {
            'mean': torch.mean(degrees).item(),
            'max': torch.max(degrees).item(),
            'min': torch.min(degrees).item(),
            'var': torch.var(degrees).item(),
            'std': torch.std(degrees).item(),
            'Q1': torch.quantile(degrees, 0.25).item(),
            'Q2': torch.median(degrees).item(),
            'Q3': torch.quantile(degrees, 0.75).item()
        }
        return statistics

    degree_type_options = ['deg', 'in', 'out', 'all']
    assert degree_type in degree_type_options, f"Expect degree_type in {degree_type_options}, but got {degree_type_options}."
    if use_tqdm:
        graphs = tqdm(graphs)

    in_degrees = []
    out_degrees = []
    for _g in graphs:
        _in_deg = _g.in_degrees()
        _out_deg = _g.out_degrees()
        in_degrees.append(_in_deg)
        out_degrees.append(_out_deg)

    in_degrees = torch.cat(in_degrees, dim=0)
    out_degrees = torch.cat(out_degrees, dim=0)

    if degree_type == 'in':
        return _get_stats(in_degrees)
    elif degree_type == 'out':
        return _get_stats(out_degrees)
    elif degree_type == 'deg':
        return _get_stats(in_degrees + out_degrees)
    else:
        return {
            'in-degree': _get_stats(in_degrees),
            'out-degree': _get_stats(out_degrees),
            'degree': _get_stats(in_degrees + out_degrees),
        }

def get_graph_info(g):
    """
    Get basic information and degree statistics of a graph.

    Parameters:
    g (object): Graph object.

    Returns:
    dict: Dictionary containing number of nodes, number of edges, and degree statistics.
    """
    num_nodes = g.number_of_nodes()
    num_edges = g.number_of_edges()
    degree_stats = iterate_nested_dict(get_degree_stats([g], degree_type='all'))
    return {'num_nodes': num_nodes, 'num_edges': num_edges, **degree_stats}

def get_canonical_etypes_set(all_hetero_graphs):
    """
    Get the set of canonical edge types from a list of heterogeneous graphs.

    Args:
        all_hetero_graphs (list): List of heterogeneous graphs.

    Returns:
        set: A set of canonical edge types.
    """
    canonical_etypes_set = set()
    for g in all_hetero_graphs:
        canonical_etypes_set = canonical_etypes_set.union(set(g.canonical_etypes))
    return canonical_etypes_set

