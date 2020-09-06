from load_graph import generate_graph
import numpy as np
from graph_nets import graphs
from graph_nets import utils_np
from graph_nets import utils_tf
from graph_nets.demos import models
import collections
import networkx as nx
from collections import Counter
import tensorflow as tf

BOITYPENUM = 23
VESTYPENUM = 25

def to_one_hot(indices, max_value, axis=-1):
    one_hot = np.eye(max_value)[indices]
    if axis not in (-1, one_hot.ndim):
        one_hot = np.moveaxis(one_hot, -1, axis)
    return one_hot


def generate_networkx_graphs(graphcache,all_db,rand,num_examples,dataset):
    """Generate graphs for training.
 
    Args:
        rand: A random seed (np.RandomState instance).
        num_examples: Total number of graphs to generate.
        dataset: 'train', 'val', 'test'
    Returns:
        input_graphs: The list of input graphs.
        target_graphs: The list of output graphs.
        graphs: The list of generated graphs.
    """
    input_graphs = []
    target_graphs = []
    graphs = []
    
    totnum = len(all_db[dataset])
    if totnum==num_examples:
        selids = np.arange(totnum)
    else:
        selids = rand.choice(totnum,num_examples)
    for ni in range(num_examples):
        #print(all_db[dataset][selids[ni]])
        if dataset == 'test':
            tt = True
            da = False
        elif dataset == 'val':
            tt = False
            da = False
        else:
            tt = False
            da = True
        graph = generate_graph(graphcache,all_db[dataset][selids[ni]],rand,data_aug=da,test=tt)
        #graph = add_shortest_path(rand, graph)
        input_graph, target_graph = graph_to_input_target(graph)
        input_graphs.append(input_graph)
        target_graphs.append(target_graph)
        graphs.append(graph)
    return input_graphs, target_graphs, graphs, selids

def create_placeholders(graphcache,rand,batch_size,all_db):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
    Returns:
        input_ph: The input graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.
    """
    # Create some example data for inspecting the vector sizes.
    input_graphs, target_graphs, _, _ = generate_networkx_graphs(graphcache,all_db, rand, batch_size,'train')
    input_ph = utils_tf.placeholders_from_networkxs(input_graphs)
    target_ph = utils_tf.placeholders_from_networkxs(target_graphs)
    return input_ph, target_ph

def graph_to_input_target(graph):
    """Returns 2 graphs with input and target feature vectors for training.

    Args:
        graph: An `nx.DiGraph` instance.

    Returns:
        The input `nx.DiGraph` instance.
        The target `nx.DiGraph` instance.

    Raises:
        ValueError: unknown node type
    """
    
    def create_feature(attr, fields):
        return np.hstack([np.array(attr[field], dtype=float) for field in fields])

    input_node_fields = ("pos","rad","deg","dir")#
    input_edge_fields = ("rad","dist","dir")#
    target_node_fields = ("boitype",)
    target_edge_fields = ("vestype",)#"vestype",

    input_graph = graph.copy()
    target_graph = graph.copy()

    #solution_length = 0
    for node_index, node_feature in graph.nodes(data=True):
        input_graph.add_node(
                node_index, features=create_feature(node_feature, input_node_fields))
        target_node = to_one_hot(
                create_feature(node_feature, target_node_fields).astype(int), BOITYPENUM)[0]
        target_graph.add_node(node_index, features=target_node)
        #solution_length += int(node_feature["solution"])
    #solution_length /= graph.number_of_nodes()

    for receiver, sender, features in graph.edges(data=True):
        input_graph.add_edge(
                sender, receiver, features=create_feature(features, input_edge_fields))
        target_edge = create_feature(features, target_edge_fields)#.astype(int)
        #to_one_hot()[0]
        target_graph.add_edge(sender, receiver, features=target_edge)

    input_graph.graph["features"] = np.array([0.0])
    target_graph.graph["features"] = np.array([0.0], dtype=float)

    return input_graph, target_graph

def create_feed_dict(all_db, rand, batch_size, input_ph, target_ph, dataset, graphcache):
    """Creates placeholders for the model training and evaluation.

    Args:
        rand: A random seed (np.RandomState instance).
        batch_size: Total number of graphs per batch.
        input_ph: The input graph's placeholders, as a graph namedtuple.
        target_ph: The target graph's placeholders, as a graph namedtuple.
        dataset: 'train', 'val', 'test'
    Returns:
        feed_dict: The feed `dict` of input and target placeholders and data.
        raw_graphs: The `dict` of raw networkx graphs.
        
    """
    inputs, targets, raw_graphs, selids = generate_networkx_graphs(graphcache, all_db, rand, batch_size, dataset)
    input_graphs = utils_np.networkxs_to_graphs_tuple(inputs)
    target_graphs = utils_np.networkxs_to_graphs_tuple(targets)
    feed_dict = {input_ph: input_graphs, target_ph: target_graphs}
    return feed_dict, raw_graphs, selids


def pairwise(iterable):
    """s -> (s0,s1), (s1,s2), (s2, s3), ..."""
    a, b = itertools.tee(iterable)
    next(b, None)
    return zip(a, b)

def set_diff(seq0, seq1):
    """Return the set difference between 2 sequences as a list."""
    return list(set(seq0) - set(seq1))

def get_node_dict(graph, attr, ignoreaxis = 2):
    """Return a `dict` of node:attribute pairs from a graph."""
    if ignoreaxis==2:
        return {k: v[attr][:2] for k, v in graph.nodes.items()}
    elif ignoreaxis==0:
        return {k: v[attr][1:] for k, v in graph.nodes.items()}
    elif ignoreaxis==1:
        return {k: [v[attr][0],v[attr][2]] for k, v in graph.nodes.items()}
    else:
        print('wrong axis')


def compute_accuracy(target, output, use_nodes=True, use_edges=False):
    """Calculate model accuracy.

    Returns the number of correctly predicted shortest path nodes and the number
    of completely solved graphs (100% correct predictions).

    Args:
        target: A `graphs.GraphsTuple` that contains the target graph.
        output: A `graphs.GraphsTuple` that contains the output graph.
        use_nodes: A `bool` indicator of whether to compute node accuracy or not.
        use_edges: A `bool` indicator of whether to compute edge accuracy or not.

    Returns:
        correct: A `float` fraction of correctly labeled nodes/edges.
        solved: A `float` fraction of graphs that are completely correctly labeled.

    Raises:
        ValueError: Nodes or edges (or both) must be used
    """
    if not use_nodes and not use_edges:
        raise ValueError("Nodes or edges (or both) must be used")
    tdds = utils_np.graphs_tuple_to_data_dicts(target)
    odds = utils_np.graphs_tuple_to_data_dicts(output)
    cs = []
    ss = []
    for td, od in zip(tdds, odds):
        xn = np.argmax(td["nodes"], axis=-1)
        yn = np.argmax(od["nodes"], axis=-1)
        xe = np.argmax(td["edges"], axis=-1)
        ye = np.argmax(od["edges"], axis=-1)
        c = []
        if use_nodes:
            c.append(xn == yn)
        if use_edges:
            c.append(xe == ye)
        c = np.concatenate(c, axis=0)
        s = np.all(c)
        cs.append(c)
        ss.append(s)
    correct = np.mean(np.concatenate(cs, axis=0))
    solved = np.mean(np.stack(ss))
    return correct, solved


def create_loss_ops(target_op, output_ops, node_class_weight=None, edge_class_weight=None, weighted = False):
    if weighted:
        assert node_class_weight is not None and edge_class_weight is not None
        node_class_weight_tf = tf.constant(node_class_weight)
        nodeweights = tf.reduce_sum(node_class_weight_tf * target_op.nodes, axis=1)
        edge_class_weight_tf = tf.constant(edge_class_weight)
        edgeweights = tf.reduce_sum(edge_class_weight_tf * target_op.edges, axis=1)
        loss_ops = [
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_op.nodes, logits=output_op.nodes)*nodeweights) +
                tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=target_op.edges, logits=output_op.edges)*edgeweights)
                for output_op in output_ops
        ]
    else:
        loss_ops = [
                tf.losses.softmax_cross_entropy(target_op.nodes, output_op.nodes) +
                tf.losses.softmax_cross_entropy(target_op.edges, output_op.edges)
                for output_op in output_ops
        ]
    return loss_ops



def make_all_runnable_in_session(*args):
    """Lets an iterable of TF graphs be output from a session as NP graphs."""
    return [utils_tf.make_runnable_in_session(a) for a in args]

class GraphPlotter(object):

    def __init__(self, ax, graph, pos):
        self._ax = ax
        self._graph = graph
        self._pos = pos
        self._base_draw_kwargs = dict(G=self._graph, pos=self._pos, ax=self._ax)
        self._nodes = None
        self._edges = None
        self._solution_nodes = None
        self._solution_edges = None
        self._non_solution_nodes = None
        self._non_solution_edges = None
        self._ax.set_axis_off()
        self.title = ''

    @property
    def nodes(self):
        if self._nodes is None:
            self._nodes = self._graph.nodes()
        return self._nodes

    @property
    def edges(self):
        if self._edges is None:
            self._edges = self._graph.edges()
        return self._edges

    @property
    def solution_nodes(self):
        if self._solution_nodes is None:
            self._solution_nodes = [
                    n for n in self.nodes if self._graph.nodes[n].get("boitype", False)>0
            ]
        return self._solution_nodes

    @property
    def solution_edges(self):
        if self._solution_edges is None:
            self._solution_edges = [
                    e for e in self.edges
                    if np.argmax(self._graph.get_edge_data(e[0], e[1]).get("vestype", False))>0
            ]
        return self._solution_edges

    @property
    def non_solution_nodes(self):
        if self._non_solution_nodes is None:
            self._non_solution_nodes = [
                    n for n in self.nodes
                    if self._graph.nodes[n].get("boitype", False)==0
            ]
        return self._non_solution_nodes

    @property
    def non_solution_edges(self):
        if self._non_solution_edges is None:
            self._non_solution_edges = [
                    e for e in self.edges
                    if np.argmax(self._graph.get_edge_data(e[0], e[1]).get("vestype", False))==0
            ]
        return self._non_solution_edges

    def _make_draw_kwargs(self, **kwargs):
        kwargs.update(self._base_draw_kwargs)
        return kwargs

    def _draw(self, draw_function, zorder=None, **kwargs):
        draw_kwargs = self._make_draw_kwargs(**kwargs)
        collection = draw_function(**draw_kwargs)
        if collection is not None and zorder is not None:
            try:
                # This is for compatibility with older matplotlib.
                collection.set_zorder(zorder)
            except AttributeError:
                # This is for compatibility with newer matplotlib.
                collection[0].set_zorder(zorder)
        return collection

    def draw_nodes(self, **kwargs):
        """Useful kwargs: nodelist, node_size, node_color, linewidths."""
        if ("node_color" in kwargs and
                isinstance(kwargs["node_color"], collections.Sequence) and
                len(kwargs["node_color"]) in {3, 4} and
                not isinstance(kwargs["node_color"][0],
                                             (collections.Sequence, np.ndarray))):
            num_nodes = len(kwargs.get("nodelist", self.nodes))
            kwargs["node_color"] = np.tile(
                    np.array(kwargs["node_color"])[None], [num_nodes, 1])
        return self._draw(nx.draw_networkx_nodes, **kwargs)

    def draw_edges(self, **kwargs):
        """Useful kwargs: edgelist, width."""
        return self._draw(nx.draw_networkx_edges, **kwargs)

    def draw_graph(self,
                                 node_size=200,
                                 node_color=(0.4, 0.8, 0.4),
                                 node_linewidth=1.0,
                                 edge_width=1.0):
        # Plot nodes.
        self.draw_nodes(
                nodelist=self.nodes,
                node_size=node_size,
                node_color=node_color,
                linewidths=node_linewidth,
                zorder=20)
        # Plot edges.
        self.draw_edges(edgelist=self.edges, width=edge_width, zorder=10)

    def draw_graph_with_solution(self,
                                                             node_size=100,
                                                             node_color=(0.4, 0.8, 0.4),
                                                             edge_color=None,
                                                             node_linewidth=1.0,
                                                             edge_width=1.0,
                                                             solution_node_linewidth=1.5,
                                                             solution_edge_width=1.5):
        node_border_color = (0.0, 0.0, 0.0, 1.0)
        node_collections = {}
        
        # Plot solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.solution_nodes]
        else:
            c = node_color
        node_collections["solution nodes"] = self.draw_nodes(
                nodelist=self.solution_nodes,
                node_size=node_size,
                node_color=c,
                linewidths=node_linewidth,
                edgecolors=node_border_color,
                node_shape='^',
                zorder=40)
        
        # Plot solution edges.
        if isinstance(edge_color, dict):
            c = [edge_color[n] for n in self.solution_edges]
        else:
            c = 'k'
        node_collections["solution edges"] = self.draw_edges(
                edgelist=self.solution_edges, width=solution_edge_width, edge_color=c, zorder=20)
        
        # Plot non-solution nodes.
        if isinstance(node_color, dict):
            c = [node_color[n] for n in self.non_solution_nodes]
        else:
            c = node_color
        node_collections["non-solution nodes"] = self.draw_nodes(
                nodelist=self.non_solution_nodes,
                node_size=node_size,
                node_color=c,
                linewidths=node_linewidth,
                edgecolors=node_border_color,
                zorder=30)
        
        # Plot non-solution edges.
        if isinstance(edge_color, dict):
            c = [edge_color[n] for n in self.non_solution_edges]
        else:
            c = 'k'
        node_collections["non-solution edges"] = self.draw_edges(
                edgelist=self.non_solution_edges, width=edge_width, edge_color=c, zorder=10)
        # Set title as solution length.
        self._ax.set_title(self.title)
        return node_collections
