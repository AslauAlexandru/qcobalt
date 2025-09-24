"""
...the Foundation was never about curating knowledge. It was about curating people.

~ Isaac Asimov, Foundation
"""

from collections.abc import Iterable, Mapping
from matplotlib import colors as colours, pyplot as plot
from numpy.typing import NDArray
from typing import Any, Hashable
import numpy
import networkx


class Graph(networkx.Graph):
	"""
	A NetworkX graph with built-in visualisation methods for optimisation problems.

	This class extends the standard NetworkX Graph class to include specialised
	visualisation methods for common combinatorial optimisation problems including
	MaxCut, graph colouring, and the Travelling Salesman Problem.
	"""

	def __init__(
		self,
		data: None
		| Iterable[Any]
		| Mapping[Any, dict[Any, dict[str, Any]]]
		| Mapping[Any, Iterable[Any]]
		| NDArray
		| networkx.Graph = None,
		config: dict[str, Any] | None = None,
		**attributes: Any,
	):
		"""
		Initialise a graph with edges, name, or graph attributes.

		Args:
			data: Input graph data in any format accepted by NetworkX.
				Can be None for an empty graph.
			config: Dictionary of visualisation options.
			**attributes: Keyword arguments for graph attributes as key-value pairs.
		"""
		super().__init__(data, **attributes)

		default_config = {
			"font_color": "whitesmoke",
			"node_size": 300,
			"width": 1.0,
			"font_size": 12,
			"font_weight": "normal",
		}

		if config is None:
			self.config = default_config
		else:
			self.config = {**default_config, **config}

	@property
	def max_cut_matrix(self) -> NDArray:
		"""Return the upper-triangular Q matrix for the max-cut QUBO as a NumPy array."""
		vertices = list(self.nodes)
		n = len(vertices)
		vertex_index = {vertex: index for (index, vertex) in enumerate(vertices)}
		Q = numpy.zeros((n, n), dtype=float)
		for u, v in self.edges:
			i = vertex_index[u]
			j = vertex_index[v]
			Q[i, i] -= 1.0
			Q[j, j] -= 1.0
			if i < j:
				Q[i, j] += 2.0
			else:
				Q[j, i] += 2.0
		return Q

	def cut_size(self, assignment: Mapping[Hashable, int | bool]) -> int:
		"""Return the number of edges crossing the cut described by ``assignment``."""
		missing = [node for node in self.nodes if node not in assignment]
		if missing:
			raise KeyError(f"Missing assignments for nodes: {missing}")
		return sum(
			1 for (u, v) in self.edges if bool(assignment[u]) != bool(assignment[v])
		)

	def display(self):
		"""
		Display the graph using Kamada-Kawai layout.

		This method provides a standard visualisation of the graph structure
		with labelled nodes and weighted edges. Edge weights are displayed
		if present in the graph data.
		"""
		node_positions = networkx.kamada_kawai_layout(self)
		networkx.draw_networkx(self, node_positions, **self.config)
		edge_labels = networkx.get_edge_attributes(self, "weight")
		if edge_labels:
			networkx.draw_networkx_edge_labels(self, node_positions, edge_labels)
		plot.axis("off")
		plot.show()

	def display_cut(self, cut: dict[Hashable, int] | set[Hashable]):
		"""
		Visualise a cut of the graph (for example, one produced when solving a Max-Cut instance).

		Nodes in the first partition are coloured red and the complementary set is coloured green.
		Cut edges appear as dashed blue lines and uncut edges remain solid.

		Args:
			cut: Either a dictionary mapping nodes to binary values where 1 marks one side of the cut,
				or a set containing the nodes in one partition.
		"""
		if isinstance(cut, dict):
			cut = {node for (node, value) in cut.items() if value == 1}
		else:
			cut = set(cut)
		partition1 = [node for node in self.nodes if node in cut]
		partition2 = [node for node in self.nodes if node not in cut]
		cut_edges = [
			(u, v)
			for (u, v) in self.edges
			if (u in partition1 and v not in partition1)
			or (u in partition2 and v not in partition2)
		]
		uncut_edges = [
			(u, v)
			for u, v in self.edges
			if (u in partition1 and v in partition1)
			or (u in partition2 and v in partition2)
		]
		node_positions = networkx.kamada_kawai_layout(self)
		plot.figure(figsize=(4, 4))
		networkx.draw_networkx_nodes(
			self,
			node_positions,
			nodelist=partition1,
			node_color="tab:red",
			node_size=self.config.get("node_size", 300),
		)
		networkx.draw_networkx_nodes(
			self,
			node_positions,
			nodelist=partition2,
			node_color="tab:green",
			node_size=self.config.get("node_size", 300),
		)
		networkx.draw_networkx_edges(
			self,
			node_positions,
			edgelist=cut_edges,
			style="dashed",
			edge_color="tab:blue",
			alpha=0.7,
			width=self.config.get("width", 1.0),
		)
		networkx.draw_networkx_edges(
			self,
			node_positions,
			edgelist=uncut_edges,
			style="solid",
			width=self.config.get("width", 1.0),
		)
		networkx.draw_networkx_labels(
			self,
			node_positions,
			font_size=self.config.get("font_size", 12),
			font_weight=self.config.get("font_weight", "normal"),
			font_color=self.config.get("font_color", "whitesmoke"),
		)
		plot.tight_layout()
		plot.axis("off")
		plot.show()

	def display_colouring(
		self, colouring: dict[Hashable, int | str | list[object]]
	) -> None:
		"""
		Visualise the solution to a graph colouring optimisation problem.

		This method displays the graph with nodes coloured according to the
		provided colouring solution. The method supports integer colour indices,
		string colour codes, and predefined colour names.

		Args:
			colouring: A dictionary mapping each node to its assigned colour.
				Colours can be integers (mapped to tableau colours), single-character
				strings (B, O, G, R, P, Y), or lists (displayed as black).
		"""
		colour_map = {
			"B": "tab:blue",
			"O": "tab:orange",
			"G": "tab:green",
			"R": "tab:red",
			"P": "tab:pink",
			"Y": "tab:olive",
		}
		tableau_colours = list(colours.TABLEAU_COLORS)
		node_positions: dict[Hashable, Any] = networkx.kamada_kawai_layout(self)
		for node, colour in colouring.items():
			if isinstance(colour, int):
				node_colour = tableau_colours[colour % len(tableau_colours)]
			elif isinstance(colour, str) and colour in colour_map:
				node_colour = colour_map[colour]
			elif isinstance(colour, list):
				node_colour = "tab:black"
			else:
				node_colour = str(colour)
			networkx.draw_networkx_nodes(
				self,
				node_positions,
				nodelist=[node],
				node_color=[node_colour],
			)
		networkx.draw_networkx_edges(
			self,
			node_positions,
			edgelist=self.edges,
			style="solid",
			width=self.config.get("width", 1.0),
		)
		networkx.draw_networkx_labels(
			self,
			node_positions,
			font_size=self.config.get("font_size", 12),
			font_weight=self.config.get("font_weight", "normal"),
			font_color=self.config.get("font_color", "whitesmoke"),
		)
		plot.tight_layout()
		plot.axis("off")
		plot.show()

	def display_path(
		self, path: list[Hashable] | dict[int, list[Hashable]]
	) -> None:
		"""
		Visualise the solution to a Travelling Salesman Problem.

		This method displays the graph with the optimal tour highlighted using
		solid arrows, while unused edges are shown as dashed lines. Edge weights
		are displayed if present in the graph data.

		Args:
			path: Either a list representing the order of cities
				visited in the tour, or a dictionary mapping positions to city
				lists in the format {position: [city]}.
		"""
		if isinstance(path, dict):
			ordered_path = [city for (_, [city]) in sorted(path.items())]
		else:
			ordered_path = list(path)
		tour_edges: list[tuple[Hashable, Hashable]] = []
		for i in range(len(ordered_path)):
			current_city = ordered_path[i]
			next_city = ordered_path[(i + 1) % len(ordered_path)]
			tour_edges.append((current_city, next_city))
		non_tour_edges = [
			(u, v)
			for u, v in self.edges
			if (u, v) not in tour_edges and (v, u) not in tour_edges
		]
		node_positions = networkx.kamada_kawai_layout(self)
		networkx.draw_networkx_nodes(
			self,
			node_positions,
			nodelist=self.nodes,
			node_color="tab:blue",
		)

		networkx.draw_networkx_edges(
			self,
			node_positions,
			edgelist=tour_edges,
			style="solid",
			width=self.config.get("width", 1.0),
			arrows=True,
		)
		networkx.draw_networkx_edges(
			self,
			node_positions,
			edgelist=non_tour_edges,
			style="dashed",
			edge_color="tab:blue",
			alpha=0.7,
			width=self.config.get("width", 1.0),
		)
		networkx.draw_networkx_labels(
			self,
			node_positions,
			font_size=self.config.get("font_size", 12),
			font_weight=self.config.get("font_weight", "normal"),
			font_color=self.config.get("font_color", "whitesmoke"),
		)
		edge_labels = networkx.get_edge_attributes(self, "weight")
		if edge_labels:
			networkx.draw_networkx_edge_labels(self, node_positions, edge_labels)
		plot.tight_layout()
		plot.axis("off")
		plot.show()

	def set_config(self, **config: Any) -> None:
		"""
		Update default visualisation options for all visualisation methods.

		This method allows customisation of the visual appearance across all
		visualisation methods in the class. Changes persist for subsequent
		visualisation calls until modified again.

		Args:
			**options: Keyword arguments for visualisation parameters including
				node_size (int), width (int), font_size (int), font_weight (str),
				font_colour (str), arrowstyle (str), and arrowsize (int).
		"""
		self.config.update(config)


def optimise(matrix: list[list[float]]) -> tuple[float, list[int]]:
	"""Find the binary vector that minimises the QUBO instance defined by ``weights``."""
	from itertools import product

	Q = numpy.asarray(matrix, dtype=float)
	if Q.ndim != 2 or Q.shape[0] != Q.shape[1]:
		raise ValueError("QUBO matrix must be square")
	best_value = float("inf")
	best_vector: list[int] = []
	for candidate in product([0, 1], repeat=Q.shape[0]):
		x = numpy.array(candidate, dtype=float).reshape(-1, 1)
		value = float((x.T @ Q @ x)[0][0])
		if value < best_value:
			best_value = value
			best_vector = list(candidate)
	return (best_value, best_vector)
