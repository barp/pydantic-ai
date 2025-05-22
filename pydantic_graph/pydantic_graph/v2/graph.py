from __future__ import annotations

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass, replace, field
from functools import cached_property
from typing import Any, Callable, Never, overload

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.dominating_forks import DominatingFork, DominatingForkFinder
from pydantic_graph.v2.fork import BroadcastFork, UnpackFork
from pydantic_graph.v2.id_types import WalkerId, NodeRunId
from pydantic_graph.v2.join import Join, Reducer
from pydantic_graph.v2.node import (
    END,
    START,
    AnyDestinationNode,
    AnyNode,
    AnySourceNode,
    EndNode,
    NodeId,
    StartNode,
    get_default_unpack_fork_id,
    get_root_fork_id,
    is_destination,
    is_source,
)
from pydantic_graph.v2.step import Step, StepCallProtocol
from pydantic_graph.v2.transform import TransformFunction, TransformContext
from pydantic_graph.v2.util import get_callable_name, get_unique_string


@dataclass
class Edge:
    source_id: NodeId
    transform: TransformFunction[Any, Any, Any, Any] | None
    destination_id: NodeId

    def source(self, nodes: dict[NodeId, AnyNode]) -> AnySourceNode:
        node = nodes.get(self.source_id)
        if node is None:
            raise ValueError(f'Node {self.source_id} not found in graph')
        if not is_source(node):
            raise ValueError(f'Node {self.source_id} is not a source node: {node}')
        return node

    def destination(self, nodes: dict[NodeId, AnyNode]) -> AnyDestinationNode:
        node = nodes.get(self.source_id)
        if node is None:
            raise ValueError(f'Node {self.source_id} not found in graph')
        if not is_destination(node):
            raise ValueError(f'Node {self.source_id} is not a source node: {node}')
        return node


class GraphBuilder[StateT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    input_type: type[GraphInputT]
    output_type: type[GraphOutputT]

    _nodes: dict[NodeId, AnyNode]
    _edges_by_source: dict[NodeId, list[Edge]]
    _edges_by_destination: dict[NodeId, list[Edge]]

    type Source[OutputT] = Step[StateT, Any, OutputT] | Join[StateT, Any, OutputT]
    type SourceWithInputs[InputT, OutputT] = Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT]
    type Destination[InputT] = Step[StateT, InputT, Any] | Join[StateT, InputT, Any] | Decision[InputT, GraphOutputT]

    # Node building:
    def build_step[InputT, OutputT](
        self, call: StepCallProtocol[StateT, InputT, OutputT]
    ) -> Step[StateT, InputT, OutputT]:
        return Step[StateT, InputT, OutputT](
            id=NodeId(f'step-{get_callable_name(call)}-{get_unique_string()}'), call=call
        )

    def build_join[InputT, OutputT](
        self, reducer_factory: Callable[[StateT, InputT], Reducer[StateT, InputT, OutputT]]
    ) -> Join[StateT, InputT, OutputT]:
        return Join[StateT, InputT, OutputT](
            id=NodeId(f'join-{get_callable_name(reducer_factory)}-{get_unique_string()}'),
            reducer_factory=reducer_factory,
        )

    @staticmethod
    def decision() -> Decision[Never, Never]:
        raise NotImplementedError

    # note: forks are built by calls to `xyz_unpack`, by calling `start_with` multiple times, or by calling `edge` multiple times with the same source

    # TODO: Need to add `handle` method for DecisionBranch somewhere that is aware of the graph state, possibly here

    # Edge building
    # Node "types" to be connected into edges: 'start', 'end', Step, Decision, Join, Fork.
    # You typically don't manually create forks — they are inferred from multiple edges coming out of a single node.
    @overload
    def start_with(self, destination: Destination[GraphInputT]) -> None: ...

    @overload
    def start_with[DestinationInputT](
        self,
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, GraphInputT, GraphInputT, DestinationInputT],
    ) -> None: ...

    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=START,
            transform=transform,
            destination=destination,
        )

    @overload
    def start_with_unpack[GraphInputItemT](
        self: GraphBuilder[StateT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[GraphInputItemT],
    ) -> None: ...

    @overload
    def start_with_unpack[DestinationInputT](
        self,
        node: Destination[DestinationInputT],
        *,
        pre_unpack_transform: TransformFunction[StateT, GraphInputT, GraphInputT, Sequence[DestinationInputT]],
    ) -> None: ...

    @overload
    def start_with_unpack[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_unpack_transform: TransformFunction[StateT, Sequence[GraphInputItemT], GraphInputItemT, DestinationInputT],
    ) -> None: ...

    @overload
    def start_with_unpack[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_unpack_transform: TransformFunction[StateT, GraphInputT, GraphInputT, Sequence[IntermediateT]],
        post_unpack_transform: TransformFunction[StateT, GraphInputT, IntermediateT, DestinationInputT],
    ) -> None: ...

    def start_with_unpack(
        self,
        node: Destination[Any],
        *,
        pre_unpack_transform: TransformFunction[StateT, Any, Any, Sequence[Any]] | None = None,
        post_unpack_transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        # TODO: Accept an optional node_id, necessary for persistence
        #   Probably should require all nodes have a manually-specified unique ID or persistence will break..
        #   If so, might need to make forks a manually-created thing, rather than auto-created
        fork = UnpackFork[Any, Any, Any](id=get_default_unpack_fork_id(START, node))
        self._add_edge_from_nodes(
            source=START,
            transform=pre_unpack_transform,
            destination=fork,
        )
        self._add_edge_from_nodes(
            source=fork,
            transform=post_unpack_transform,
            destination=node,
        )

    @overload
    def edge[SourceOutputT](self, source: Source[SourceOutputT], destination: Destination[SourceOutputT]) -> None: ...

    @overload
    def edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[StateT, SourceInputT, SourceOutputT, DestinationInputT],
    ) -> None: ...

    def edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        transform: TransformFunction[Any, Any, Any, Any] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=destination,
        )

    @overload
    def edge_unpack[SourceInputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[DestinationInputT]],
        destination: Destination[DestinationInputT],
    ) -> None: ...

    @overload
    def edge_unpack[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_unpack_transform: TransformFunction[StateT, SourceInputT, SourceOutputT, Sequence[DestinationInputT]],
    ) -> None: ...

    @overload
    def edge_unpack[SourceInputT, SourceOutputItemT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, Sequence[SourceOutputItemT]],
        destination: Destination[DestinationInputT],
        *,
        post_unpack_transform: TransformFunction[
            StateT,
            SourceInputT,
            SourceOutputItemT,
            DestinationInputT,
        ],
    ) -> None: ...

    @overload
    def edge_unpack[SourceInputT, SourceOutputT, IntermediateT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        pre_unpack_transform: TransformFunction[StateT, SourceInputT, SourceOutputT, Sequence[IntermediateT]],
        post_unpack_transform: TransformFunction[StateT, SourceInputT, IntermediateT, DestinationInputT],
    ) -> None: ...

    def edge_unpack[SourceInputT](
        self,
        source: SourceWithInputs[SourceInputT, Any],
        destination: Destination[Any],
        *,
        pre_unpack_transform: TransformFunction[StateT, SourceInputT, Any, Sequence[Any]] | None = None,
        post_unpack_transform: TransformFunction[StateT, SourceInputT, Any, Any] | None = None,
    ) -> None:
        fork = UnpackFork[Any, Any, Any](
            id=get_default_unpack_fork_id(source, destination),
        )
        self._add_edge_from_nodes(
            source=source,
            transform=pre_unpack_transform,
            destination=fork,
        )
        self._add_edge_from_nodes(
            source=fork,
            transform=post_unpack_transform,
            destination=destination,
        )

    @overload
    def end_from(self, source: Source[GraphOutputT]) -> None: ...

    @overload
    def end_from[SourceInputT, SourceOutputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        *,
        transform: TransformFunction[StateT, SourceInputT, SourceOutputT, GraphOutputT],
    ) -> None: ...

    def end_from(
        self,
        source: Source[Any],
        *,
        transform: TransformFunction[StateT, Any, Any, GraphOutputT] | None = None,
    ) -> None:
        self._add_edge_from_nodes(
            source=source,
            transform=transform,
            destination=END,
        )

    def _add_edge_from_nodes(
        self,
        *,
        source: AnySourceNode,
        transform: TransformFunction[Any, Any, Any, Any] | None,
        destination: AnyDestinationNode,
    ) -> None:
        self._add_node(source)
        self._add_node(destination)

        edge = Edge(source_id=source.id, transform=transform, destination_id=destination.id)
        self._add_edge(edge)

    def _add_node(self, node: AnyNode) -> None:
        existing = self._nodes.get(node.id)
        if existing is None or isinstance(existing, (StartNode, EndNode)):
            pass  # it's not a problem to have non-unique instances of StartNode and EndNode
        elif existing is not node:
            raise ValueError(f'All nodes must have unique node IDs. {node.id!r} was the ID for {existing} and {node}')

        self._nodes[node.id] = node

    def _add_edge(self, edge: Edge) -> None:
        assert edge.source_id in self._nodes, f'Edge source {edge.source_id} not found in graph'
        assert edge.destination_id in self._nodes, f'Edge destination {edge.destination_id} not found in graph'
        self._edges_by_source[edge.source_id].append(edge)
        self._edges_by_destination[edge.destination_id].append(edge)

    def build(self) -> Graph[StateT, GraphInputT, GraphOutputT]:
        """Need to do the following:
        * Warn/error if the graph is not connected
        * Error if the graph does not meet the every-join-has-a-source-fork requirement (otherwise can't know when to proceed past joins)
        * Generate forks for sources with multiple edges
        """
        nodes, edges_by_source = self._build_broadcast_forks()
        dominating_forks = _collect_dominating_forks(nodes, edges_by_source)

        return Graph[StateT, GraphInputT, GraphOutputT](
            # deps_type=self.deps_type,
            state_type=self.state_type,
            input_type=self.input_type,
            output_type=self.output_type,
            nodes=nodes,
            edges_by_source=edges_by_source,
            dominating_forks=dominating_forks,
        )

    def _build_broadcast_forks(self) -> tuple[dict[NodeId, AnyNode], dict[NodeId, list[Edge]]]:
        # Make copies of nodes and edges_by_source before we make modifications:
        nodes = dict(self._nodes)
        edges_by_source: dict[NodeId, list[Edge]] = defaultdict(list)
        for k, v in self._edges_by_source.items():
            edges_by_source[k] = list(v)  # copy the list to prevent modifications

        initial_nodes = list(nodes.items())  # copy the nodes items to prevent modification during iteration
        for source_id, source in initial_nodes:
            if isinstance(source, BroadcastFork):
                continue  # Broadcast forks are the only nodes that are allowed to be the source of multiple edges

            edges_from_source = edges_by_source.get(source_id, [])
            if len(edges_from_source) <= 1:
                continue  # no need to insert a broadcast fork between this node and its destinations

            # Create the "root fork" for this source
            root_fork = BroadcastFork[Any, Any, Any](id=get_root_fork_id(source))
            nodes[root_fork.id] = root_fork

            for e in edges_from_source:
                assert not isinstance(e.destination, BroadcastFork), (
                    'Broadcast forks should only be created while building the graph; this is a bug.'
                )
                edges_by_source[root_fork.id].append(replace(e, source_id=root_fork.id))
            edges_by_source[source_id] = [Edge(source_id=source.id, transform=None, destination_id=root_fork_id)]
        return nodes, edges_by_source


def _collect_dominating_forks(
    graph_nodes: dict[NodeId, AnyNode], graph_edges_by_source: dict[NodeId, list[Edge]]
) -> dict[NodeId, DominatingFork[NodeId]]:
    nodes = set(graph_nodes)
    start_ids = {StartNode.start.id}
    fork_ids = {node_id for node_id, node in graph_nodes.items() if isinstance(node, (BroadcastFork, UnpackFork))}
    edges = {source_id: [e.destination_id for e in edges] for source_id, edges in graph_edges_by_source.items()}
    join_ids = {
        node_id
        for node_id, node in graph_nodes.items()
        if isinstance(node, Join) and node.id not in start_ids and node.id not in fork_ids
    }
    finder = DominatingForkFinder(
        nodes=nodes,
        start_ids=start_ids,
        fork_ids=fork_ids,
        edges=edges,
    )
    join_parents: dict[NodeId, DominatingFork[NodeId]] = {}

    for join_id in join_ids:
        dominating_fork = finder.find_dominating_fork(join_id)
        if dominating_fork is None:
            # TODO: Print out the mermaid graph and explain the problem
            raise ValueError(f'Join node {join_id} has no dominating fork')
        join_parents[join_id] = dominating_fork

    return join_parents


@dataclass
class Graph[StateT, InputT, OutputT]:
    # deps_type: type[Any]
    state_type: type[StateT]
    input_type: type[InputT]
    output_type: type[OutputT]

    nodes: dict[NodeId, AnyNode]
    edges_by_source: dict[NodeId, list[Edge]]
    dominating_forks: dict[NodeId, DominatingFork[NodeId]]  # mapping from join node to the dominating fork

    @cached_property
    def start_edge(self) -> Edge:
        start_edges = self.edges_by_source.get(START.id, [])
        assert len(start_edges) == 1, f'Graphs must have exactly one start edge; got {start_edges}'
        # Note: the way to handle multiple "start edges" is to create a broadcast fork. This should be done when
        # building the graph. Note that the reason we need an explicit broadcast fork is to
        return start_edges[0]

    def __post_init__(self):
        for join_id, dominating_fork in self.dominating_forks.items():
            join_node = self.nodes.get(join_id)
            fork_id = dominating_fork.fork_id
            fork_node = self.nodes.get(fork_id)
            assert isinstance(join_node, Join), f'Node {join_id} is not a Join node: {join_node}'
            assert isinstance(fork_node, (BroadcastFork, UnpackFork)), f'Node {fork_id} is not a Fork node: {fork_node}'

        # Eagerly compute the start edge to raise an error if it doesn't exist
        # We could probably drop this if we are confident there aren't bugs in the implementation; I've added it
        # to help with debugging while working on the implementation
        assert self.start_edge



@dataclass
class GraphWalkerState:
    id: WalkerId

    # With our current BaseNode thing, next_node_id and next_node_inputs are merged into `next_node` itself
    next_node_id: NodeId
    next_node_inputs: Any
    fork_stack: list[tuple[NodeId, NodeRunId]]  # stack of forks that have been entered; used so that the GraphRunner can decide when to proceed through joins


@dataclass
class GraphRun[StateT, InputT, OutputT]:
    graph: Graph[StateT, InputT, OutputT]
    state: StateT
    inputs: InputT

    # persistence: Any  # TODO: Implement this
    walkers: dict[WalkerId, GraphWalkerState]  # mapping from node ID to the walker for that node
    result: Any | None = None  # Note: should probably use a monad (i.e., `Maybe`) for this to distinguish between "no result" and "None is the result"

    def run(self) -> None:
        start_edge = self.graph.start_edge

        next_node_inputs = self.inputs
        if edge.transform is not None:
            ctx = TransformContext(self.state, self.inputs, self.inputs)
            next_node_inputs = edge.transform(ctx)

        new_walker = GraphWalkerState(
            id=WalkerId(f'walker-{get_unique_string()}'),
            next_node_id=edge.destination_id,
            next_node_inputs=next_node_inputs,
            fork_stack=[],
        )
        transformed_input = edge.transform(self.state, self.inputs) if edge.transform else self.inputs

