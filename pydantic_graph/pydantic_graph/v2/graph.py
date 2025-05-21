from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Never, overload

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.fork import BroadcastFork, UnpackFork, get_broadcast_fork_id, get_root_fork_id, \
    get_unpack_fork_id
from pydantic_graph.v2.join import Join, Reducer
from pydantic_graph.v2.node import NodeId
from pydantic_graph.v2.step import Step, StepCallProtocol
from pydantic_graph.v2.transform import TransformFunction
from pydantic_graph.v2.util import get_callable_name, get_unique_string

type Node = Step[Any, Any, Any] | Join[Any, Any, Any] | Decision[Any, Any] | BroadcastFork[Any, Any, Any] | UnpackFork[Any, Any, Any]

@dataclass
class Edge:
    # Aliases intended to make it clearer where the types are
    type SourceOutputT = Any
    type DestinationInputT = Any

    source: (
        Literal['start'] | Step[Any, Any, SourceOutputT] | Join[Any, Any, SourceOutputT] | BroadcastFork[Any, Any, SourceOutputT] | UnpackFork[Any, Any, SourceOutputT]
    )
    transform: TransformFunction[Any, Any, SourceOutputT, DestinationInputT] | None
    destination: (
        Literal['end']
        | Step[Any, DestinationInputT, Any]
        | Join[Any, DestinationInputT, Any]
        | BroadcastFork[Any, DestinationInputT, Any]
        | UnpackFork[Any, DestinationInputT, Any]
        | Decision[DestinationInputT, Any]
    )


class GraphBuilder[StateT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    input_type: type[GraphInputT]
    output_type: type[GraphOutputT]

    _nodes: dict[NodeId, Node]
    _edges_by_source: dict[Literal['start'] | NodeId, list[Edge]]
    _edges_by_destination: dict[Literal['end'] | NodeId, list[Edge]]

    type SourceWithInputs[InputT, OutputT] = Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT]
    type Source[T] = Step[StateT, Any, T] | Join[StateT, Any, T]
    type Destination[T] = Step[StateT, T, Any] | Join[StateT, T, Any] | Decision[T, GraphOutputT]

    # Node building:
    def build_step[InputT, OutputT](
        self, call: StepCallProtocol[StateT, InputT, OutputT]
    ) -> Step[StateT, InputT, OutputT]:
        return Step[StateT, InputT, OutputT](id=NodeId(f'step-{get_callable_name(call)}-{get_unique_string()}'), call=call)

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
        self._add_edge(
            Edge(
                source='start',
                transform=transform,
                destination=destination,
            )
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
        fork = UnpackFork[Any, Any, Any](id=get_unpack_fork_id('start', node))
        self._add_edge(
            Edge(
                source='start',
                transform=pre_unpack_transform,
                destination=fork,
            )
        )
        self._add_edge(
            Edge(
                source=fork,
                transform=post_unpack_transform,
                destination=node,
            )
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
        self._add_edge(
            Edge(
                source=source,
                transform=transform,
                destination=destination,
            )
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
            id=get_unpack_fork_id(source, destination),
        )
        self._add_edge(
            Edge(
                source=source,
                transform=pre_unpack_transform,
                destination=fork,
            )
        )
        self._add_edge(
            Edge(
                source=fork,
                transform=post_unpack_transform,
                destination=destination,
            )
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
        self._add_edge(
            Edge(
                source=source,
                transform=transform,
                destination='end',
            )
        )

    def _add_edge(self, edge: Edge) -> None:
        # Store the edge, and validate and store its nodes
        if edge.source == 'start':
            self._edges_by_source['start'].append(edge)
        else:
            self._edges_by_source[edge.source.id].append(edge)
            self._validate_and_store_node(edge.source.id, edge.source)

        if edge.destination == 'end':
            self._edges_by_destination['end'].append(edge)
        else:
            self._edges_by_destination[edge.destination.id].append(edge)
            self._validate_and_store_node(edge.destination.id, edge.destination)

    def _validate_and_store_node(self, id: NodeId, node: Any) -> None:
        existing = self._nodes.get(id)
        if existing is not None and existing is not node:
            raise ValueError(
                f'All nodes must have unique node IDs. {id!r} was the ID for {existing} and {node}'
            )
        self._nodes[id] = node

    def _forkify_edges(self):
        for source_id, edges in list(self._edges_by_source.items()):  # copy the .items() to avoid modifying the dict while iterating
            source: Node | Literal['start'] = 'start' if source_id == 'start' else self._nodes[source_id]
            if isinstance(source, BroadcastFork):
                continue  # Broadcast forks are the only nodes that are allowed to be the source of multiple edges

            if len(edges) > 1:
                root_fork_id = get_broadcast_fork_id(source)
                root_fork: Fork[Any, Any, Any] | None = next((x for x in edges if isinstance(x, Fork) and x.id == root_fork_id and x.mode == 'broadcast'), None)
                if root_fork is None:
                    root_fork = Fork[Any, Any, Any](id=root_fork_id, mode='broadcast')

                for start_edge in edges:
                    if start_edge.destination == root_fork:
                        continue
                    if isinstance(start_edge.destination, Fork) and start_edge.destination.mode == 'broadcast':
                        # Move the edges from this fork to the root fork, and delete this fork
                        redundant_fork = start_edge.destination
                        for edge in self._edges_by_source_id[start_edge.destination.id]:
                            self._add_edge(Edge(source=root_fork, transform=edge.transform, destination=edge.destination))
                        self._edges_by_source_id.pop(start_edge.destination.id)
                        self._nodes.pop(start_edge.destination.id)

                        raise NotImplementedError
                    elif start_edge.destination == 'end':
                        raise NotImplementedError
                    else:
                        destination_id = start_edge.destination.id
                        # rewire these edges to start at start_fork
                        for edge in self._edges_by_source_id[destination_id]:
                            self._add_edge(Edge(source=root_fork, transform=edge.transform, destination=edge.destination))
                        # replace the old edges with one that goes to the source

                self._edges_by_source_id[source_id] = [Edge(source='start', transform=None, destination=root_fork)]

        # Forkify the remaining nodes:
        for source_id, edges in self._edges_by_source_id.items():
            if len(edges) > 1:
                fork = Fork[Any, Any, Any](id=NodeId(f'fork-{source_id}'), mode='broadcast')
                for edge in edges:
                    self._add_edge(Edge(source=fork, transform=edge.transform, destination=edge.destination))
                self._edges_by_source_id[source_id] = [Edge(source=source_id, transform=None, destination=fork)]





    def build(self) -> Graph[StateT, GraphInputT, GraphOutputT]:
        """
        Need to do the following:
        * Warn/error if the graph is not connected
        * Error if the graph does not meet the every-join-has-a-source-fork requirement (otherwise can't know when to proceed past joins)
        * Generate forks for sources with multiple edges
        """


        raise NotImplementedError

        # return Graph[StateT, GraphInputT, GraphOutputT](
        #     state_type=self.state_type,
        #     deps_type=None,
        #     input_type=self.input_type,
        #     output_type=self.output_type,
        # )


class Graph[StateT, InputT, OutputT]:
    state_type: type[StateT]
    deps_type: type[Any]
    input_type: type[InputT]
    output_type: type[OutputT]

    # nodes: Sequence[Node]
    # edges: Sequence[Edge]

