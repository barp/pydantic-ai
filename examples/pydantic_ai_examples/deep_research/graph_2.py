"""Ideas:
- Probably need something analogous to Command ...
- Graphs need a way to specify whether to end eagerly or after all forked tasks complete finished
    - In the non-eager case, graph needs a way to specify a reducer for multiple entries to g.end()
        - Default is ignore and warn after the first, but a reducer _can_ be used
    - I think the general case should be a JoinNode[GraphStateT, GraphOutputT, GraphOutputT, Any]

Need to be able to:
* Decision (deterministically decide which node to transition to based on the input, possibly the input type)
* Unpack-fork (send each item of an input sequence to the same node by creating multiple GraphWalkers)
* Broadcast-fork (send the same input to multiple nodes by creating multiple GraphWalkers)
* Join (wait for all upstream GraphWalkers to finish before continuing, reducing their inputs as received)
* Streaming (by providing a channel to deps)
* Interruption
    * Implementation 1: if persistence is necessary, return an Interrupt, and use the `resume` API to continue. Note that you need to snapshot graph state (including all GraphWalkers) to resume
    * Implementation 2: if persistence is not necessary and the implementation can just wait, use channels
* Command (?)
* Persistence (???) — how should this work with multiple GraphWalkers?
"""

from __future__ import annotations

import uuid
from collections.abc import Sequence
from dataclasses import dataclass
from typing import Any, Callable, Literal, Never, Protocol, overload


class StepCallProtocol[StateT, InputT, OutputT](Protocol):
    """The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work."""

    def __call__(self, ctx: NodeContext[StateT, InputT]) -> OutputT:
        raise NotImplementedError


class Step[StateT, InputT, OutputT]:
    id: NodeId
    call: StepCallProtocol[StateT, InputT, OutputT]

    # async def run(self, ctx: NodeContext[StateT, InputT]) -> OutputT:
    #     raise NotImplementedError
    #
    # def with_transformed_inputs[NewInputT](self, transform: Callable[[NewInputT], InputT]) -> Step[StateT, NewInputT, OutputT]:
    #     raise NotImplementedError
    #
    # def with_transformed_output[NewOutputT](self, transform: Callable[[OutputT], NewOutputT]) -> Step[StateT, InputT, NewOutputT]:
    #     raise NotImplementedError


class Decision[SourceT, EndT]:
    id: NodeId

    # _branches: list[DecisionBranch]
    _force_source_invariant: Callable[[SourceT], SourceT]
    _force_end_covariant: Callable[[], EndT]

    destinations: Sequence[tuple[Callable[[SourceT], bool], NodeId]] = ()

    def branch[S, E, S2, E2](
        self: Decision[S, E], edge: Decision[S2, E2]
    ) -> Decision[S | S2, E | E2]:
        raise NotImplementedError

    def otherwise[E2](self, edge: Decision[Any, E2]) -> Decision[Any, EndT | E2]:
        raise NotImplementedError


class Join[StateT, InputT, OutputT]:
    id: NodeId

    # sources: list[NodeId]  # should be stored in edges, not in this instance
    reducer_factory: Callable[[StateT, InputT], Reducer[StateT, InputT, OutputT]]


@dataclass
class Fork[StateT, InputT, OutputT]:
    id: NodeId

    mode: Literal['unpack', 'broadcast']
    # destinations: list[NodeId]  # should be stored in edges, not in this instance

    # TODO: Need to add a distributor_factory that accepts the inputs and a GraphWalker(?), and returns a list of tuple[NodeId, OutputT]
    #  - For a with_parallel execution, it would unpack the sequence
    #  - For a multi-edge execution, it just passes the inputs through without modification
    #  For now, it may be enough to assume that one destination means with_parallel, and multiple destinations means multi-edge


# type AnyJoin = Join[Any, Any, Any]
# type EdgeStart = Literal['start'] | Step | AnyJoin
# type EdgeEnd = Step | Decision | AnyJoin | Literal['end']
#
# class Edge:
#     start: EdgeStart
#     end: EdgeEnd


class Graph[StateT, InputT, OutputT]:
    state_type: type[StateT]
    deps_type: type[Any]
    input_type: type[InputT]
    output_type: type[OutputT]

    # nodes: Sequence[Node]
    # edges: Sequence[Edge]


@dataclass
class Edge:
    # Aliases intended to make it clearer where the types are
    type SourceOutputT = Any
    type DestinationInputT = Any

    source: (
        Literal['start']
        | Step[Any, Any, SourceOutputT]
        | Join[Any, Any, SourceOutputT]
        | Fork[Any, Any, SourceOutputT]
    )
    transform: TransformFunction[Any, Any, SourceOutputT, DestinationInputT] | None
    destination: (
        Literal['end']
        | Step[Any, DestinationInputT, Any]
        | Join[Any, DestinationInputT, Any]
        | Fork[Any, DestinationInputT, Any]
        | Decision[DestinationInputT, Any]
    )


class GraphBuilder[StateT, GraphInputT, GraphOutputT]:
    state_type: type[StateT]
    input_type: type[GraphInputT]
    output_type: type[GraphOutputT]

    nodes: list[Any]  # TODO: Replace Any with a more specific type
    edges: list[Edge]

    type SourceWithInputs[InputT, OutputT] = (
        Step[StateT, InputT, OutputT] | Join[StateT, InputT, OutputT]
    )
    type Source[T] = Step[StateT, Any, T] | Join[StateT, Any, T]
    type Destination[T] = (
        Step[StateT, T, Any] | Join[StateT, T, Any] | Decision[T, GraphOutputT]
    )

    # Node building:
    def build_step[InputT, OutputT](
        self, call: Callable[[NodeContext[StateT, InputT]], OutputT]
    ) -> Step[StateT, InputT, OutputT]:
        raise NotImplementedError

    def build_join[InputT, OutputT](
        self, reducer: type[Reducer[StateT, InputT, OutputT]]
    ) -> Join[StateT, InputT, OutputT]:
        raise NotImplementedError

    @staticmethod
    def decision() -> Decision[Never, Never]:
        raise NotImplementedError

    # note: forks are built by calls to `..._with_parallel`, by calling `start_with` multiple times, or by calling `edge` multiple times with the same source

    # TODO: Need to add DecisionBranch class here
    # TODO: Need to add `handle` method somewhere that is aware of the graph state, possibly here

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
        transform: TransformFunction[
            StateT, GraphInputT, GraphInputT, DestinationInputT
        ],
    ) -> None: ...
    def start_with(
        self,
        destination: Destination[Any],
        *,
        transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        self.edges.append(
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
        pre_unpack_transform: TransformFunction[
            StateT, GraphInputT, GraphInputT, Sequence[DestinationInputT]
        ],
    ) -> None: ...
    @overload
    def start_with_unpack[GraphInputItemT, DestinationInputT](
        self: GraphBuilder[StateT, Sequence[GraphInputItemT], GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        post_unpack_transform: TransformFunction[
            StateT, Sequence[GraphInputItemT], GraphInputItemT, DestinationInputT
        ],
    ) -> None: ...
    @overload
    def start_with_unpack[IntermediateT, DestinationInputT](
        self: GraphBuilder[StateT, GraphInputT, GraphOutputT],
        node: Destination[DestinationInputT],
        *,
        pre_unpack_transform: TransformFunction[
            StateT, GraphInputT, GraphInputT, Sequence[IntermediateT]
        ],
        post_unpack_transform: TransformFunction[
            StateT, GraphInputT, IntermediateT, DestinationInputT
        ],
    ) -> None: ...
    def start_with_unpack(
        self,
        node: Destination[Any],
        *,
        pre_unpack_transform: TransformFunction[StateT, Any, Any, Sequence[Any]]
        | None = None,
        post_unpack_transform: TransformFunction[StateT, Any, Any, Any] | None = None,
    ) -> None:
        fork = Fork[Any, Any, Any](
            id=NodeId(f'fork-unpack-start-{node.id}-{_get_unique_string()}'),
            mode='unpack',
        )
        self.edges.append(
            Edge(
                source='start',
                transform=pre_unpack_transform,
                destination=fork,
            )
        )
        self.edges.append(
            Edge(
                source=fork,
                transform=post_unpack_transform,
                destination=node,
            )
        )

    @overload
    def edge[SourceOutputT](
        self, source: Source[SourceOutputT], destination: Destination[SourceOutputT]
    ) -> None: ...
    @overload
    def edge[SourceInputT, SourceOutputT, DestinationInputT](
        self,
        source: SourceWithInputs[SourceInputT, SourceOutputT],
        destination: Destination[DestinationInputT],
        *,
        transform: TransformFunction[
            StateT, SourceInputT, SourceOutputT, DestinationInputT
        ],
    ) -> None: ...
    def edge(
        self,
        source: Source[Any],
        destination: Destination[Any],
        *,
        transform: TransformFunction[Any, Any, Any, Any] | None = None,
    ) -> None:
        self.edges.append(
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
        pre_unpack_transform: TransformFunction[
            StateT, SourceInputT, SourceOutputT, Sequence[DestinationInputT]
        ],
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
        pre_unpack_transform: TransformFunction[
            StateT, SourceInputT, SourceOutputT, Sequence[IntermediateT]
        ],
        post_unpack_transform: TransformFunction[
            StateT, SourceInputT, IntermediateT, DestinationInputT
        ],
    ) -> None: ...
    def edge_unpack[SourceInputT](
        self,
        source: SourceWithInputs[SourceInputT, Any],
        destination: Destination[Any],
        *,
        pre_unpack_transform: TransformFunction[
            StateT, SourceInputT, Any, Sequence[Any]
        ]
        | None = None,
        post_unpack_transform: TransformFunction[StateT, SourceInputT, Any, Any]
        | None = None,
    ) -> None:
        fork = Fork[Any, Any, Any](
            id=NodeId(
                f'fork-unpack-{source.id}-{destination.id}-{_get_unique_string()}'
            ),
            mode='unpack',
        )
        self.edges.append(
            Edge(
                source=source,
                transform=pre_unpack_transform,
                destination=fork,
            )
        )
        self.edges.append(
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
        self.edges.append(
            Edge(
                source=source,
                transform=transform,
                destination='end',
            )
        )


#
#
# class GraphRunContext[StateT]:
#     state: StateT
#
#
# @dataclass
# class Some[OutputT]:
#     output: OutputT
#
#
# type Maybe[OutputT] = Some[OutputT] | None


def _get_unique_string() -> str:
    return str(uuid.uuid4())
