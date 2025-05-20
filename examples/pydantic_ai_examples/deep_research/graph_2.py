"""
Ideas:
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

from dataclasses import dataclass
from typing import Any, Protocol, Sequence, Callable, Literal, NewType, Never


class NodeContext[StateT, InputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, inputs: InputT):
        self._state = state
        self._inputs = inputs

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def inputs(self) -> InputT:
        return self._inputs

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, inputs={self.inputs})'


class TransformContext[StateT, InputT, OutputT]:
    """The main reason this is not a dataclass is that we need it to be covariant in its type parameters."""

    def __init__(self, state: StateT, inputs: InputT, output: OutputT):
        self._state = state
        self._inputs = inputs
        self._output = output

    @property
    def state(self) -> StateT:
        return self._state

    @property
    def inputs(self) -> InputT:
        return self._inputs

    @property
    def output(self) -> OutputT:
        return self._output

    def __repr__(self):
        return f'{self.__class__.__name__}(state={self.state}, inputs={self.inputs}, output={self.output})'


class ReducerContext:
    pass


class Reducer[GraphStateT, InputT, OutputT]:
    def __init__(self, state: GraphStateT):
        self._state = state
        self._internal_state = None

    def cancel_other_requests(self) -> None:
        raise NotImplementedError

    def reduce(self, graph_state: GraphStateT, input: InputT) -> None:
        raise NotImplementedError

    def finalize(self, graph_state: GraphStateT) -> OutputT:
        raise NotImplementedError

    @staticmethod
    def list_reducer[T](member_type: type[T]) -> type[Reducer[object, T, list[T]]]:
        raise NotImplementedError

    @staticmethod
    def dict_reducer[T](member_type: type[T]) -> type[Reducer[object, dict[str, T], dict[str, T]]]:
        raise NotImplementedError


NodeId = NewType('NodeId', str)


class StepCallProtocol[StateT, InputT, OutputT](Protocol):
    """
    The purpose of this is to make it possible to deserialize step calls similar to how Evaluators work.
    """
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

    def branch[S, E, S2, E2](self: Decision[S, E], edge: Decision[S2, E2]) -> Decision[S | S2, E | E2]:
        raise NotImplementedError

    def otherwise[E2](self, edge: Decision[Any, E2]) -> Decision[Any, EndT | E2]:
        raise NotImplementedError


class Join[StateT, InputT, OutputT]:
    id: NodeId

    sources: list[NodeId]
    reducer_class: type[Reducer[StateT, InputT, OutputT]]

class Fork[StateT, InputT, OutputT]:
    id: NodeId

    mode: Literal['unpack', 'broadcast']
    destinations: list[NodeId]

    # TODO: Need to add a distributor_class that accepts the inputs and a GraphWalker(?), and returns a list of tuple[NodeId, OutputT]
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


class Edge:
    # Aliases intended to make it clearer where the types are
    type SourceOutputT = Any
    type DestinationInputT = Any

    source: (
        Literal['start'] | Step[Any, Any, SourceOutputT] | Join[Any, Any, SourceOutputT] | Fork[Any, Any, SourceOutputT]
    )
    transform: (
        Callable[[SourceOutputT], DestinationInputT] | None
    )  # should convert from source output type to destination input type
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

    edges: list[Edge]

    type Source[T] = Step[StateT, Any, T] | Join[StateT, Any, T]
    type Destination[T] = Step[StateT, T, Any] | Join[StateT, T, Any] | Decision[T, GraphOutputT]

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
    def start_with(self, destination: Destination[GraphInputT]) -> None:
        # Corresponds to an edge from `start` to the node
        # Call this multiple times to start with a fork
        raise NotImplementedError

    def start_with_parallel[T](self: GraphBuilder[StateT, Sequence[T], GraphOutputT], node: Destination[T]) -> None:
        raise NotImplementedError

    def edge[T](self, source: Source[T], destination: Destination[T]) -> None:
        raise NotImplementedError

    def edge_parallel[T](self, source: Source[Sequence[T]], destination: Destination[T]) -> None:
        raise NotImplementedError

    def end_from(self, source: Step[StateT, Any, GraphOutputT] | Join[StateT, Any, GraphOutputT]) -> None:
        raise NotImplementedError


class GraphRunContext[StateT]:
    state: StateT


@dataclass
class Some[OutputT]:
    output: OutputT


type Maybe[OutputT] = Some[OutputT] | None
