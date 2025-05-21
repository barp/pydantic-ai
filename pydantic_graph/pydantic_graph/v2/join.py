from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic_graph.v2.id_types import NodeId


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
    def dict_reducer[T](
        member_type: type[T],
    ) -> type[Reducer[object, dict[str, T], dict[str, T]]]:
        raise NotImplementedError


type ReducerFactory[StateT, InputT, OutputT] = Callable[[StateT, InputT], Reducer[StateT, InputT, OutputT]]


@dataclass
class Join[StateT, InputT, OutputT]:
    id: NodeId

    reducer_factory: ReducerFactory[StateT, InputT, OutputT]
