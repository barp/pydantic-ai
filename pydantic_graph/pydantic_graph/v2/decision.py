from __future__ import annotations

from collections.abc import Sequence
from typing import Any, Callable

from pydantic_graph.v2.node import NodeId


# TODO: Need to add DecisionBranch class here

class Decision[SourceT, EndT]:
    id: NodeId

    # _branches: list[DecisionBranch]
    _force_source_invariant: Callable[[SourceT], SourceT]
    _force_end_covariant: Callable[[], EndT]

    destinations: Sequence[tuple[Callable[[SourceT], bool], NodeId]] = ()

    def branch[S, E, S2, E2](self: Decision[S, E], edge: Decision[S2, E2]) -> Decision[S | S2, E | E2]:
        raise NotImplementedError

    def otherwise[E2](self, edge: Decision[Any, E2]) -> Decision[Any, EndT | E2]:
        raise NotImplementedError


