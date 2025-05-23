from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.id_types import ForkId


@dataclass
class Spread[StateT, InputT, OutputT]:
    id: ForkId
