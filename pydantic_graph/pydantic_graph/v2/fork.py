from __future__ import annotations

from dataclasses import dataclass

from pydantic_graph.v2.id_types import NodeId


@dataclass
class BroadcastFork[StateT, InputT, OutputT]:
    id: NodeId


@dataclass
class UnpackFork[StateT, InputT, OutputT]:
    id: NodeId
