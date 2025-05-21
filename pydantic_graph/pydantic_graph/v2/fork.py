from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic_graph.v2.node import HasNodeId, NodeId, get_node_slug


@dataclass
class BroadcastFork[StateT, InputT, OutputT]:
    id: NodeId


@dataclass
class UnpackFork[StateT, InputT, OutputT]:
    id: NodeId


def get_broadcast_fork_id(source: Literal['start'] | HasNodeId) -> NodeId:
    return NodeId(f'broadcast-fork-{get_node_slug(source)}')


def get_unpack_fork_id(source: Literal['start'] | HasNodeId, destination: HasNodeId) -> NodeId:
    # Note: may need to support multiple forks between a single source and destination
    return NodeId(f'unpack-fork-{get_node_slug(source)}-{get_node_slug(destination)}')
