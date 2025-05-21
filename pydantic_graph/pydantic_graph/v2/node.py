from __future__ import annotations

from enum import Enum
from typing import Any

from typing_extensions import TypeGuard

from pydantic_graph.v2.decision import Decision
from pydantic_graph.v2.fork import BroadcastFork, UnpackFork
from pydantic_graph.v2.id_types import NodeId
from pydantic_graph.v2.join import Join
from pydantic_graph.v2.step import Step


class StartNode(str, Enum):
    start = 'start'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


class EndNode(str, Enum):
    end = 'end'

    @property
    def id(self) -> NodeId:
        return NodeId(f'__{self.value}__')


START = StartNode.start
END = EndNode.end

type AnyNode = (
    StartNode
    | EndNode
    | Step[Any, Any, Any]
    | Join[Any, Any, Any]
    | Decision[Any, Any]
    | BroadcastFork[Any, Any, Any]
    | UnpackFork[Any, Any, Any]
)
type AnySourceNode = (
    StartNode | Step[Any, Any, Any] | Join[Any, Any, Any] | BroadcastFork[Any, Any, Any] | UnpackFork[Any, Any, Any]
)
type AnyDestinationNode = (
    EndNode
    | Step[Any, Any, Any]
    | Join[Any, Any, Any]
    | Decision[Any, Any]
    | BroadcastFork[Any, Any, Any]
    | UnpackFork[Any, Any, Any]
)


def is_source(node: AnyNode) -> TypeGuard[AnySourceNode]:
    return isinstance(node, (StartNode, Step, Join, BroadcastFork, UnpackFork))


def is_destination(node: AnyNode) -> TypeGuard[AnyDestinationNode]:
    return isinstance(node, (EndNode, Step, Join, BroadcastFork, UnpackFork, Decision))


def get_root_fork_id(source: AnyNode) -> NodeId:
    return NodeId(f'__root-broadcast-fork__:{source.id}')


def get_default_unpack_fork_id(source: AnySourceNode, destination: AnyDestinationNode) -> NodeId:
    return NodeId(f'__unpack-fork__:{source.id}:{destination.id}')
