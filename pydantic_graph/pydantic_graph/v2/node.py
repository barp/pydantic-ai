from __future__ import annotations

from typing import NewType, Protocol, Literal

NodeId = NewType('NodeId', str)

class HasNodeId(Protocol):
    id: NodeId

def get_node_slug(node: Literal['start'] | Literal['end'] | HasNodeId) -> str:
    if node == 'start':
        return 'start'
    if node == 'end':
        return 'end'
    return node.id
