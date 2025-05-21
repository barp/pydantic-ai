from __future__ import annotations

from typing import Protocol


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


class TransformFunction[StateT, SourceInputT, SourceOutputT, DestinationInputT](Protocol):
    def __call__(self, ctx: TransformContext[StateT, SourceInputT, SourceOutputT]) -> DestinationInputT:
        raise NotImplementedError
