"""Agent node factories.

Each module exposes a ``make_*_node(llm=...)`` factory returning an async
callable ``(MeetingState) -> partial_update``. Factories accept an injected
structured LLM so the graph is fully unit-testable offline.
"""
