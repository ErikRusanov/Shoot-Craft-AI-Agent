"""Port contract tests.

Each module pins one port's behavioral contract through a fixture parametrized by
implementation. Today the only implementation is the in-memory fake; when a real
connector lands, it is appended to that module's parameter list and must pass the
*same* assertions — that is the whole point of keeping these implementation-blind.
"""
