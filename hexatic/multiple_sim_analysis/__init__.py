"""Aggregate analysis across radius-sweep cylinder simulations."""


def main() -> None:
    from .script import main as script_main

    script_main()

__all__ = ["main"]
