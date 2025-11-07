# scripts/numeric_rep_bench/__init__.py

__all__ = ["NumericRepBenchRunner", "main"]


def __getattr__(name):
    if name in ("NumericRepBenchRunner", "main"):
        from .runner import NumericRepBenchRunner, main
        globals()["NumericRepBenchRunner"] = NumericRepBenchRunner
        globals()["main"] = main
        return globals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
