from .base import MemoryBackend


def build_backend(config) -> MemoryBackend:
    if config.backend == "mem0":
        from .mem0_adapter import Mem0Backend

        return Mem0Backend(config)
    from .local import LocalBackend

    return LocalBackend(config.db_path, ann=config.ann)


__all__ = ["MemoryBackend", "build_backend"]
