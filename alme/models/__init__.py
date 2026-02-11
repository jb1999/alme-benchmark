"""Model adapters for ALME evaluation."""

from .base import ModelAdapter, ModelResponse

# Registry: name → (module, class, kwargs)
_REGISTRY = {
    "ultravox": ("ultravox", "UltravoxAdapter", {}),
}

ALL_MODELS = sorted(_REGISTRY.keys())


def get_adapter(model_name: str) -> ModelAdapter:
    """Get a model adapter by name.

    Args:
        model_name: One of the registered model names.
                    See ALL_MODELS for the full list.

    Returns:
        ModelAdapter instance (not yet loaded)
    """
    if model_name not in _REGISTRY:
        raise ValueError(
            f"Unknown model: {model_name}. "
            f"Must be one of: {', '.join(ALL_MODELS)}"
        )

    module_name, class_name, kwargs = _REGISTRY[model_name]

    import importlib
    mod = importlib.import_module(f".{module_name}", package=__package__)
    cls = getattr(mod, class_name)
    return cls(**kwargs)


__all__ = ["ModelAdapter", "ModelResponse", "get_adapter", "ALL_MODELS"]
