import inspect
from pathlib import Path
from typing import Any, Type, TypeVar

import torch
import torch.nn as nn

T = TypeVar('T', bound=nn.Module)


class ModelLoader:
    """
    Handles saving and loading of PyTorch models with automatic config inference.
    Config is extracted from __init__ parameters — no manual field listing required.
    """

    def __init__(self, model_class: Type[T], checkpoint_path: str | Path):
        self.model_class    = model_class
        self.checkpoint_path = Path(checkpoint_path)

    @staticmethod
    def get_model_config(model: nn.Module) -> dict[str, Any]:
        """Extract config by matching model attributes to __init__ parameters."""
        sig = inspect.signature(model.__init__)
        return {
            name: getattr(model, name)
            for name in sig.parameters
            if name != 'self' and hasattr(model, name)
        }

    def save(self, model: nn.Module, output: str | Path, **extras: Any) -> None:
        """
        Save model checkpoint. Config is inferred automatically from __init__.
        Pass any extra non-model data as kwargs.
        """
        output = Path(output)
        output.parent.mkdir(parents=True, exist_ok=True)
        torch.save({
            'model_state': model.state_dict(),
            'model_class': model.__class__.__name__,
            **self.get_model_config(model),
            **extras,
        }, output)
        print(f"Saved: {output}")

    def load(self) -> tuple[T, dict[str, Any]]:
        """
        Load model from checkpoint_path using model_class.

        Returns
        -------
        model : T
            Loaded model in eval mode
        ckpt : dict
            Full checkpoint dict for downstream use
        """
        ckpt = torch.load(self.checkpoint_path, map_location='cpu')
        RESERVED = {'model_state', 'model_class'}
        config = {k: v for k, v in ckpt.items() if k not in RESERVED}
        model  = self.model_class(**config)
        model.load_state_dict(ckpt['model_state'])
        model.eval()
        return model, ckpt