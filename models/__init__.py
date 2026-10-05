# Lazy imports to allow --help and CLI commands without mamba_ssm installed.
# Model classes are imported on demand to avoid requiring GPU at import time.

def __getattr__(name):
    if name == 'DenoisingAutoencoder':
        from .dae import DenoisingAutoencoder
        return DenoisingAutoencoder
    elif name == 'BiMambaAutoencoder':
        from .bimamba_ae import BiMambaAutoencoder
        return BiMambaAutoencoder
    elif name == 'BiMambaLayer':
        from .bimamba import BiMambaLayer
        return BiMambaLayer
    elif name == 'BiMambaClassifier':
        from .classifier import BiMambaClassifier
        return BiMambaClassifier
    elif name == 'MLPClassifier':
        from .mlp_classifier import MLPClassifier
        return MLPClassifier
    elif name == 'CompositeModel':
        from .composite_model import CompositeModel
        return CompositeModel
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
