# Lazy imports to allow CLI commands without shap/captum installed.

def __getattr__(name):
    if name in ('compute_shap_attributions', 'compute_shap_attributions_composite', 'save_shap_results'):
        from .shap_attrib import compute_shap_attributions, compute_shap_attributions_composite, save_shap_results
        return locals()[name]
    elif name in ('compute_ig_attributions', 'compute_ig_attributions_composite', 'save_ig_results'):
        from .ig_attrib import compute_ig_attributions, compute_ig_attributions_composite, save_ig_results
        return locals()[name]
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
