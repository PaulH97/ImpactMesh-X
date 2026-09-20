"""OmegaConf resolvers used inside the config tree.

Registered on import. `train.py` and `predict.py` import this module before Hydra
composes, so the interpolations below resolve during config assembly.

They exist so the config tree stays free of keys that only restate information it
already holds -- a pointer naming which modality to read, or a whole duplicated
dataset file that changes one field.
"""

from omegaconf import OmegaConf


def single_modality_channels(modality_infos) -> int:
    """Channel count of the one temporal modality in `modality_infos`.

    Used by the single-modality baselines (TSViT, UNet3D, ConvLSTM). They take a
    plain `num_bands`, unlike UTAE_MM which consumes the whole `modality_infos`
    dict, so the count has to be pulled out during config assembly.
    """
    temporal = {m: i for m, i in modality_infos.items() if i.get("temporal", True)}
    if len(temporal) != 1:
        raise ValueError(
            f"single_modality_channels expects exactly one temporal modality, "
            f"got {sorted(temporal)} -- this model cannot run on that dataset."
        )
    return int(next(iter(temporal.values()))["channels"])


def variant_suffix(s2_version) -> str:
    """Display suffix for an S2 variant: '' for clear, '-Cloudy', '-Realistic'.

    Keeps `dataset.name` distinct across robustness variants that are otherwise
    the same dataset with `s2_version` overridden.
    """
    if s2_version in (None, "", "clear"):
        return ""
    return "-" + str(s2_version).capitalize()


def scale(value, factor) -> float:
    """`value * factor`, so a config value can be pinned to a fraction of another.

    Used for the cosine's `eta_min`, which is only meaningful relative to the peak
    LR: an absolute 1e-6 is 1% of lr=1e-4 but 0.1% of lr=1e-3, so a hardcoded value
    silently changes the shape of the schedule as soon as the LR is swept.
    """
    return float(value) * float(factor)


def register() -> None:
    OmegaConf.register_new_resolver("single_modality_channels", single_modality_channels, replace=True)
    OmegaConf.register_new_resolver("variant_suffix", variant_suffix, replace=True)
    OmegaConf.register_new_resolver("scale", scale, replace=True)


register()
