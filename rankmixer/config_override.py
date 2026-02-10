import json
import os
from copy import deepcopy


def _deep_update(dst, src):
    for key, value in (src or {}).items():
        if isinstance(value, dict) and isinstance(dst.get(key), dict):
            _deep_update(dst[key], value)
        else:
            dst[key] = value
    return dst


def _load_json_or_yaml(path):
    if not path:
        return None
    with open(path, "r", encoding="utf-8") as rf:
        text = rf.read()
    # Prefer JSON; optionally YAML if available.
    if path.endswith((".yml", ".yaml")):
        try:
            import yaml  # type: ignore
        except Exception as exc:
            raise RuntimeError("PyYAML is required for .yaml overrides.") from exc
        return yaml.safe_load(text)
    return json.loads(text)


def _merge_groups(base_groups, override):
    if override is None:
        return base_groups
    if isinstance(override, list):
        return override
    if isinstance(override, dict):
        # Override by group name.
        groups = {g.get("name"): deepcopy(g) for g in (base_groups or []) if isinstance(g, dict)}
        for name, spec in override.items():
            if isinstance(spec, dict):
                groups[name] = deepcopy(spec)
                if "name" not in groups[name]:
                    groups[name]["name"] = name
            else:
                groups[name] = {"name": name, "features": list(spec)}
        return list(groups.values())
    raise TypeError("group overrides must be a list or dict")


def apply_overrides(config_cls, overrides=None, groups_path=None, group_overrides=None):
    """Apply overrides to TrainConfig class in-place.

    overrides: dict, e.g. {"train_params": {"rankmixer": {"d_model": 256}}}
    groups_path: path to JSON/YAML file defining semantic_groups
    group_overrides: list (replace) or dict (by group name)
    """
    if overrides:
        for key, value in overrides.items():
            if hasattr(config_cls, key) and isinstance(getattr(config_cls, key), dict) and isinstance(value, dict):
                current = deepcopy(getattr(config_cls, key))
                _deep_update(current, value)
                setattr(config_cls, key, current)
            else:
                setattr(config_cls, key, value)

    groups_from_path = _load_json_or_yaml(groups_path) if groups_path else None
    groups_override = groups_from_path if groups_from_path is not None else group_overrides

    if groups_override is not None:
        train_params = deepcopy(getattr(config_cls, "train_params", {}))
        rankmixer_params = deepcopy(train_params.get("rankmixer", {}))
        base_groups = rankmixer_params.get("semantic_groups")
        rankmixer_params["semantic_groups"] = _merge_groups(base_groups, groups_override)
        train_params["rankmixer"] = rankmixer_params
        setattr(config_cls, "train_params", train_params)


def load_overrides_from_env():
    """Load overrides from env vars (JSON string or JSON/YAML file)."""
    overrides = None
    groups_path = os.environ.get("RANKMIXER_GROUPS_PATH") or None
    group_overrides = None

    overrides_path = os.environ.get("RANKMIXER_OVERRIDES_PATH")
    overrides_str = os.environ.get("RANKMIXER_OVERRIDES")
    group_overrides_str = os.environ.get("RANKMIXER_GROUP_OVERRIDES")

    if overrides_path:
        overrides = _load_json_or_yaml(overrides_path)
    elif overrides_str:
        overrides = json.loads(overrides_str)

    if group_overrides_str:
        group_overrides = json.loads(group_overrides_str)

    return overrides, groups_path, group_overrides
