import h5py
import json
import numpy as np
import hashlib
import torch
import datetime
from utils import to_numpy
"""

"""

def initialize_dataset(path):
    with h5py.File(path, "w") as f:
        f.create_group("runs")
        f.create_group("run_groups")

def filter_model_kwargs(model_settings):
    if not model_settings:
        return {}
    allowed = {
        "dt",
        "tau_init",
        "device",
        "weights",
        "remove_reciprocal",
        "vrest_init",
        "tau_by_type",
        "vrest_by_type",
        "default_scale",
        "scale_by_connection_type",
    }
    return {key: value for key, value in model_settings.items() if key in allowed}

def _next_numeric_group_id(group):
    numeric_ids = [int(name) for name in group.keys() if str(name).isdigit()]
    return max(numeric_ids, default=0) + 1

def hash_model_params(param):
    if isinstance(param, (torch.Tensor, np.ndarray)):
        arr = np.ascontiguousarray(to_numpy(param))
        hasher = hashlib.sha256()
        hasher.update(arr.tobytes())
        hasher.update(str(arr.shape).encode("utf-8"))
        hasher.update(str(arr.dtype).encode("utf-8"))
        return hasher.hexdigest()
    return param
def _decode_attr(value):
    if isinstance(value, (bytes, np.bytes_)):
        return value.decode("utf-8")
    return value


def _load_json_attr(attrs, key, default=None):
    raw = attrs.get(key)
    if raw is None:
        return default
    raw = _decode_attr(raw)
    if raw == "":
        return default
    try:
        return json.loads(raw)
    except (TypeError, json.JSONDecodeError):
        return default

def _format_param_key(key):
    if isinstance(key, str):
        return key
    if isinstance(key, (np.integer, np.floating)):
        return str(key.item())
    if isinstance(key, (tuple, list)):
        # Preserve tuple/list structure deterministically in a JSON-safe key.
        return json.dumps(_format_param_value(key), sort_keys=True)
    return str(key)


def _format_param_value(value):
    if isinstance(value, (torch.Tensor, np.ndarray)):
        return hash_model_params(value)
    if isinstance(value, dict):
        formatted = {}
        normalized_items = [(_format_param_key(k), v) for k, v in value.items()]
        for key, nested_value in sorted(normalized_items, key=lambda item: item[0]):
            formatted[key] = _format_param_value(nested_value)
        return formatted
    if isinstance(value, (list, tuple)):
        return [_format_param_value(item) for item in value]
    if isinstance(value, (np.integer, np.floating)):
        return value.item()
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    return str(value)


def format_model_params(params):
    if params is None:
        return {}
    if not isinstance(params, dict):
        raise TypeError("model params must be a dictionary or None.")
    return _format_param_value(params)


def _write_single_run(runs_group, stimulus_params, v_final, v_history, t, model_params=None, run_group_id=None):
    run_id = _next_numeric_group_id(runs_group)
    run = runs_group.create_group(str(run_id))
    run.create_dataset("v_final", data=v_final)
    run.create_dataset("v_history", data=v_history)
    run.create_dataset("t", data=t)
    run.attrs["stimulus_params"] = json.dumps(stimulus_params)
    if model_params is not None:
        run.attrs["model_params"] = json.dumps(format_model_params(model_params))
    if run_group_id is not None:
        run.attrs["run_group_id"] = int(run_group_id)
    return run_id


def add_single_run(path, stimulus_params, v_final, v_history, t, model_params=None):
    with h5py.File(path, "a") as f:
        runs = f.require_group("runs")
        return _write_single_run(
            runs,
            stimulus_params=stimulus_params,
            v_final=v_final,
            v_history=v_history,
            t=t,
            model_params=model_params,
        )


def add_run_group(path, runs_data, model_params=None, group_label=None):
    if not runs_data:
        raise ValueError("runs_data must contain at least one run.")

    with h5py.File(path, "a") as f:
        runs = f.require_group("runs")
        run_groups = f.require_group("run_groups")
        run_group_id = _next_numeric_group_id(run_groups)
        run_group = run_groups.create_group(str(run_group_id))

        if model_params is not None:
            run_group.attrs["model_params"] = json.dumps(format_model_params(model_params))
        if group_label is not None:
            run_group.attrs["group_label"] = str(group_label)
        run_group.attrs["created_at"] = datetime.datetime.now().isoformat()
        run_ids = []
        for idx, run_data in enumerate(runs_data):
            required = {"stimulus_params", "v_final", "v_history", "t"}
            if not required.issubset(run_data):
                missing = sorted(required - set(run_data.keys()))
                raise ValueError(f"runs_data[{idx}] is missing required fields: {missing}")

            run_model_params = run_data.get("model_params", model_params)
            run_id = _write_single_run(
                runs,
                stimulus_params=run_data["stimulus_params"],
                v_final=run_data["v_final"],
                v_history=run_data["v_history"],
                t=run_data["t"],
                model_params=run_model_params,
                run_group_id=run_group_id,
            )
            run_ids.append(run_id)

        run_group.create_dataset("run_ids", data=np.asarray(run_ids, dtype=np.int64))
        return run_group_id, run_ids

def load_single_run(path, run_id=None, model_params=None, stimulus_params=None):
    with h5py.File(path, "r") as f:
        runs = f.require_group("runs")
        if run_id is not None:
            run = runs.get(str(run_id))
            if run is None:
                raise ValueError(f"Run id {run_id} not found.")
            return run["v_final"][()], run["v_history"][()], run["t"][()]
        else:
            for r in runs.values():
                if (
                    r.attrs.get("model_params") == json.dumps(format_model_params(model_params))
                    and r.attrs.get("stimulus_params") == json.dumps(stimulus_params)
                ):
                    return r["v_final"][()], r["v_history"][()], r["t"][()]
            raise ValueError("No run found with the given model params and stimulus params.")

def load_run_group(path, group_id=None, group_label=None, model_params=None, include_data=False):
    with h5py.File(path, "r") as f:
        run_groups = f.require_group("run_groups")
        group = None

        if group_id is not None:
            group = run_groups.get(str(group_id))
        else:
            for g in run_groups.values():
                if (
                    g.attrs.get("group_label") == group_label
                    and g.attrs.get("model_params") == json.dumps(format_model_params(model_params))
                ):
                    group = g
                    break

        if group is None:
            raise ValueError("No run group found with the given group label and model params.")

        resolved_group_id = int(group.name.split("/")[-1])
        run_ids = [int(x) for x in group["run_ids"][()]]

        result = {
            "group_id": resolved_group_id,
            "group_label": _decode_attr(group.attrs.get("group_label")),
            "created_at": _decode_attr(group.attrs.get("created_at")),
            "model_params": _load_json_attr(group.attrs, "model_params", default=None),
            "run_ids": run_ids,
        }

        if not include_data:
            return result

        runs = f.require_group("runs")
        runs_data = []
        for run_id in run_ids:
            run = runs.get(str(run_id))
            if run is None:
                continue
            runs_data.append(
                {
                    "run_id": run_id,
                    "stimulus_params": _load_json_attr(run.attrs, "stimulus_params", default={}),
                    "model_params": _load_json_attr(run.attrs, "model_params", default=None),
                    "v_final": run["v_final"][()],
                    "v_history": run["v_history"][()],
                    "t": run["t"][()],
                }
            )
        result["runs"] = runs_data
        return result