"""openpi data transforms and data config for the conveyor contract.

Dataset keys (`reanchor_sft.convert`): `image`, `wrist_image`, `state` (8), `actions` (8), task
string. Model inputs: `base_0_rgb` ← fixed_cam, `left_wrist_0_rgb` ← wrist_cam, `right_wrist_0_rgb`
zeros with mask False, `state`, `prompt`; training targets `actions (50, 8)` with
`action_loss_mask` from LeRobot's `actions_is_pad` (patch 0003): padded chunk tails at an
episode's end carry no loss and are zeroed, never repeated real actions. Imports openpi lazily;
run inside `temporary_patched_openpi_copy` (the patches add `action_loss_mask`).
"""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from reanchor_sft import contract


def _require_patched_openpi() -> None:
    from openpi.models import model as _model

    if "action_loss_mask" not in _model.Observation.__dataclass_fields__:
        raise RuntimeError(
            "openpi is not the patched copy: enter "
            "reanchor_sft.source.temporary_patched_openpi_copy first"
        )


def _parse_image(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if np.issubdtype(image.dtype, np.floating):
        image = (255 * image).astype(np.uint8)
    if image.shape[0] == 3:  # LeRobot stores (C, H, W)
        image = np.einsum("chw->hwc", image)
    return image


def _make_transforms():
    """Build the transform classes against whatever openpi is importable (lazy)."""
    import openpi.transforms as transforms
    from openpi.models import model as _model

    @dataclasses.dataclass(frozen=True)
    class ConveyorInputs(transforms.DataTransformFn):
        model_type: _model.ModelType

        def __call__(self, data: dict) -> dict:
            state = np.asarray(data["observation/state"], dtype=np.float32)
            if state.shape != (contract.STATE_DIM,) or not np.isfinite(state).all():
                raise ValueError(f"state must be finite ({contract.STATE_DIM},), got {state.shape}")
            base = _parse_image(data["observation/image"])
            wrist = _parse_image(data["observation/wrist_image"])
            inputs: dict[str, Any] = {
                "state": state,
                "image": {
                    "base_0_rgb": base,
                    "left_wrist_0_rgb": wrist,
                    "right_wrist_0_rgb": np.zeros_like(base),
                },
                "image_mask": {
                    "base_0_rgb": np.True_,
                    "left_wrist_0_rgb": np.True_,
                    "right_wrist_0_rgb": np.True_
                    if self.model_type == _model.ModelType.PI0_FAST
                    else np.False_,
                },
            }
            if "actions" in data:  # training only
                actions = np.asarray(data["actions"], dtype=np.float32)
                if actions.shape != (contract.ACTION_HORIZON, contract.ACTION_DIM):
                    raise ValueError(f"actions must be (50, 8), got {actions.shape}")
                is_pad = np.asarray(data.get("actions_is_pad", np.zeros(len(actions), bool)))
                if is_pad.shape != (len(actions),) or is_pad.dtype != np.bool_:
                    raise ValueError("actions_is_pad must be a boolean (horizon,) array")
                if is_pad[0] or np.any(np.diff(is_pad.astype(np.int8)) < 0):
                    raise ValueError("a source starts at a real action with a contiguous prefix")
                mask = np.broadcast_to(~is_pad[:, None], actions.shape).copy()
                if not np.isfinite(actions[mask]).all():
                    raise ValueError("real action targets must be finite")
                inputs["actions"] = np.where(mask, actions, 0.0).astype(np.float32)
                inputs["action_loss_mask"] = mask
            if "prompt" in data:
                inputs["prompt"] = data["prompt"]
            return inputs

    @dataclasses.dataclass(frozen=True)
    class ConveyorOutputs(transforms.DataTransformFn):
        def __call__(self, data: dict) -> dict:
            return {"actions": np.asarray(data["actions"][:, : contract.ACTION_DIM])}

    return ConveyorInputs, ConveyorOutputs


def __getattr__(name: str):
    if name in ("ConveyorInputs", "ConveyorOutputs", "ConveyorDataConfig"):
        _require_patched_openpi()
        inputs_cls, outputs_cls = _make_transforms()
        if name == "ConveyorInputs":
            return inputs_cls
        if name == "ConveyorOutputs":
            return outputs_cls
        return _make_data_config(inputs_cls, outputs_cls)
    raise AttributeError(name)


def _make_data_config(inputs_cls, outputs_cls):
    import openpi.transforms as transforms
    from openpi.training.config import DataConfig, DataConfigFactory, ModelTransformFactory

    @dataclasses.dataclass(frozen=True)
    class ConveyorDataConfig(DataConfigFactory):
        def create(self, assets_dirs, model_config) -> DataConfig:
            if not getattr(model_config, "pi05", False) or not model_config.discrete_state_input:
                raise ValueError("the conveyor recipe is pi0.5 with discrete state input")
            if getattr(model_config, "active_action_dim", None) != contract.ACTION_DIM:
                raise ValueError(f"active_action_dim must be {contract.ACTION_DIM}")
            base = dataclasses.replace(
                self.base_config or DataConfig(), prompt_from_task=True, drop_n_last_frames=0
            )
            factory = dataclasses.replace(self, base_config=base)
            config = DataConfigFactory.create_base_config(factory, assets_dirs, model_config)
            repack = transforms.Group(
                inputs=[
                    transforms.RepackTransform(
                        {
                            "observation/image": "image",
                            "observation/wrist_image": "wrist_image",
                            "observation/state": "state",
                            "actions": "actions",
                            "actions_is_pad": "actions_is_pad",
                            "prompt": "prompt",
                        }
                    )
                ]
            )
            data_transforms = transforms.Group(
                inputs=[inputs_cls(model_type=model_config.model_type)], outputs=[outputs_cls()]
            )
            return dataclasses.replace(
                config,
                repack_transforms=repack,
                data_transforms=data_transforms,
                model_transforms=ModelTransformFactory()(model_config),
            )

    return ConveyorDataConfig
