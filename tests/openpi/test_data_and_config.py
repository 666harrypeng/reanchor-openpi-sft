"""The openpi data config and train config for the conveyor contract (patched copy, CPU JAX)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from reanchor_sft import contract
from reanchor_sft.profile import load_profile
from reanchor_sft.schedule import SFTLaunchRequest
from tests.test_convert_hdf5_reader import H, W

P = load_profile()


def _config(tmp_path: Path, repo_id: str, devices=8, batch=256, run="formal-h100x8"):
    from reanchor_sft.training import build_train_config

    req = SFTLaunchRequest(run, "formal", False, devices, batch)
    return build_train_config(
        P,
        req,
        dataset_repo_id=repo_id,
        assets_root=tmp_path / "assets",
        checkpoint_root=tmp_path / "ckpt",
        wandb_enabled=False,
    )


def test_train_config_pins_the_recipe(patched_openpi, tiny_dataset, tmp_path):
    from reanchor_sft.training import CONFIG_NAME, PROJECT_NAME

    cfg = _config(tmp_path, tiny_dataset)
    assert (cfg.name, cfg.project_name, cfg.exp_name) == (
        CONFIG_NAME,
        "reanchor-sft",
        "formal-h100x8",
    )
    assert PROJECT_NAME == "reanchor-sft"
    m = cfg.model
    assert m.pi05 and m.action_horizon == 50 and m.discrete_state_input and m.active_action_dim == 8
    assert m.max_token_len == 200
    assert (cfg.lr_schedule.warmup_steps, cfg.lr_schedule.decay_steps) == (150, 3000)
    assert (cfg.lr_schedule.peak_lr, cfg.lr_schedule.decay_lr) == (5e-5, 5e-6)
    assert cfg.optimizer.clip_gradient_norm == 1.0 and cfg.ema_decay == 0.999
    assert (cfg.batch_size, cfg.num_train_steps, cfg.save_interval, cfg.keep_period) == (
        256,
        3000,
        250,
        1000,
    )
    assert (cfg.log_interval, cfg.num_workers, cfg.fsdp_devices, cfg.seed) == (25, 8, 1, 42)
    assert cfg.data.repo_id == tiny_dataset and cfg.weight_loader.params_path == P.base_checkpoint
    assert cfg.overwrite is False and cfg.resume is False and cfg.wandb_enabled is False
    assert cfg.checkpoint_dir == tmp_path / "ckpt" / CONFIG_NAME / "formal-h100x8"
    md = cfg.policy_metadata
    assert (md["training_device_count"], md["per_device_batch"], md["training_examples"]) == (
        8,
        32,
        768_000,
    )
    assert md["training_parallelism"] == "replicated_data_parallel"
    assert md["state_names"] == list(contract.STATE_NAMES) and md["fps"] == 25
    cfg4 = _config(tmp_path, tiny_dataset, devices=4, batch=128, run="formal-h100x4")
    assert (cfg4.num_train_steps, cfg4.keep_period, cfg4.lr_schedule.warmup_steps) == (
        6000,
        2000,
        300,
    )


def test_register_config_makes_it_visible_to_openpi(patched_openpi, tiny_dataset, tmp_path):
    from openpi.training.config import get_config

    from reanchor_sft.training import register_config

    cfg = _config(tmp_path, tiny_dataset)
    register_config(cfg)
    assert get_config(cfg.name) is cfg


def _items(cfg, with_model_transforms: bool):
    from openpi.training import data_loader as dl

    data_config = cfg.data.create(cfg.assets_dirs, cfg.model)
    ds = dl.create_torch_dataset(data_config, cfg.model.action_horizon, cfg.model)
    steps = [*data_config.repack_transforms.inputs, *data_config.data_transforms.inputs]
    if with_model_transforms:
        steps += list(data_config.model_transforms.inputs)
    return dl.TransformedDataset(ds, steps)


def test_inputs_mask_padded_tail_and_keep_eight_dims(patched_openpi, tiny_dataset, tmp_path):
    cfg = _config(tmp_path, tiny_dataset, devices=1, batch=8)
    item = _items(cfg, with_model_transforms=False)[4]  # episode 0, frame 4 of 6: 2 real targets
    assert item["actions"].shape == (50, 8) and item["action_loss_mask"].shape == (50, 8)
    assert item["action_loss_mask"][:2].all() and not item["action_loss_mask"][2:].any()
    assert not item["actions"][2:].any()  # padded targets are zeroed, never repeated real actions
    np.testing.assert_allclose(item["actions"][1], 1 + np.arange(8) / 10, rtol=1e-6)
    assert item["state"].shape == (8,) and item["prompt"] == contract.TASK_PROMPT
    assert set(item["image"]) == {"base_0_rgb", "left_wrist_0_rgb", "right_wrist_0_rgb"}
    assert item["image"]["base_0_rgb"].shape == (H, W, 3)
    assert item["image_mask"]["left_wrist_0_rgb"] and not item["image_mask"]["right_wrist_0_rgb"]
    full = _items(cfg, with_model_transforms=True)[4]
    assert full["actions"].shape == (50, 32) and full["state"].shape == (32,)
    assert full["action_loss_mask"].shape == (50, 32) and not full["action_loss_mask"][:, 8:].any()
    assert full["action_loss_mask"][:2, :8].all()
    assert "tokenized_prompt" in full and full["image"]["base_0_rgb"].shape == (224, 224, 3)


def test_inputs_refuse_a_non_contract_state(patched_openpi):
    from openpi.models import model as _model

    from reanchor_sft.data import ConveyorInputs

    tr = ConveyorInputs(model_type=_model.ModelType.PI05)
    img = np.zeros((8, 8, 3), np.uint8)
    with pytest.raises(ValueError, match="state"):
        tr(
            {
                "observation/image": img,
                "observation/wrist_image": img,
                "observation/state": np.zeros(7),
            }
        )


def test_outputs_return_eight_dims(patched_openpi):
    from reanchor_sft.data import ConveyorOutputs

    out = ConveyorOutputs()({"actions": np.arange(50 * 32, dtype=np.float32).reshape(50, 32)})
    assert out["actions"].shape == (50, 8) and out["actions"][0, 7] == 7


def test_transforms_and_data_config_are_picklable_for_loader_workers(patched_openpi, tiny_dataset):
    """openpi's torch loader spawns workers (num_workers 8) and pickles the dataset + transforms."""
    import pickle

    from openpi.models import model as _model

    from reanchor_sft.data import ConveyorDataConfig, ConveyorInputs, ConveyorOutputs

    for obj in (
        ConveyorInputs(model_type=_model.ModelType.PI05),
        ConveyorOutputs(),
        ConveyorDataConfig(repo_id=tiny_dataset),
    ):
        assert type(pickle.loads(pickle.dumps(obj))) is type(obj)
