# openpi patches

Applied by `reanchor_sft.source.temporary_patched_openpi_copy` to a temporary `git archive` copy of
the openpi submodule (`15a9616a00943ada6c20a0f158e3adb39df2ccac`); the submodule itself is never
modified. Same three patches as the LIBERO pipeline that produced our reference recipe.

| patch | sha256 | what |
| --- | --- | --- |
| `0001-filter-incomplete-action-chunks.patch` | `6c08348e…02660` | norm-stats loader keeps the last partial batch (`drop_last=False`, ceil batches); `DataConfig.drop_n_last_frames` (0 here) |
| `0002-save-completed-step-checkpoints.patch` | `883907ba…4176` | checkpoints are written at the completed update count (`save_interval`, `keep_period`, last step) |
| `0003-mask-action-tails.patch` | `eaf44510…b39f` | `Observation.action_loss_mask` and `Pi0Config.active_action_dim`: padded chunk tails and padding action dims carry no loss, the sampler keeps unsupervised dims at their noise |
