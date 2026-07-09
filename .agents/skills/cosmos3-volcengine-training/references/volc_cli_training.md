# Volc CLI Custom Training Reference

This reference condenses the user-provided Machine Learning Platform CLI PDF for
Cosmos3 Volcengine custom training jobs.

## Setup

- Install the CLI with the Volc installer, then make sure `$HOME/.volc/bin` is on
  `PATH`. In this workspace, `~/.bashrc` sources `~/.volc/.profile` only after
  the interactive-shell guard, so non-interactive scripts should export the path
  explicitly if `volc` is not found.
- Run `volc configure` before the first submission. It records AK, SK, and
  region in `$HOME/.volc/config` and `$HOME/.volc/credentials`.
- Valid documented regions are `cn-beijing`, `cn-shanghai`, and `cn-guangzhou`.
  Existing Cosmos3 Volcengine jobs normally use `cn-beijing` unless the queue,
  image, or mounted storage is known to live elsewhere.
- Never put long-lived secrets directly into a task YAML unless the platform
  field is explicitly private. Prefer mounted token files such as
  `/dexmal-datainfra-swy/bootstrap/wandb_token`.

## Submit

Submit a custom training task with:

```bash
volc ml_task submit --conf <task.yaml>
```

Common overrides:

- `--task_name` / `-n`: task name.
- `--description` / `-d`: task description.
- `--user_code_path` / `--cp`: local code path, overriding `UserCodePath`.
- `--entrypoint` / `-e`: entry command, overriding `Entrypoint`. Quote values
  that contain spaces.
- `--args` / `-a`: appended command arguments. Current observed CLI rejects
  `Args` inside the YAML config; if arguments are needed, fold them into
  `Entrypoint` or use the CLI flag only after testing.
- `--image` / `-i`: image URL, overriding `ImageUrl`.
- `--resource_queue_id` / `-q`: queue ID, overriding `ResourceQueueID`.
- `-resource_queue_name` / `-queue_name`: queue name; this overrides
  `--resource_queue_id`.
- `--framework` / `-f`: framework, usually `Custom` for Cosmos3 launchers.
- `--local_diff`: `on` or `off`; default is `on`.
- `--copy-links` / `-L`: upload symlink targets. Use it when symlinks are
  absolute or point outside the uploaded code path.
- `--links`: upload symlinks as symlinks. Use only when identical link targets
  exist inside the container.
- `--access_type`: `Public`, `Queue`, or `Private`.
- `--access_users`: visible users when using restricted access.
- `--preemptible`: submit as a preemptible task.
- `--priority`: numeric priority. Higher values are scheduled first when policy
  and capacity permit.
- `--output json`: request machine-readable output. The value is lowercase.
- `--set Key=Value`: override config fields. Dedicated flags above have higher
  priority than `--set`.

Do not mix placeholder UI values such as `<queue>` or `TODO` into YAML or CLI
flags. Resolve them before submission.

`[Metrics(v3)]` or `[Metrics(v4)]` socket warnings on a bare development
machine are nonfatal submission-side telemetry warnings; diagnose the first later
CLI error line instead.

## YAML Shape

Keep the platform-level task config in YAML, and keep Cosmos3 run controls in
environment variables consumed by the launcher.

Baked-image custom-training fields with no local repo upload:

```yaml
TaskName: "cosmos3_robolabsim_ur5_eef_smoke"
Description: "Cosmos3 RoboLabSim UR5 EEF smoke test"
Entrypoint: "bash /root/code/cosmos-framework/examples/launch_volcengine_robolabsim_ur5_eef_h20.sh"
ImageUrl: "<image-url>"
ResourceQueueName: "<queue-name>"
Framework: "Custom"
TaskRoleSpecs:
  - RoleName: "worker"
    RoleReplicas: 1
    Flavor: "<selected-flavor>"
ActiveDeadlineSeconds: 7200
DelayExitTimeSeconds: 0
AccessType: "Queue"
Preemptible: false
Priority: 4
Envs:
  - Name: "MLP_LOG_PATH"
    Value: "/root/logs"
  - Name: "NNODES"
    Value: "1"
  - Name: "NPROC_PER_NODE"
    Value: "8"
  - Name: "NODE_RANK"
    Value: "0"
  - Name: "DP_SHARD"
    Value: "8"
  - Name: "JOB_NAME"
    Value: "robolabsim_ur5_eef_smoke_001"
  - Name: "MAX_ITER"
    Value: "60"
  - Name: "SAVE_ITER"
    Value: "60"
  - Name: "LOGGING_ITER"
    Value: "10"
  - Name: "MAX_SAMPLES_PER_BATCH"
    Value: "4"
  - Name: "RUN_DRYRUN_FIRST"
    Value: "1"
  - Name: "WANDB_MODE"
    Value: "online"
  - Name: "USE_CUDA_COMPAT"
    Value: "auto"
```

Do not include `Args` in the YAML config with the current CLI. It errors with:
`please remove Args from config and write it into Entrypoint`.

If submitting from a machine with a local repo and you intentionally want Volc to
upload that repo, add:

```yaml
UserCodePath: "/root/code/cosmos-framework/"
RemoteMountCodePath: "/root/code/cosmos-framework/"
```

For baked-image jobs, omit those two fields so the uploaded code mount does not
override the repo already inside the image.

## Dexmal H3c Flavor Table

Use these `Flavor` values for the Dexmal H3c queue when drafting task YAML.

| Flavor | vCPU | Memory | GPU | Network | Disk IOPS | Disk Bandwidth |
| --- | ---: | ---: | --- | ---: | ---: | ---: |
| `ml.pni3ln.4xlarge` | 16 | 224 GiB | GPU-H3c x 1 | 40.0 Gbit/s | 50k | 3.0 Gbit/s |
| `ml.pni3ln.5xlarge` | 20 | 200 GiB | GPU-H3c x 1 | 40.0 Gbit/s | 50k | 3.0 Gbit/s |
| `ml.pni3ln.8xlarge` | 32 | 448 GiB | GPU-H3c x 2 | 80.0 Gbit/s | 100k | 6.0 Gbit/s |
| `ml.pni3ln.11xlarge` | 44 | 440 GiB | GPU-H3c x 2 | 80.0 Gbit/s | 100k | 6.0 Gbit/s |
| `ml.pni3ln.17xlarge` | 68 | 952 GiB | GPU-H3c x 4 | 160.0 Gbit/s | 200k | 12.0 Gbit/s |
| `ml.pni3ln.22xlarge` | 88 | 880 GiB | GPU-H3c x 4 | 160.0 Gbit/s | 200k | 12.0 Gbit/s |
| `ml.pni3ln.35xlarge` | 140 | 1960 GiB | GPU-H3c x 8 | 320.0 Gbit/s | 400k | 24.0 Gbit/s |
| `ml.pni3ln.45xlarge` | 180 | 1960 GiB | GPU-H3c x 8 | 320.0 Gbit/s | 400k | 24.0 Gbit/s |

Selection rule:

- For current Cosmos3 action-policy smoke training launchers that require
  `NPROC_PER_NODE=8` and `DP_SHARD=8`, choose `ml.pni3ln.35xlarge` by default.
- Choose `ml.pni3ln.45xlarge` only when CPU-side dataloading, preprocessing, or
  logging overhead needs the extra vCPU headroom.
- Use 1/2/4-GPU flavors only for launcher preflight, config dryrun, or recipes
  whose topology has been explicitly changed to match that smaller GPU count.

Quota failure triage:

- `Operation is denied: The requested resources exceed the upper limit` means
  Volc authentication, queue selection, and YAML parsing have already succeeded;
  the failure is the submitter personal resource quota.
- Current 8-GPU smoke YAML requests `ml.pni3ln.35xlarge`, i.e. 140 Core,
  1960 GiB, and H20 x 8. If the personal quota is lower, ask the queue admin to
  raise the personal quota to at least that amount before retrying the same YAML.
- With a quota around 54 Core, 588 GiB, and H20 x 2.4, only the 1-GPU and 2-GPU
  flavors fit. Use them only for `PREFLIGHT_ONLY=1` or `DRYRUN_ONLY=1` after
  lowering `NPROC_PER_NODE`, `DP_SHARD`, `MIN_WORLD_SIZE`, and
  `RECOMMENDED_WORLD_SIZE`; do not treat this as a valid replacement for the
  current 8-GPU training smoke.

Only include `ImageCredential` when using a private image registry. Mark private
environment variables with `IsPrivate: true` when they cannot be moved to a
mounted token file.

For mounted data, do not assume environment variables mount storage. `FAST_ROOT` paths such as
`/mlp_vepfs/share/swy/cosmos3-framework` require the corresponding vePFS
storage to be visible inside the training container. If adding `Storages`, match
the platform storage type and container `MountPath` exactly; for vePFS under `volc ml_task submit --conf` 1.2.55, the smoke-proven UI-export shape is `Type: "Vepfs"`, `VepfsId`, `VepfsName`, `SubPath`, `MountPath`, and `ReadOnly`. `VepfsId` is required; omitting it fails with `VepfsId is empty`. If `VepfsName` is reported as unsupported but the task is still created, treat it as a CLI warning; if it blocks submission, drop only `VepfsName` and keep `VepfsId`. Verify the mount with a short `ls` task before launching training. For ordinary TOS Fuse mounts exported by the Volc UI, use `Type: "TosFuse"`
with `Bucket`, `Prefix`, and `MountPath`. Do not substitute `Type: "Tos"` or
`Type: "Sfcs"` unless you also have their required platform-specific names:
`Tos` requires `FsName` and submit fails with `The Cloudfs Name is required`;
`Sfcs` requires `FileSystemName` and submit fails with `The Sfcs FileSystemName
is missing`.
For TOS storage, budget sidecar memory rather than assuming defaults will fit
large concurrent reads.

## Cosmos3 Launcher Mapping

- `Entrypoint` should be the shell launcher, not `python -m ...`; prefer an
  absolute launcher path or `cd /root/code/cosmos-framework && bash examples/...`
  for CLI-submitted jobs. The launcher performs path, topology, CUDA, W&B,
  metadata, dryrun, and log setup.
- Do not include `Args` in YAML. If command arguments are required, append them inside `Entrypoint` because the current CLI rejects config-level `Args`.
- Put routine run changes in `Envs`: `JOB_NAME`, `MAX_ITER`, `SAVE_ITER`,
  `LOGGING_ITER`, `MAX_SAMPLES_PER_BATCH`, `WANDB_MODE`, and topology knobs.
- For one 8-H20 node: set `NNODES=1`, `NPROC_PER_NODE=8`, `NODE_RANK=0`, and
  `DP_SHARD=8`. Let the launcher compute `WORLD_SIZE` and `DP_REPLICATE`.
- For multi-node tasks: set `NNODES`, `NPROC_PER_NODE`, `NODE_RANK`,
  `MASTER_ADDR`, and `MASTER_PORT`, or confirm the launcher maps platform
  variables such as `MLP_WORKER_NUM`, `MLP_ROLE_INDEX`, and `MLP_WORKER_0_HOST`.
- Set `MLP_LOG_PATH=/root/logs` so task logs can be downloaded from the
  platform, while the launcher also writes shared logs under `OUTPUT_ROOT`.
- If the launcher fails with `missing file: .venv/bin/activate`, check whether
  `/root/code/cosmos-framework/.venv` is a dangling symlink. A baked image may
  contain `.venv -> /mlp_vepfs/share/swy/cosmos3-framework/venvs/...`, but
  the custom training container still fails if `/mlp_vepfs/share/swy/.../venvs`
  is not mounted into that task. Fix by mounting the target shared venv path or
  by baking a real venv into the image; relinking `.venv` to the same missing
  target does not help.
- Use the ladder: `PREFLIGHT_ONLY=1`, then `DRYRUN_ONLY=1`, then short smoke
  training, then longer pilot/full runs.

## Lifecycle Commands

Use lowercase `json` output when scripting or pasting results into run notes:

```bash
volc ml_task list --status Queue,Staging,Running --output json
volc ml_task get --id <task-id> --output json
```

Follow a worker log after the task starts:

```bash
volc ml_task logs --task <task-id> --instance worker_0 --lines 200 -f
```

Useful variants:

- Add `--content error` to search logs with a Lucene-style keyword query.
- Add `--reverse` to fetch logs in reverse order.
- For environment checks, log both the symlink and its target, for example:
  `ls -la /root/code/cosmos-framework/.venv /root/code/cosmos-framework/.venv/bin /mlp_vepfs/share/swy/cosmos3-framework/venvs`.
- Use `volc ml_task cancel --id <task-id>` to cancel a submitted task.
- Export an existing task config before cloning or modifying it.
