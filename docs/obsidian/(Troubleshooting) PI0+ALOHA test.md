# Trial 1: Local Python 가상환경 기반 실행
```bash title:"Local Python 가상환경 기반 실행 (Client - Simulator)" info:2-3,5-6, error:36
(aloha) $ MUJOCO_GL=egl python examples/aloha_sim/main.py
INFO:absl:MUJOCO_GL=egl, attempting to import specified OpenGL backend.
INFO:OpenGL.acceleratesupport:No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
INFO:absl:MuJoCo library version is: 2.3.7
INFO:root:Waiting for server at ws://0.0.0.0:8000...
INFO:root:Still waiting for server...
INFO:root:Still waiting for server...
INFO:root:Still waiting for server...
INFO:root:Starting episode...
Traceback (most recent call last):
  File "/home/kiro-jwhong/git-repos/openpi/examples/aloha_sim/main.py", line 55, in <module>
    tyro.cli(main)
  File "/home/kiro-jwhong/git-repos/openpi/examples/aloha_sim/.venv/lib/python3.11/site-packages/tyro/_cli.py", line 191, in cli
    return run_with_args_from_cli()
           ^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kiro-jwhong/git-repos/openpi/examples/aloha_sim/main.py", line 50, in main
    runtime.run()
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 35, in run
    self._run_episode()
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 64, in _run_episode
    self._step()
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 83, in _step
    action = self._agent.get_action(observation)
             ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py", line 15, in get_action
    return self._policy.infer(observation)
           ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/action_chunk_broker.py", line 29, in infer
    self._last_results = self._policy.infer(obs)
                         ^^^^^^^^^^^^^^^^^^^^^^^
  File "/home/kiro-jwhong/git-repos/openpi/packages/openpi-client/src/openpi_client/websocket_client_policy.py", line 47, in infer
    response = self._ws.recv()
               ^^^^^^^^^^^^^^^
  File "/home/kiro-jwhong/git-repos/openpi/examples/aloha_sim/.venv/lib/python3.11/site-packages/websockets/sync/connection.py", line 290, in recv
    raise self.protocol.close_exc from self.recv_exc
websockets.exceptions.ConnectionClosedError: no close frame received or sent
```
- 2-3행: OpenGL 관련 이슈가 있는 것으로 보이나, [이슈 트래킹](https://github.com/Physical-Intelligence/openpi/issues/462) 시 $\pi_0$ 모델 동작 자체에 영향 주는 문제는 아닌 것으로 사료
- 10-36행: server 연결 시 강제 종료되며 에러 발생.

```bash title:"Local Python 가상환경 기반 실행 (Server - Policy)" info:2-3,17-19
(openpi) $ uv run scripts/serve_policy.py --env ALOHA_SIM
INFO:root:Loading model...
INFO:2025-09-12 13:53:00,346:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
INFO:2025-09-12 13:53:00,346:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
INFO:absl:orbax-checkpoint version: 0.11.13
INFO:absl:Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x705e22bf24d0>
INFO:absl:Restoring checkpoint from /home/kiro-jwhong/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params.
INFO:absl:[thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
INFO:absl:[process=0][thread=MainThread] No metadata found for any process_index, checkpoint_dir=/home/kiro-jwhong/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params. time elapsed=0.0008418560028076172 seconds. If the checkpoint does not contain jax.Array then it is expected. If checkpoint contains jax.Array then it should lead to an error eventually; if no error is raised then it is a bug.
INFO:absl:[process=0] /jax/checkpoint/read/bytes_per_sec: 1.1 GiB/s (total bytes: 6.0 GiB) (time elapsed: 5 seconds) (per-host)
INFO:absl:Finished restoring checkpoint in 5.41 seconds from /home/kiro-jwhong/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/params.
INFO:root:Norm stats not found in /home/kiro-jwhong/git-repos/openpi/assets/pi0_aloha_sim/lerobot/aloha_sim_transfer_cube_human, skipping.
INFO:root:Loaded norm stats from /home/kiro-jwhong/.cache/openpi/openpi-assets/checkpoints/pi0_aloha_sim/assets/lerobot/aloha_sim_transfer_cube_human
INFO:root:Creating server (host: kirojwhong-MS-7D99, ip: 127.0.1.1)
INFO:websockets.server:server listening on 0.0.0.0:8000
INFO:websockets.server:connection open
INFO:openpi.serving.websocket_policy_server:Connection from ('127.0.0.1', 33526) opened
```
- 17-19행: Server 연결 성공하나 Client와 함께 강제 종료

# Trial 2: Docker 기반 실행
```bash title:"Docker기반 실행" info:25-26,33-34,35,52-53 error:80-82
 $ export SERVER_ARGS="--env ALOHA_SIM"
 $ docker compose -f examples/aloha_sim/compose.yml up --build
 ...
 => [openpi_server] resolving provenance for metadata file                                         0.0s
[+] Running 3/3
 ✔ openpi_server                  Built                                                            0.0s 
 ✔ aloha_sim                      Built                                                            0.0s 
 ✔ Container aloha_sim-runtime-1  Recreated                                                        0.1s 
Attaching to openpi_server-1, runtime-1
openpi_server-1  | 
openpi_server-1  | ==========
openpi_server-1  | == CUDA ==
openpi_server-1  | ==========
openpi_server-1  | 
openpi_server-1  | CUDA Version 12.2.2
openpi_server-1  | 
openpi_server-1  | Container image Copyright (c) 2016-2023, NVIDIA CORPORATION & AFFILIATES. All rights reserved.
openpi_server-1  | 
openpi_server-1  | This container image and its contents are governed by the NVIDIA Deep Learning Container License.
openpi_server-1  | By pulling and using the container, you accept the terms and conditions of this license:
openpi_server-1  | https://developer.nvidia.com/ngc/nvidia-deep-learning-container-license
openpi_server-1  | 
openpi_server-1  | A copy of this license is made available in this container at /NGC-DL-CONTAINER-LICENSE for your convenience.
openpi_server-1  | 
runtime-1        | INFO:absl:MUJOCO_GL=egl, attempting to import specified OpenGL backend.
runtime-1        | INFO:OpenGL.acceleratesupport:No OpenGL_accelerate module loaded: No module named 'OpenGL_accelerate'
runtime-1        | libEGL warning: egl: failed to create dri2 screen
runtime-1        | libEGL warning: egl: failed to create dri2 screen
runtime-1        | libEGL warning: egl: failed to create dri2 screen
runtime-1        | libEGL warning: egl: failed to create dri2 screen
runtime-1        | INFO:absl:MuJoCo library version is: 2.3.7
runtime-1        | INFO:root:Waiting for server at ws://0.0.0.0:8000...
runtime-1        | INFO:root:Still waiting for server...
openpi_server-1  | INFO:root:Loading model...
openpi_server-1  | INFO:2025-09-12 04:53:23,269:jax._src.xla_bridge:925: Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
openpi_server-1  | INFO:jax._src.xla_bridge:Unable to initialize backend 'rocm': module 'jaxlib.xla_extension' has no attribute 'GpuAllocatorConfig'
openpi_server-1  | INFO:2025-09-12 04:53:23,269:jax._src.xla_bridge:925: Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
openpi_server-1  | INFO:jax._src.xla_bridge:Unable to initialize backend 'tpu': INTERNAL: Failed to open libtpu.so: libtpu.so: cannot open shared object file: No such file or directory
openpi_server-1  | INFO:absl:orbax-checkpoint version: 0.11.13
openpi_server-1  | INFO:absl:Created BasePyTreeCheckpointHandler: use_ocdbt=True, use_zarr3=False, pytree_metadata_options=PyTreeMetadataOptions(support_rich_types=False), array_metadata_store=<orbax.checkpoint._src.metadata.array_metadata_store.Store object at 0x7c608f5e5790>
openpi_server-1  | INFO:absl:Restoring checkpoint from /openpi_assets/openpi-assets/checkpoints/pi0_aloha_sim/params.
openpi_server-1  | INFO:absl:[thread=MainThread] Failed to get flag value for EXPERIMENTAL_ORBAX_USE_DISTRIBUTED_PROCESS_ID.
openpi_server-1  | INFO:absl:[process=0][thread=MainThread] No metadata found for any process_index, checkpoint_dir=/openpi_assets/openpi-assets/checkpoints/pi0_aloha_sim/params. time elapsed=0.00036978721618652344 seconds. If the checkpoint does not contain jax.Array then it is expected. If checkpoint contains jax.Array then it should lead to an error eventually; if no error is raised then it is a bug.
runtime-1        | INFO:root:Still waiting for server...
openpi_server-1  | INFO:absl:[process=0] /jax/checkpoint/read/bytes_per_sec: 1.1 GiB/s (total bytes: 6.0 GiB) (time elapsed: 5 seconds) (per-host)
openpi_server-1  | INFO:absl:Finished restoring checkpoint in 5.41 seconds from /openpi_assets/openpi-assets/checkpoints/pi0_aloha_sim/params.
openpi_server-1  | INFO:root:Norm stats not found in /app/assets/pi0_aloha_sim/lerobot/aloha_sim_transfer_cube_human, skipping.
openpi_server-1  | INFO:root:Loaded norm stats from /openpi_assets/openpi-assets/checkpoints/pi0_aloha_sim/assets/lerobot/aloha_sim_transfer_cube_human
openpi_server-1  | INFO:root:Creating server (host: kirojwhong-MS-7D99, ip: 127.0.1.1)
openpi_server-1  | INFO:websockets.server:server listening on 0.0.0.0:8000
openpi_server-1  | INFO:websockets.server:connection open
openpi_server-1  | INFO:openpi.serving.websocket_policy_server:Connection from ('127.0.0.1', 35274) opened
runtime-1        | INFO:root:Starting episode...
runtime-1        | Traceback (most recent call last):
runtime-1        |   File "/app/examples/aloha_sim/main.py", line 55, in <module>
runtime-1        |     tyro.cli(main)
runtime-1        |   File "/.venv/lib/python3.11/site-packages/tyro/_cli.py", line 191, in cli
runtime-1        |     return run_with_args_from_cli()
runtime-1        |            ^^^^^^^^^^^^^^^^^^^^^^^^
runtime-1        |   File "/app/examples/aloha_sim/main.py", line 50, in main
runtime-1        |     runtime.run()
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 35, in run
runtime-1        |     self._run_episode()
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 64, in _run_episode
runtime-1        |     self._step()
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/runtime/runtime.py", line 83, in _step
runtime-1        |     action = self._agent.get_action(observation)
runtime-1        |              ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/runtime/agents/policy_agent.py", line 15, in get_action
runtime-1        |     return self._policy.infer(observation)
runtime-1        |            ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/action_chunk_broker.py", line 29, in infer
runtime-1        |     self._last_results = self._policy.infer(obs)
runtime-1        |                          ^^^^^^^^^^^^^^^^^^^^^^^
runtime-1        |   File "/app/packages/openpi-client/src/openpi_client/websocket_client_policy.py", line 47, in infer
runtime-1        |     response = self._ws.recv()
runtime-1        |                ^^^^^^^^^^^^^^^
runtime-1        |   File "/.venv/lib/python3.11/site-packages/websockets/sync/connection.py", line 290, in recv
runtime-1        |     raise self.protocol.close_exc from self.recv_exc
runtime-1        | websockets.exceptions.ConnectionClosedError: no close frame received or sent
runtime-1 exited with code 1
openpi_server-1 exited with code 1
```
- Trial 1과 같은 메시지 출력 후 Server & Client 강제 종료
