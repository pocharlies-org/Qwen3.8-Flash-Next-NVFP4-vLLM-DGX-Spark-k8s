# Qwen3.8-Flash-Next NVFP4 on 2× DGX Spark — vLLM on Kubernetes

Deployment of [`RadixArk/Qwen3.8-Flash-Next-NVFP4`](https://huggingface.co/RadixArk/Qwen3.8-Flash-Next-NVFP4)
across **2× NVIDIA DGX Spark (GB10, `sm_121`)** with TP=2 + expert parallel, MTP-3 speculative
decoding and CUDA graphs — running as **Kubernetes Deployments**, not `docker run`.

This is a companion to
[getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark),
whose launcher and PLE patch this deployment is built on. Same image, same checkpoint, same
digest (`vllm/vllm-openai@sha256:fc120ece…`).

**What this repo adds is the k8s translation and the things that only bite there** — plus a list
of levers that turned out not to work, so nobody else spends a night on them.

---

## The one thing that will cost you a night: `--headless`

In vLLM multi-node with `--distributed-executor-backend mp`, the follower (`--node-rank != 0`)
**must** be launched with `--headless`. Without it the follower starts a full server — API server
*and* its own EngineCore — and that EngineCore calls `collective_rpc("get_kv_cache_spec")`, which
is leader-only:

```
AssertionError: collective_rpc should not be called on follower node
  vllm/v1/executor/multiproc_executor.py:385
```

The follower dies. The head then hangs in an NCCL collective waiting for a rank-1 from a dead
process group: **34 minutes with not a single log line and no restart**, because a lenient
`startupProbe` (120 × 30 s) never fires.

The symptom points nowhere near the cause. Things it is *not*, all measured before we found it:

| suspected | evidence against |
|---|---|
| RDMA link down | all four ports `ACTIVE`, `200 Gb/sec (4X HDR)` on both nodes |
| memory | the worker node had **111 GiB free** — precisely because the follower died before loading weights |
| our own config changes | the hang predated them by 92 minutes |

The upstream recipe never hits this because it launches the follower itself with a separate
`docker run`. In k8s **both Deployments run the same launch script**, so they have to be told
apart:

```bash
if [ "${VLLM_NODE_RANK}" != "0" ]; then
  ARGS+=( --headless )
fi
```

`--headless` sets `api_server_count=0` (`serve.py:65-72`); `run_headless` then sees
`node_rank_within_dp > 0` and builds the follower `MultiprocExecutor` — the branch the code itself
comments as *"Run headless workers (for multi-node PP/TP)"* (`serve.py:213`).

## Second thing: restart both ranks together

A normal rollout moves head and worker independently. The worker survives in a stale process group
and the head crash-loops against it. Recycle both:

```bash
kubectl scale deploy -n llm qwen38-flash-next-head   --replicas=0
kubectl scale deploy -n llm qwen38-flash-next-worker --replicas=0
# wait for zero pods
kubectl scale deploy -n llm qwen38-flash-next-worker --replicas=1
kubectl scale deploy -n llm qwen38-flash-next-head   --replicas=1
```

---

## Benchmarks

**Read the caveats before the numbers.** Measured with streaming, `temperature 0`, decode rate =
`(tokens−1)/(end − first_token)` — i.e. excluding TTFT. Median of 4–6 reps.

### Thinking off / `reasoning_effort: low` — responses that actually finish

| content | effort | tokens | finish | accept | steps/s | **decode tok/s** | σ |
|---|---|---|---|---|---|---|---|
| code | `low` | 4644 | length/stop | 3.569 | 15.18 | **54.2** | 3.01 |
| code | **off** | 2914 | **stop** | 3.716 | 16.35 | **60.8** | 0.25 |
| csharp | `low` | 3794 | **stop** | 3.373 | 16.49 | **55.6** | 0.41 |
| csharp | **off** | 2532 | **stop** | 3.543 | 16.57 | **58.7** | 0.48 |

### Thinking on (default) — what we measured first, and why it was wrong

| content | tok/s | accept | steps/s |
|---|---|---|---|
| code | 43.8 | 2.639 | 16.6 |
| reasoning | 47.4 | 2.909 | 16.3 |
| math | 56.7 | 3.434 | 16.5 |
| csharp | 40.2 | 2.428 | 16.6 |

Those code/csharp figures **do not measure code generation**. With thinking on and a 4000-token
budget the model spends the budget reasoning and returns *empty content*:

```
thinking on :  total=4000  finish=length  reasoning=15700 chars  content=0
thinking off:  total=2881  finish=stop    reasoning=0            content=10801
```

Acceptance tells the same story: 2.639 → 3.569 on code once it actually writes code. The draft
predicts real code far better than a reasoning stream.

### The decomposition that makes this legible

`tok/s = steps/s × accept_length`. **Steps/s is constant across content types** (16.3–16.6); every
difference between categories is MTP acceptance. Formulaic maths accepts 3.43 tokens per draft,
C# with XML docs and generics accepts 2.43.

If you compare two deployments, compare `steps/s` — comparing `tok/s` across different prompts
compares different workloads.

### Caveats, in order of how much they should worry you

1. **Measurement noise is large.** Identical config, identical prompt, `temperature 0`, 8 reps:
   `37.9 43.9 46.3 40.7 42.0 42.6 42.7 49.6` → mean 43.2, **27% range**. With 6 reps per category σ
   drops to ~1.2–1.6 (≈3%). **A config change is only credible above ~5%.** We attributed five
   improvements and one regression before measuring this; all were inside the noise.
2. **These numbers come from a cluster that also serves live traffic.** Two later measurement runs
   were contaminated by 2–5 concurrent production requests and showed step rates collapsing to
   7 steps/s. Check `vllm:num_requests_running` *before* you measure, not after you are surprised.
3. **We do not claim a comparison against the upstream recipe.** Their probes complete in 545–1200
   tokens; ours ran to 4000+ with thinking on. Different output lengths are different workloads.

---

## Falsified levers — measured, not argued

Everything here was tried on this hardware and did **not** help. Listed so you can skip them.

| lever | result |
|---|---|
| **KV cache in NVFP4** | **Impossible.** `NotImplementedError: Qwen3.8-Flash-Next QSA requires a BF16 main KV cache` (`qwen3_8_flash_next/nvidia/qsa.py`). The QSA rejects *any* non-BF16 KV, not just `fp8_e4m3`. Blind spot: vLLM **accepts** the flag and logs *"Using nvfp4 data type to store kv cache… boosts the performance"* (`cache.py:282`) before rejecting it at backend construction. |
| **`VLLM_PLE_CPU_OFFLOAD=1`** | **Not available multinode.** `ValueError: … Unsupported settings: nnodes=2` (`gpu_worker.py:247`). Which means the upstream recipe did not use it either on its two Sparks. |
| `cudagraph_capture_sizes` | Sizes are in **sequences**, not tokens, and vLLM silently clamps them to `max_num_seqs`. Asking for `[1,2,4,8,16,24,32,40]` with `max_num_seqs=10` captures 4. Check the `0/N` in the startup log. |
| GPU clock cap on the worker | 2200 → 2600 MHz gave +1% (noise). Clock rose (2411 MHz, 61 °C) — the workload is memory-bound, not compute-bound. |
| `NCCL_CUMEM_ENABLE=1` | Worse across all five categories, though within noise. `0` is the right value here. |
| Removing `--enable-prefix-caching` | **No-op** — vLLM V1 enables it by default. You would need `--no-enable-prefix-caching`, which you do not want: prefix caching is what helps agentic multi-turn load. |
| 10 min soak | Acceptance went **down** (2.755 → 2.517), not up. |
| Power / thermal / CPU | No throttling active. `nr_throttled=0`, no CPU limit, `performance` governor, 2808 MHz. |
| RoCE bandwidth | 5.40 MB/token = **0.2 ms** of wire time, 0.8% of a 25 ms token budget. Not the bottleneck. |
| DCGM profiling | 0 `DCGM_FI_PROF_*` metrics collected; it is not stealing GPU cycles. |

### Where the time actually goes

`torch.profiler` on a real 150-token generation (4378 ms of kernel time):

| family | % |
|---|---|
| GEMM / MoE experts | **63.2** |
| other | 22.9 |
| NCCL collectives | 10.2 |
| elementwise / copies | 1.7 |
| embedding / gather (PLE) | 1.3 |
| norm | 0.6 |
| attention | 0.1 |

Top kernel is `cutlass_80_wmma_tensorop_bf16_s161616gemm_bf16_16x16_128` — 35 000 calls, **43%**.
Those are SM80 (Ampere) WMMA kernels running on SM121, and that is **not** a defect: for the skinny
decode shapes (M=1–4) they hit **98–100% of the memory roofline**. Forcing cuBLASLt or classic
cuBLAS both made it slower. You cannot beat memory bandwidth with a better kernel.

The NVFP4 experts *do* use the modern path — `MainloopSm120ArrayTmaWarpSpecializedBlockScaled`
(14.7%). The BF16 kernels are the modules the checkpoint excludes from quantisation
(`quantization_config.ignore`: `mtp.*`, `*.self_attn.*`, `*.linear_attn.*`, `*.mlp.gate*`,
`*.mlp.shared_expert.*`, `lm_head`), and with MTP-3 the draft runs 3 of the 4 forwards per step.

---

## `VLLM_USE_DEEP_GEMM=0`

Preventive, credit to [vllm#54125](https://github.com/vllm-project/vllm/issues/54125). On `sm_121`
`support_deep_gemm()` accepts the whole `120` capability family, so the gate opens on GB10 —
verified here: `support_deep_gemm() = True`. With a blockwise-FP8 layer in front it selects
`DeepGemmFp8BlockScaledMMKernel` and takes a CUDA fault during the startup `profile_run`.

It does **not** affect this checkpoint (`quant_algo NVFP4`, weight schemes `{(4 bits, None, None)}`,
no `FP8_PB_WO`; zero `Selected …BlockScaledMMKernel` lines, zero DeepGEMM kernels in the profiler
trace out of 173). Set anyway because it costs nothing and the failure mode is
`Engine core initialization failed. Failed core proc(s): {}` with the DeepGEMM frame buried — the
same generic message that hid the `--headless` bug for 34 minutes.

## Also worth knowing

- **`NCCL_IB_HCA` needs a leading `=`** for exact match. Without it NCCL matches by *prefix*, so
  `rocep1s0f0` also matches `rocep1s0f1`. With expert parallel this stops being cosmetic — the
  all2all sprays traffic across every listed port.
- The **RDMA link autoneg does not survive a reboot** on this hardware; pin it
  (`ethtool -s … autoneg off speed 200000` on both ends) or the head sits in Init behind a
  misleading NFS error.
- The upstream README's `sync; echo 3 > /proc/sys/vm/drop_caches` before load is worth honouring on
  unified memory.

## Environment

| | |
|---|---|
| nodes | 2× DGX Spark (GB10, `sm_121`), 200G ConnectX RoCE |
| driver / CUDA | `580.173.02` / 13.0 |
| DGX OS / kernel | 7.2.3 / `6.17.0-1031-nvidia` |
| image | `vllm/vllm-openai:qwen38-flash-next` (`sha256:fc120ece…`) |
| vLLM / torch | `0.1.dev20073+g8e685d198` / `2.13.0+cu130` |
| measured memory bandwidth | 222–233 GB/s streaming (both nodes) |

## Files

- `qwen38-flash-next-nvfp4-vllm.yaml` — the full manifest, sanitised. Replace `HEAD_USER`,
  `WORKER_USER`, `10.0.0.x`, `example.com` and the node names.

## Credit

The launcher, the PLE FP8 patch and the original benchmark methodology come from
[getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark](https://github.com/getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark).
The DeepGEMM gate issue is [vllm#54125](https://github.com/vllm-project/vllm/issues/54125), reported
by someone else.
