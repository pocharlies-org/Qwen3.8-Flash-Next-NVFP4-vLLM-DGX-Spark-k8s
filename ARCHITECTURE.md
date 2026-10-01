# ARCHITECTURE.md — Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark-k8s

Receta pública (README + un manifiesto saneado) para servir `RadixArk/Qwen3.8-Flash-Next-NVFP4` en dos NVIDIA DGX Spark (GB10, `sm_121`) con vLLM, TP=2 + expert parallel, MTP-3 y CUDA graphs, como Deployments de Kubernetes. Tronco: **`main`**. No es un servicio desplegable: es documentación con un manifiesto de referencia. El despliegue real vive en otro repo (ver Dependencias).

## Clientes y versiones
- Sin clientes ni versiones propias: un README y un YAML (`qwen38-flash-next-nvfp4-vllm.yaml`, ConfigMaps `qwen38-flash-next-vllm-ple` y `qwen38-flash-next-vllm-launch` + Deployments `qwen38-flash-next-head` y `-worker`, namespace `llm`).
- Quien consume el modelo servido no mira este repo: llega por LiteLLM (`k8s-litellm-pocharlies`, `model_name: qwen38-flash-next`) y por el pool `tooling`.
- Hermano con otro motor para el mismo modelo y hardware: `pocharlies/qwen38-flash-next-dgx-spark-sglang`.

## Dependencias (ambos sentidos)
- **De** `getrefined/Qwen3.8-Flash-Next-NVFP4-vLLM-DGX-Spark` (lanzador y parche PLE originales), la imagen `vllm/vllm-openai:qwen38-flash-next` fijada por digest (`sha256:fc120ece…`) y el checkpoint `RadixArk/Qwen3.8-Flash-Next-NVFP4` en Hugging Face.
- **Producción real**: `k8s-ai-pocharlies` (`k8s/qwen38-flash-next-nvfp4-vllm.yaml`, namespace `llm`; la fuente de verdad del clúster). Este repo es la copia pública saneada: **no se aplica con `kubectl` al clúster** y no se mantiene en sincronía automática; ante diferencia manda `k8s-ai-pocharlies`.
- **Quién depende de este repo**: nadie en código.

## Stack
vLLM (imagen `vllm/vllm-openai:qwen38-flash-next`, build `0.1.dev…` de la rama del modelo), torch 2.13 + CUDA 13.0, NCCL sobre RoCE (200G ConnectX, enlace directo sin switch), MTP-3 con `--speculative-config`, parsers `qwen3_coder` (tools) y `qwen3` (reasoning). No se usa: `docker run` (son Deployments), KV cache NVFP4 (el QSA exige BF16: imposible), `VLLM_PLE_CPU_OFFLOAD` en multinodo (no soportado), DeepGEMM (`VLLM_USE_DEEP_GEMM=0`).

## Componentes compartidos (canónicos)
Ninguno. Lo canónico de la familia: manifiestos y pines en `k8s-ai-pocharlies`; enrutado en `k8s-litellm-pocharlies`; selector de perfil y árbitro en `dgx-infra` (`services/dashboard/compute_mode.py`, `gpu_arbiter.py`).

## Cómo se construye aquí
- El lanzador `launch.sh` (ConfigMap) es el mismo para head y worker; el worker (`VLLM_NODE_RANK != 0`) lleva `--headless`. Sin eso el follower arranca un servidor completo, muere, y el head se cuelga en NCCL sin log durante ~34 min.
- Reiniciar head y worker JUNTOS (un rollout independiente deja el worker en un process group viejo).
- Para cargas y réplicas en el clúster DGX **se lee antes el ConfigMap `gpu-arbiter-state` (ns `comfyui`)**: `kubectl get cm -n comfyui gpu-arbiter-state -o jsonpath='{.data.compute_mode}'`. Con `llm-tp` efectivo los Sparks son del LLM residente y no cabe otra carga; con `phase: switching` no se toca nada. Las réplicas del residente las posee `compute_mode.py` de dgx-infra: **los `kubectl scale` del README son para quien replica la receta en un clúster propio; en el clúster de Dani nunca a mano (se revierten en segundos)**.
- Mediciones: compara `steps/s`, no `tok/s` entre prompts distintos; un cambio solo es creíble por encima de ~5 % (ruido medido del 27 %).

## Tests y validaciones
No hay tests ni CI en el repo. Validación documentada: arranque limpio (`0/N` de CUDA graphs en el log), `vllm:num_requests_running` a 0 antes de medir, y las tablas de benchmark del README (fechadas; no son el estado actual del clúster).

## CI/CD y despliegue
Ninguno aquí: ni ArgoCD ni workflows. El despliegue ocurre en `k8s-ai-pocharlies` (pin de imagen y manifiesto) y el residente lo gobierna el árbitro de dgx-infra.

## Decisiones y trampas
- Trampa nº 1: `--headless` en el follower; nº 2: reiniciar ambos rangos juntos.
- `NCCL_IB_HCA` con `=` inicial (coincidencia exacta); la autonegociación del enlace RDMA no sobrevive a un reinicio.
- Palancas medidas que NO ayudan están listadas en el README («Falsified levers»): no repetirlas.
- Solape con `qwen38-flash-next-dgx-spark-sglang` (mismo modelo, otro motor): ver su `ARCHITECTURE.md`; qué motor es el residente se mira (`kubectl get pods -n llm`), no se afirma aquí.
