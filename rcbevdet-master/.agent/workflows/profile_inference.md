# Workflow: RCBEVDet Inference Profiling

This workflow describes how to use **NVIDIA Nsight Systems** and **Nsight Compute** to identify compute and memory bottlenecks in the RCBEVDet inference pipeline.

---

## Prerequisites
- NVIDIA Nsight Systems (`nsys`) and Nsight Compute (`ncu`) must be installed (verified at `/usr/local/cuda-12.8/bin/`).
- A trained checkpoint or the pre-trained weights.

---

## Step 1: Macro-Profiling with Nsight Systems (NSYS)
Use NSYS to get a high-level overview of the execution timeline, including CPU-GPU synchronization, memory copies (HtoD/DtoH), and kernel durations.

### 1.1 Preparation
To avoid massive profile files, we will limit the inference to the first 5 samples. Create a temporary config or use `--cfg-options`:

### 1.2 Execution
Run the following command to capture the trace:

```bash
nsys profile \
    --trace=cuda,cudnn,cublas,osrt,nvtx \
    --output=rcbevdet_nsys_report \
    --force-overwrite=true \
    --stop-on-exit=true \
    python tools/test.py \
    configs/rcbevdet/rcbevdet-v2x-radar.py \
    checkpoints/rcbevdet_ckpt/rcbevdet.pth \
    --eval kitti
```
*Tip: If the script takes too long, you can terminate it with `Ctrl+C` after ~10-20 iterations; nsys will still save the captured data.*

### 1.3 Analysis
Open `rcbevdet_nsys_report.nsys-rep` in the Nsight Systems UI on your local machine.
- **Look for**: Large gaps in the GPU timeline (CPU bottlenecks).
- **Look for**: Long-running kernels. In RCBEVDet, look specifically for `bev_pool` or `view_transformer` kernels.
- **Identify**: Which specific kernel (e.g., ResNet-Backbone vs. BEV-Pooling) consumes the highest percentage of GPU time.

---

## Step 2: Micro-Profiling with Nsight Compute (NCU)
Once you identify a specific "hot" kernel (e.g., the BEV pooling kernel), use NCU to perform a deep-dive analysis of its compute and memory efficiency.

### 2.1 Identify the Kernel Name
From the NSYS report in Step 1, find the mangled name of the kernel you want to profile (e.g., `void bev_pool_v2_kernel...`).

### 2.2 Execution
Run the profile for that specific kernel. Profiling with NCU is slow, so we limit it to **1 sample** and **specific kernel names**:

```bash
ncu --target-processes all \
    --kernel-name "regex:bev_pool" \
    --launch-skip 5 \
    --launch-count 1 \
    --export rcbevdet_ncu_report \
    --force-overwrite \
    python tools/test.py \
    configs/rcbevdet/rcbevdet-v2x-radar.py \
    checkpoints/rcbevdet_ckpt/rcbevdet.pth \
    --eval kitti
```

### 2.3 Analysis
Open `rcbevdet_ncu_report.ncu-rep` in the Nsight Compute UI.
- **SOL (Speed of Light) Analysis**: Check the "Compute Workload Analysis" and "Memory Workload Analysis".
    - If **Compute SOL > Memory SOL**: The kernel is **Compute Bound**. Optimization should focus on instruction throughput or math operations.
    - If **Memory SOL > Compute SOL**: The kernel is **Memory Bound**. Optimization should focus on coalesced access, reducing global memory traffic, or better cache utilization.
- **Occupancy**: Check if the GPU is fully utilized (Number of active warps vs. maximum).

---

## Step 3: Optimization Decisions
- **If HtoD/DtoH is high**: Consider pinning memory or overlapping transfers with computation.
- **If BEV Pool is the bottleneck**: Look into fused kernels or half-precision (FP16/AMP).
- **If Backbone is the bottleneck**: Consider TensorRT export or model pruning.
