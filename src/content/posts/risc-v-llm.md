---
title: 'Optimising LLM Inference on RISC-V with Custom GEMM Kernels'
description: 'Extending llama.cpp RISC-V GEMM tile from 4×8 to 4×16 — register pressure analysis, LLVM IR metrics, and cycle-accurate gem5 simulation showing a 31% improvement in cycles per output value.'
pubDate: 'Mar 23 2026'
author: 'Paddy McNabb'
tags: ['risc-v', 'llm', 'gemm', 'llama.cpp', 'rvv', 'gem5', 'performance']
draft: false
---

All source files — kernels, benchmarks, analysis scripts, and gem5 config — are in the [github repository](https://github.com/paddymac83/llama.cpp/tree/master/ggml/src/ggml-cpu/arch/riscv).

---

## Overview

This article documents a kernel optimisation experiment on the RISC-V Vector Extension (RVV): extending llama.cpp's existing 4×8 GEMM tile to a 4×16 two-pass variant for the Qwen2.5-0.5B model. The idea behind this work is that doubling the output tile width — by amortising activation (A) matrix loads across two weight column group passes — reduces effective memory traffic and improves throughput on bandwidth-constrained in-order RISC-V cores.

The experiment is fully software-defined: cross-compilation from x86-64, QEMU user-mode emulation for functional validation, LLVM IR and assembly analysis for register and instruction accounting, and gem5 MinorCPU simulation for cycle-accurate cache miss attribution.

The result: **4×16 delivers 2× the output columns for 1.38× the cycles — a 31% improvement in cycles per output value**, measured on a representative cycle-accurate in-order RISC-V pipeline.

---

## 1. Why Qwen2.5-0.5B

Qwen2.5-0.5B is a 24-layer decoder-only transformer with 896-dimensional hidden state, 14 query heads, 2 KV heads (GQA), and 4864-dimensional FFN intermediate size. It is small enough to run in QEMU emulation while remaining architecturally representative of the transformer family. Its weight matrices are large enough that the inner GEMM loop is a genuine compute bottleneck.

Each forward pass executes 217 batched matmul operations (169 weight projections + 48 attention operations). During prefill all weight projections take the GEMM path; during decode they degrade to GEMV as only a single token vector flows through the network.

### Kernel dispatch

llama.cpp loads the model in GGUF format and quantises weights offline to Q4_0 — 4-bit integers packed into 32-element blocks. At runtime, the ggml lib builds a computation graph of tensor operations and dispatches each node to a backend. On RISC-V, the CPU backend inspects the operand types and shapes to select the most specific kernel available. The dispatch chain runs: ggml_backend_cpu → ggml_compute_forward_mul_mat → shape and type checks → RVV kernel selection. 

The key shape parameter driving kernel selection is `ne01` — the row count of the activation tensor — which distinguishes a multi-row prefill batch (matrix multiplication) from a single-token decode step (vector multiplication):

```
ne01 ≥ 4  →  call ggml_gemm_q4_0_8x8_q8_0   (prefill)
ne01 = 1  →  call ggml_gemv_q4_0_8x8_q8_0   (decode)
```

Both the 4x8 tile and proposed 4x16 tile kernels share the same repacked B layout. The 4×16 optimisation affects only the GEMM (prefill) path.

### Register Pressure

Register pressure refers to the number of vector registers simultaneously live at any point in the kernel. RISC-V provides 32 physical vector registers (v0–v31). If a compiler needs more live values than there are physical registers it will spill to memory and reload it later.

Keeping peak pressure below 32 is a hard design constraint: a kernel that exceeds the physical register budget will spill and pay a runtime penalty. The register allocation for the 4×8 kernel at its most demanding point — inside the MAC chain — is covered below.

---

## 2. The 4×8 Baseline Kernel

[`ggml_gemm_q4_0_8x8_q8_0`](https://github.com/paddymac83/llama.cpp/tree/master/ggml/src/ggml-cpu/arch/riscv/repack.cpp) is a GEMM kernel operating on Q4_0 weights (4-bit, 32-element blocks) and Q8_0 activations (8-bit, 32-element blocks), designed for an 8×8 output tile. For RISC-V, a 4-row variant is implemented at VLEN=256 accumulated across `nb = K/32` K-blocks. For each K-block:

1. **Load B once** — unpack 256 nibbles into four `i8m2` vectors (`lo_0`, `lo_1`, `hi_0`, `hi_1`), to ve shared across all 4 activation rows
2. **Per row, repeated 4×** — load 4 × 8-byte A chunks, broadcast each across 8 column lanes, run a 4-step widening MAC chain (`vwmul` + 3× `vwmacc`) → `i16m4`, reduce 64 i16 → 8 i32, dequantise and accumulate into `sumf{0..3}
3. **Store** — write four `f32m1` accumulators to output matrix S

The widget below outlines the register basics (physical and logical registers), the peak register allocation for the 4x8 tile, the full register map and an initial comparison with the proposed 4x16 tile.

Peak register pressure during the MAC chain (`vwmul` + 3× `vwmacc`) leaves 7 vector registers free — enough headroom to consider a wider tile. The peak users are the RHS (weights) and LHS (activation row chunks), each occupying 8 physical registers simultaneously. 

The tile comparison illustrates the core tradeoff of the 4×16 design. The benefit is A amortisation: by processing 16 weight columns per A row load instead of 8, the kernel halves the number of times activation data must be fetched per unit of output. The risk is register pressure: the 8 additional f32m1 accumulators required to hold results for the second column group push the peak live register count from 25 to 29 of 32, leaving only 3 spare.

<iframe
  src="/widgets/register-pressure.html"
  width="100%"
  height="720px"
  frameborder="0"
  style="border:none;">
</iframe>

---

## 3. The 4×16 Two-Pass Design

A naive 4×16 tile — all 16 columns in one MAC chain — is impossible at VLEN=256. The RHS requires 16 physical registers to hold 16 weight columns. The MAC accumulator grows from i16m4 to i16m8 — 8 physical registers — because it must hold partial sums for 128 elements rather than 64 -> where each A row is broadcast across all 16 B columns. The 16 f32m1 output accumulators require 16 registers, on per B column. Total peak: 42 of 32 — impossible without spilling the operands to cache/memory.

[`ggml_gemm_q4_0_8x8_q8_0_4x16`](https://github.com/paddymac83/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/riscv/repack_4x16.cpp) is a two-pass interleaved design that solves this by processing 8 columns per pass within a single K-loop, reusing the A registers across both passes:

1. **Initialise 8 accumulators** — `sumf{0..3}` for column group 0 (`b_ptr0`, cols 0–7) and `sumf{4..7}` for column group 1 (`b_ptr1`, cols 8–15), persistent across all K-blocks
2. **Per K-block, per row, repeated 4×:**
   - Load A once into `lhs_0..3` (`i8m2` × 4 = 8 phys regs) — **held across both passes**
   - **Pass 1 (b_ptr0):** load B, run MAC chain → `i16m4`, reduce, dequantise, accumulate into `sumf{0..3}`, free RHS and MAC registers
   - **Pass 2 (b_ptr1):** load B into freed slots, repeat MAC chain using **same `lhs_0..3`** — no A reload — accumulate into `sumf{4..7}`, free all..
3. **Store** — eight `f32m1` accumulators, two `vse32` per row

Peak register budget: 8 sumf + 8 LHS + 8 RHS (one pass at a time) + 4 MAC + 1 b_scales = **29 of 32** — viable with 3 spare.

<iframe
  src="/widgets/register-pressure_4x16.html"
  width="100%"
  height="720px"
  frameborder="0"
  style="border:none;">
</iframe>

---

## 4. IR and Assembly Analysis

Both kernels were compiled to LLVM IR and RISC-V assembly using Clang with `-march=rv64gcv_zvl256b`. The 256-bit VLEN flag is required to satisfy a guard in the llama.cpp RISC-V kernels:

```bash
FLAGS="-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=gnu++17 --target=riscv64-linux-gnu"
clang++ $FLAGS $INCLUDES -emit-llvm -S repack.cpp      -o repack.ll
clang++ $FLAGS $INCLUDES -emit-llvm -S repack_4x16.cpp -o repack_4x16.ll
clang++ $FLAGS $INCLUDES -S repack.cpp      -o repack.s
clang++ $FLAGS $INCLUDES -S repack_4x16.cpp -o repack_4x16.s
```

### Key metrics (4×16 with 4x8 fallback path removed)

[`analyse.sh`](https://github.com/paddymac83/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/riscv/analyse.sh) extracts instruction counts from both the compiled LLVM IR and the generated RISC-V assembly, scanning for RVV intrinsic calls, vector loads and stores, MAC chain operations, reduction steps, stack spills, and arithmetic density. Running it against both kernel files produces a side-by-side comparison that quantifies the cost of the 4×16 design before any execution takes place.

**LLVM IR**

| Metric | 4×8 | 4×16 | Ratio | Note |
| :--- | ---: | ---: | :---: | :--- |
| Total IR instructions | 344 | 870 | 2.5× | Baseline for comparison |
| alloca (stack slots) | 1 | 8 | 8× | Scalar pointer spills — not vector register spills |
| RVV intrinsic calls | 94 | 201 | 2.1× | Slightly above 2× — compiler found limited CSE opportunities across the two passes |
| **load i64 (A reads)** | **16** | **16** | **1×** | **A loaded once per row regardless of tile width** |
| **vmv (A broadcast)** | **16** | **16** | **1×** | **Amortisation confirmed in IR** |
| vle vector loads | 2 | 16 | 8× | Two B blocks × 4 rows × 2 passes** |
| vse vector stores | 8 | 16 | 2× | Reflects doubled output columns exactly |
| vwmul + vwmacc | 16 | 32 | 2× | One MAC chain per row per pass |
| vnsrl (narrowing shift) | 24 | 48 | 2× | Reduction cascade runs twice |
| IR arithmetic density | 27.3% | 23.1% | ↓ | More scalar bookkeeping visible in IR |

**4×8 IR count reflects a single loop body (compiler did not unroll); 4×16 is fully unrolled across 4 rows × 2 passes × 2 loads per pass ->  instruction counts are only comparable when both kernels are compiled to the same level of unrolling. The correct number of vector loads (vle) for 4x8 is 8, not 2, when the 4-row iteration is fully unrolled (like 4x16).

**Assembly**

| Metric | 4×8 | 4×16 | Ratio | Note |
| :--- | ---: | ---: | :---: | :--- |
| **Total instructions** | **513** | **961** | **1.9×** | **Sub-2× for 2× output — net efficiency gain** |
| vsetvl | 27 | 61 | 2.3× | Per output: 0.84 vs 0.95 — nearly equal |
| Prologue sd sp | 13 | 13 | 1× | ABI cost unchanged between kernels |
| ASM arithmetic density | 24.4% | 28.1% | ↑ | Compiler recovers density at code-gen stage |

**The arithmetic density inversion** is the most striking finding. At IR level, 4×16 looks worse — extra address arithmetic for two B pointers to track passes inflates instruction count without adding vector ops. At assembly level the relationship flips: the compiler unrolls the longer loop body more aggressively, amortising branch overhead enabling more vector work per iteration. 4×16 produces 2.2× more vector instructions from only 1.9× more total instructions.

**On vsetvl:** RVV is a length-agnostic ISA — the same binary runs on any RISC-V vector register width (VLEN=256, 512, 1024 bits) without recompilation. `vsetvl` is the instruction that configures the vector unit before each operation, setting the element count (vl) and type (vtype — element width and LMUL) for the instructions that follow. 4x16 has more vsetvl operations in absolute terms, but it amortises them efficiently as its producing twice the output.

**On the alloca count:** assembly inspection confirms the 8 stack slots are scalar pointer spills — not vector register spills. The sumf accumulators remain in vector registers throughout. The prologue saves `ra, s0–s11` (13 callee-saved registers, unchanged between kernels) plus argument registers `a2, a3, a4` that the compiler needed for address calculations within the kernels. The spills come from the 4×16 kernel tracking two B-tile pointers simultaneously, exhausting the scalar integer register pool inside the K-loop.

---

## 5. QEMU Benchmark

QEMU user-mode emulation executes RISC-V binaries directly on an x86 host by translating each RISC-V instruction into equivalent host instructions at runtime. It provides a complete functional simulation of the RISC-V ISA including the vector extension, but has no cache model or pipeline model — every instruction costs the same regardless of whether its operands are in registers, L1 cache, or DRAM.

[`gemm_bench.cpp`](https://github.com/paddymac83/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/riscv/gemm_bench.cpp) was designed to model the Qwen2.5-0.5B attention and FFN tensor projection shapes under QEMU user-mode emulation with `vlen=256`:

```bash
qemu-riscv64 -cpu rv64,v=true,vlen=256,vext_spec=v1.0 -L /usr/riscv64-linux-gnu ./gemm_bench
```

### Qwen2.5-0.5B attention (K=896, N=896)

| M | 4×8 min (ns) | 4×16 min (ns) | 4×8 GFLOP/s | 4×16 GFLOP/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 21,667,913 | 32,472,193 | 0.296 | 0.198 | 0.667× |
| 8 | 46,073,879 | 41,548,833 | 0.279 | 0.309 | 1.109× |
| 16 | 92,672,967 | 136,586,692 | 0.277 | 0.188 | 0.678× |
| 32 | 191,088,432 | 274,994,541 | 0.269 | 0.187 | 0.695× |
| 64 | 376,808,267 | 558,980,374 | 0.273 | 0.184 | 0.674× |

### Qwen2.5-0.5B FFN (K=896, N=4864)

| M | 4×8 min (ns) | 4×16 min (ns) | 4×8 GFLOP/s | 4×16 GFLOP/s | Speedup |
| ---: | ---: | ---: | ---: | ---: | :---: |
| 4 | 117,812,407 | 176,571,224 | 0.296 | 0.197 | 0.667× |
| 8 | 267,176,155 | 385,490,756 | 0.261 | 0.181 | 0.693× |
| 16 | 554,295,041 | 796,233,147 | 0.252 | 0.175 | 0.696× |
| 32 | 1,081,070,628 | 1,558,141,402 | 0.258 | 0.179 | 0.694× |

The 4×16 kernel is consistently ~1.48× slower on QEMU across all shapes — a direct consequence of the instruction-count overhead identified in the assembly analysis. With 1.9× more total instructions and no cache model to reward the reduced memory traffic (reduced A loading), QEMU penalises 4×16 purely on instruction volume. 

The A amortisation benefit is a memory bandwidth saving that only manifests when memory access cost is non-uniform — invisible to QEMU, but directly measurable in gem5 and on real hardware.

---

## 6. gem5 Cycle-Accurate Simulation

[`gem5 v25.1`](https://github.com/gem5/gem5) is a cycle-accurate microarchitectural simulator that models a CPU pipeline and cache hierarchy at the level of individual clock cycles and memory transactions. Unlike QEMU, gem5 simulates cache hit and miss behaviour, pipeline stalls, and instruction-level timing — making it the ideal simulation tool for measuring memory traffic effects that QEMU cannot see.

[`gem5_riscv_minor.py`](https://github.com/paddymac83/llama.cpp/blob/master/ggml/src/ggml-cpu/arch/riscv/gem5_riscv_minor.py) configures a MinorCPU 4-stage in-order pipeline with a PrivateL1PrivateL2CacheHierarchy — 32kB L1 I/D, 512kB L2, 1.5GHz clock. This is a generic in-order pipeline model with cache parameters broadly representative of embedded RISC-V AI cores, though not a validated microarchitectural model of any specific core. Separate static binaries (bench_4x8, bench_4x16) were run independently so that each produces a clean stats.txt covering only its target kernel.

Configuration: 32kB L1 I/D, 512kB L2, 1.5GHz. Shape: M=4, N=896, K=896 (Qwen2.5-0.5B attention projection, minimum tile size).

### Final results (WARMUP=10, ITERS=50)

**Pipeline**

| Metric | 4×8 | 4×16 | Ratio | Interpretation |
| :--- | ---: | ---: | :---: | :--- |
| Retired instructions | 73,233,004 | 119,817,253 | 1.64× | Lower than 1.87× static — loop overhead amortised at runtime |
| Simulated cycles | 134,681,772 | 185,195,532 | 1.38× | 38% more cycles for 2× output |
| **IPC** | **0.544** | **0.647** | **↑19%** | **4×16 achieves better pipeline utilisation** |
| **Cycles per output** | — | — | **0.69×** | **31% efficiency improvement** |

**Cache**

| Metric | 4×8 | 4×16 | Ratio | Interpretation |
| :--- | ---: | ---: | :---: | :--- |
| **L1 overall miss rate** | **7.44%** | **4.63%** | **0.62×** | **38% lower — A amortisation confirmed at L1** |
| **L1 ReadReq miss rate** | **10.51%** | **6.58%** | **0.63×** | **Consistent amortisation signal** |
| L2 overall miss rate | 4.90% | 5.49% | 1.12× | Essentially equal — both caches warm |
| L2 data miss rate | 3.04% | 3.25% | 1.07× | Negligible difference |
| L2 total misses | 111,486 | 131,641 | 1.18× | Proportional to extra instruction volume |

### What the numbers mean

gem5 simulates cache and pipeline behaviour by observing the stream of memory addresses and instructions the RISC-V binary produces. gem5 measures the memory access patterns that results from those decisions: one cold A fetch per K-block paired with twice as many warm B fetches in 4×16, reducing the fraction of total memory traffic that is cold. The 38% L1 miss rate improvement is due to traffic composition, not cache residency. 

Assembly inspection and gem5 simulation are therefore complementary tools: the assembly reveals what the compiler decided to do with registers; gem5 measures what the hardware cache hierarchy experienced as a consequence of the memory access pattern those decisions produced.

**IPC (0.544 → 0.647, +19%).** The 4×16 kernel achieves better pipeline utilisation in steady state despite the scalar pointer spills identified in the assembly. The longer loop body — 1.64× more instructions per outer iteration — amortises branch overhead and produces a more pipeline-friendly instruction stream.

**L1 miss rate (7.44% → 4.63%, −38%).** The L1 miss rate improvement is a traffic composition effect: A data (cold activations, higher miss probability) represents 11% of total memory traffic in 4×16 vs 20% in 4×8, because the same A fetch is paired with twice as many B loads per unit of output.

**L2 behaviour is equal.** The L2 miss rates are essentially equal between the two kernels because the amortisation benefit is already fully realised before data reaches the cache hierarchy.

**Cycles per output: 0.69×.** 4×16 delivers 2× the output columns for 1.38× the cycles — a 31% improvement in cycles per output value.

---

## 7. Conclusions

| Finding | Evidence | Result |
| :--- | :--- | :---: |
| A amortisation works in IR | `load i64` and `vmv` counts both 1× despite 2× output width | ✓ |
| A amortisation works at runtime | L1 miss rate 38% lower for 4×16 in gem5 steady state | ✓ |
| 31% cycles-per-output improvement | 1.38× cycles for 2× output at WARMUP=10, ITERS=50 | ✓ |
| IPC improves (+19%) | Longer loop body amortises branch overhead in-order | ✓ |
| Scalar not vector spills | Prologue saves identical; inner-loop `ld sp` reloads are pointer values | ✓ |
| QEMU cannot predict the outcome | 1.48× slower on QEMU; 31% better on cycle-accurate simulator | ✓ |
| Numerical correctness | Max output diff = 0 across all shapes and iteration counts | ✓ |

The 4×16 tile is a sound optimisation for in-order RISC-V cores with RVV at VLEN=256. The design becomes more attractive as VLEN increases — at VLEN=512 the register budget relaxes further and the amortisation advantage grows proportionally. The remaining optimisation opportunity is reducing the scalar pointer pressure in the inner loop, which currently costs two `ld sp` reloads per iteration and limits IPC recovery.

**Next step:** validation on physical RISC-V hardware — Milk-V Jupiter (SpacemiT K1, X60 cores with RVV) — where DRAM bandwidth is the real bottleneck and the reduced memory traffic per output value from A amortisation will directly translate to throughput improvement.

---

## Appendix: Build Reference

```bash
# IR and assembly
FLAGS="-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=gnu++17 --target=riscv64-linux-gnu"
INCLUDES="-I.../ggml/include -I.../ggml/src -I.../ggml/src/ggml-cpu -I.../ggml/src/ggml-cpu/arch/riscv -I.../include"
clang++ $FLAGS $INCLUDES -emit-llvm -S repack.cpp      -o repack.ll
clang++ $FLAGS $INCLUDES -emit-llvm -S repack_4x16.cpp -o repack_4x16.ll
clang++ $FLAGS $INCLUDES -S repack.cpp      -o repack.s
clang++ $FLAGS $INCLUDES -S repack_4x16.cpp -o repack_4x16.s

# QEMU benchmark
riscv64-linux-gnu-g++ -O3 -march=rv64gcv -mabi=lp64d -std=gnu++17 $INCLUDES \
  gemm_bench.cpp repack.cpp repack_4x16.cpp -lm -o gemm_bench
qemu-riscv64 -cpu rv64,v=true,vlen=256,vext_spec=v1.0 -L /usr/riscv64-linux-gnu ./gemm_bench

# gem5 static binaries
FLAGS_STATIC="-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=gnu++17 -static"
riscv64-linux-gnu-g++ $FLAGS_STATIC $INCLUDES bench_4x8.cpp  repack.cpp repack_4x16.cpp -lm -o bench_4x8
riscv64-linux-gnu-g++ $FLAGS_STATIC $INCLUDES bench_4x16.cpp repack.cpp repack_4x16.cpp -lm -o bench_4x16

# gem5 runs
GEM5=~/gem5
$GEM5/build/RISCV/gem5.opt $GEM5/gem5_riscv_minor.py \
  --cmd=$(pwd)/bench_4x8 --l1d=32kB --l1i=32kB --l2=512kB --clock=1.5GHz
cp m5out/stats.txt m5out/stats_4x8.txt

$GEM5/build/RISCV/gem5.opt $GEM5/gem5_riscv_minor.py \
  --cmd=$(pwd)/bench_4x16 --l1d=32kB --l1i=32kB --l2=512kB --clock=1.5GHz
cp m5out/stats.txt m5out/stats_4x16.txt
```