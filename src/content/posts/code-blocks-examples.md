---
title: 'RISC-V LLM Code Reference'
description: 'A concise reference for RISC-V vector intrinsics, QEMU emulation, gem5 simulation, and llama.cpp kernel patterns used in LLM inference optimisation.'
pubDate: 2026-03-23
author: 'Paddy McNabb'
tags: ['risc-v', 'rvv', 'llama.cpp', 'gem5', 'qemu', 'code']
draft: false
---

A working reference for the key code patterns used across this blog — cross-compilation, QEMU emulation, gem5 simulation, and RVV kernel snippets.

---

## Cross-Compilation Flags

```bash
# Clang — emit LLVM IR
FLAGS="-O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=gnu++17 --target=riscv64-linux-gnu"
clang++ $FLAGS -emit-llvm -S kernel.cpp -o kernel.ll

# Clang — emit assembly
clang++ $FLAGS -S kernel.cpp -o kernel.s

# GCC — shared object for QEMU
riscv64-linux-gnu-g++ -O3 -march=rv64gcv -mabi=lp64d -std=gnu++17 \
  kernel.cpp -lm -o kernel_bench

# GCC — static binary for gem5
riscv64-linux-gnu-g++ -O3 -march=rv64gcv_zvl256b -mabi=lp64d -std=gnu++17 -static \
  kernel.cpp -lm -o kernel_bench_static
```

---

## QEMU User-Mode Emulation

```bash
# Run with RVV enabled at VLEN=256
qemu-riscv64 \
  -cpu rv64,v=true,vlen=256,vext_spec=v1.0 \
  -L /usr/riscv64-linux-gnu \
  ./kernel_bench

# Confirm VLEN at runtime inside your binary
#include <riscv_vector.h>
size_t vlen_bytes = __riscv_vlenb();  // e.g. 32 for VLEN=256
```

---

## gem5 Cycle-Accurate Simulation

```bash
GEM5=~/gem5

# Run MinorCPU with private L1/L2 cache hierarchy
$GEM5/build/RISCV/gem5.opt $GEM5/gem5_riscv_minor.py \
  --cmd=$(pwd)/kernel_bench_static \
  --l1d=32kB --l1i=32kB --l2=512kB --clock=1.5GHz

# Save stats per kernel
cp m5out/stats.txt m5out/stats_4x8.txt
```

Key stats to extract from `stats.txt`:

```bash
grep "simInsts"            m5out/stats.txt   # retired instructions
grep "numCycles"           m5out/stats.txt   # simulated cycles
grep "ipc"                 m5out/stats.txt   # IPC
grep "overall_miss_rate"   m5out/stats.txt   # L1/L2 miss rates
```

---

## RVV Kernel Patterns

### Load and broadcast activation (A matrix)

```cpp
// Load 8 bytes of A, broadcast across all lanes
vint8m2_t lhs = __riscv_vle8_v_i8m2(a_ptr, vl);
vint16m4_t acc = __riscv_vwmul_vx_i16m4(b_vec, a_ptr[0], vl);
```

### Widening MAC chain (4×8 tile, one row)

```cpp
vint16m4_t acc = __riscv_vwmul_vx_i16m4(b0, *a_ptr++, vl);
acc = __riscv_vwmacc_vx_i16m4(acc, *a_ptr++, b1, vl);
acc = __riscv_vwmacc_vx_i16m4(acc, *a_ptr++, b2, vl);
acc = __riscv_vwmacc_vx_i16m4(acc, *a_ptr++, b3, vl);
```

### Reduce i16 → i32 and dequantise

```cpp
vint32m4_t sum32 = __riscv_vwredsum_vs_i16m4_i32m1(acc, zero, vl);
float result = (float)__riscv_vmv_x_s_i32m1_i32(sum32) * scale;
```

### 4×16 two-pass: A loaded once, reused across both passes

```cpp
// Load A once
vint8m2_t lhs_0 = __riscv_vle8_v_i8m2(a_ptr,      vl);
vint8m2_t lhs_1 = __riscv_vle8_v_i8m2(a_ptr +  8, vl);
vint8m2_t lhs_2 = __riscv_vle8_v_i8m2(a_ptr + 16, vl);
vint8m2_t lhs_3 = __riscv_vle8_v_i8m2(a_ptr + 24, vl);

// Pass 1 — cols 0–7 (b_ptr0), using lhs_0..3
// ... MAC chain into sumf{0..3} ...

// Pass 2 — cols 8–15 (b_ptr1), reusing same lhs_0..3 — no reload
// ... MAC chain into sumf{4..7} ...
```

---

## Useful IR Metrics

```bash
# Count RVV intrinsic calls in LLVM IR
grep -c "llvm.riscv" kernel.ll

# Count A loads (i64 scalar loads from activation pointer)
grep -c "load i64" kernel.ll

# Count vector broadcasts
grep -c "vmv.v.x\|vslide\|vmv.s.x" kernel.ll

# Count stack slots (potential spills)
grep -c "alloca" kernel.ll
```

---

## llama.cpp Kernel Dispatch (ggml-cpu.cpp)

```cpp
// GEMM path (prefill): ne01 >= 4
if (ne01 >= 4) {
    ggml_gemm_q4_0_8x8_q8_0(/* ... */);  // or 4x16 variant
}

// GEMV path (decode): ne01 == 1
else {
    ggml_gemv_q4_0_8x8_q8_0(/* ... */);
}
```

The VLEN guard that selects the RVV kernel path:

```cpp
#if defined(__riscv_v_intrinsic)
if (__riscv_vlenb() >= 32) {
    // VLEN=256 path — use optimised RVV kernel
}
#endif
```
