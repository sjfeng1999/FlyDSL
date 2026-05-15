# FP8 4-wave GEMM LDS-Alias 性能问题分析

> 一份完整的根因分析文档：**为什么 `fp8_gemm_4wave` 改成单一 dyn-shared 后性能掉，怎么修最干净？**
>
> 适用对象：`kernels/fp8_gemm_4wave.py` 在 8192³ FP8 GEMM 上的性能与正确性。
> 涉及组件：FlyDSL 的 `fly.add_offset` lowering、`ptrtoint/inttoptr`、`!alias.scope` 元数据、AMDGPU 后端的 `BasicAA` / `SIInsertWaitcnts` / `MachineScheduler`。

---

## 0. 速览（TL;DR）

| 关键事实 | 含义 |
|---|---|
| HEAD 的 `_lds_dst_at` 用 `ptrtoint+add+inttoptr` "破坏 alias chain" 这条**注释是错的** | 该注释只是误传。`ptrtoint(@named_global)+add+inttoptr` 的 underlying object 仍是 `@named_global`，BasicAA 完全识别。 |
| HEAD 之所以快，**根本原因是 8 个独立 LDS 全局**（`@A_lds_cur_0` … `@B_lds_next_1`）。 | BasicAA 通过 underlying-object 分析直接得出 NoAlias —— 跟有没有 ptrtoint 无关。 |
| Work tree 改成单一 `@__dynamic_shared__0` 后，所有 LDS 访问的 underlying object 都一样。 | BasicAA 失明 → 后端的 `SIInsertWaitcnts` 转入保守模式 → `s_waitcnt vmcnt(N)` 从 154 暴涨到 1141，性能从 2750 TFLOPS 跌到 690 TFLOPS。 |
| 当前 work tree 的修复用 `!alias.scope` 元数据让 LDS 读跟其他访存"看起来 noalias"。 | **是 hack**：依赖 AMDGPU 后端"一边有 scope、一边没有 → 视为 noalias"的**未文档化**启发式。`MachineScheduler` 可能也用这条规则做指令重排。 |
| 真正干净的方案早就存在：`kernels/preshuffle_gemm_v2.py` 用 `Fly_Swizzle` + `composed_layout`。 | LDS 访问最终是 `getelementptr ptr addrspace(3) @__dynamic_shared__0, i32 %swz_offset`，**零 ptrtoint / 零 alias scope / 零 hack**，BasicAA 凭 GEP + 地址空间分析自然 work。 |

**结论**：当前 work tree 的 alias-scope pass 是一个有隐患的 workaround，可以工作但语义上不正确；真正的根治方案是把 `fp8_gemm_4wave` 的 swizzle 从 `CoordSwizzleType + crd2idx().to_py_value() + ptrtoint` 改写成 `SwizzleType + composed_layout + fx.add_offset`（即 `preshuffle_gemm_v2` 的范式）。

---

## 1. 问题背景

### 1.1 输入

`fp8_gemm_4wave` 是一个 8192³ FP8 GEMM kernel（在 gfx950/CDNA4 上）。它有两个版本：

- **HEAD `74bedbac`（"static-LDS" 版）**：用 `SmemAllocator` 分配 **8 个独立的 LDS 符号**
  （`@A_lds_cur_0`、`@A_lds_cur_1`、…、`@B_lds_next_1`）。每个 `[16384 x i8] addrspace(3)`。
- **Work tree 改动（"dyn-shared" 版）**：改用 `fx.SharedAllocator(...).allocate(_LDSStorage)`
  内部所有 sub-buffer 共享一个 `@__dynamic_shared__0 : [0 x i8] addrspace(3)`，
  通过运行时字节偏移寻址。

两版 kernel 的算法、tiling、swizzle 形状完全一致。

### 1.2 现象

完全等效的算法，性能差距巨大：

| 状态 | 性能 (TFLOPS, 8192³) | ISA 中 `s_waitcnt vmcnt(N)` 条数 |
|---|---:|---:|
| HEAD（static-LDS 8 sym） | ~2750 | 154 |
| Work tree（dyn-shared 单 sym）**不**做任何 alias-scope 处理 | ~690 | ~1141 |
| Work tree + alias-scope pass（当前修复） | ~2810 | 148 |

ISA 中 `ds_read` / `ds_write` / `v_mfma` / `buffer_load` 的数量在三种状态下**完全一致**。
唯一变化的是 `s_waitcnt vmcnt`。

---

## 2. 根因：AMDGPU 后端的 alias 分析视角

### 2.1 AMDGPU `SIInsertWaitcnts` 的判断逻辑

主循环里穿插两类访存：

- **G→LDS 写**：`buffer_load_dwordx4 ... offen lds` —— 全局读 + LDS 写，**消耗 vmcnt**。
- **LDS→reg 读**：`ds_read_b128` —— **消耗 lgkmcnt**。

`SIInsertWaitcnts` 给每条 LDS 读决定要不要在前面插 `s_waitcnt vmcnt(N)`：

> "下一条 LDS 读访问的地址，是否**可能跟前面正在飞的 vmcnt-bearing 操作**（即 `buffer_load_lds` 的 LDS dst）**alias**？如果可能 alias，必须等 vmcnt ≤ N。"

判断"是否 alias"靠的是 `MachineMemOperand::AAInfo` 和 LLVM 的 alias analyses。

### 2.2 LLVM BasicAA 怎么看 LDS 指针

BasicAA 的关键分析步骤是 **追溯 underlying object**（具名 global / alloca / kernel arg）。它对每个 ptr 调用 `getUnderlyingObject(p)`，然后在比较两个 ptr 时：

- **不同 underlying object** → 立即 **NoAlias**（多数情形）
- **相同 underlying object** → 再比较两个指针的 GEP offset 区间是否重叠（"VariableGEP" 分析）

对 LDS 而言还有一条**额外捷径**：`addrspace(3)` vs `addrspace(1)` 是**不同地址空间** → 必定 NoAlias。这条规则适用于 LDS 跟全局内存之间，但**不适用于** LDS 内部的子区域互相之间。

### 2.3 HEAD 凭"underlying object 不同"取胜

HEAD 的 LLVM IR（实测 dump）：

```llvm
@A_lds_cur_0 = external addrspace(3) global [16384 x i8], align 1024
@A_lds_cur_1 = external addrspace(3) global [16384 x i8], align 1024
... (8 个 LDS 符号)

; LDS 写（buffer.load.lds 的 dst 参数）：
%137 = add i64 ptrtoint (ptr addrspace(3) @A_lds_cur_0 to i64), %136
%138 = inttoptr i64 %137 to ptr addrspace(3)
tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(... %138 ...)

; LDS 读（vector load，直接 GEP！）：
%241 = getelementptr i8, ptr addrspace(3) @A_lds_cur_0, i64 %240
%242 = load <16 x i8>, ptr addrspace(3) %241, align 1
```

关键观察：

- LDS 读侧用的是**直接 GEP from named global** —— 完全没有 ptrtoint。
- LDS 写侧用的 `add ptrtoint(@named_global) + ... → inttoptr` 模式，**BasicAA 仍然识别** `@A_lds_cur_0` 作为 underlying object。
- 当 BasicAA 比较"对 `@A_lds_cur_0` 的写"和"对 `@B_lds_cur_0` 的读"时 → 不同 underlying object → **NoAlias**。
- `SIInsertWaitcnts` 看到 LDS 读跟前一条 G→LDS 写 NoAlias → 不需要等 vmcnt → 154 条 vmcnt 就够。

### 2.4 Work tree 单 sym 让 BasicAA 失明

Work tree 改成单一 `@__dynamic_shared__0` 后，IR 形如：

```llvm
@__dynamic_shared__0 = external dso_local addrspace(3) global [0 x i8], align 1024

; LDS 写：
%off  = add i32 ptrtoint(@__dynamic_shared__0 to i32), %wave_offset
%dst  = inttoptr i32 %off to ptr addrspace(3)
tail call void @llvm.amdgcn.raw.ptr.buffer.load.lds(... %dst ...)

; LDS 读：
%off2 = add i32 ptrtoint(@__dynamic_shared__0 to i32), %swz_offset
%src  = inttoptr i32 %off2 to ptr addrspace(3)
%v    = load <16 x i8>, ptr addrspace(3) %src
```

问题：

- 所有 LDS 访问的 underlying object 都是同一个 `@__dynamic_shared__0` → BasicAA underlying-object 分析得出 **MayAlias**。
- 即使两个 offset 是**编译期常量**，BasicAA 的 VariableGEP 分析也无效 —— 因为指针不是直接 GEP 出来的，**`inttoptr` 切断了 BasicAA 的 GEP 偏移跟踪**。
- BasicAA 失明 → `SIInsertWaitcnts` 必须假设每条 LDS 读都可能跟之前的 G→LDS 写冲突 → 每条 `ds_read` 前插 `s_waitcnt vmcnt(0)` → 1141 条 vmcnt → 性能塌方。

### 2.5 数字证据

| 状态 | LLVM IR 中 `ptrtoint`/`inttoptr` 数 | 最终 ISA `vmcnt` 数 | TFLOPS |
|---|---:|---:|---:|
| HEAD（8 sym） | 32 / 32（每个 LDS 符号 4 条） | 154 | 2750 |
| Work tree（1 sym）+ 无 alias-scope | 32 / 32（同一全局） | 1141 | 690 |
| Work tree（1 sym）+ alias-scope on LDS reads | 32 / 32 | 148 | 2810 |
| preshuffle_gemm_v2（1 sym, Fly_Swizzle） | **0 / 0** | 41 | ✓（无 hack） |

---

## 3. 当前 work tree 的"修复"

### 3.1 做了什么

在 `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp` 加了一个 pass `annotateSharedAllocAliasScopes`：

- `Arena.allocate()` 在 Python 端给生成的根 `fly.add_offset` 打 `fly.alloc_id` 标签。
- `FlyToROCDLConversionPass` 在 lowering 后，walk 找到带 `fly.alloc_id` 的 `LLVM::GEPOp`，
  forward-walk 这些 ptr 派生出的所有 SSA chain（含 GEP、cast、ptrtoint/inttoptr、整数算术）。
- 给所有 reachable 的 **LDS 读**（`llvm.load`）打上 `!alias.scope = !{!fly.shared_alloc}, !noalias = !{}`。
- **故意跳过** `ROCDL::RawPtrBufferLoadLdsOp`（不给 G→LDS 写打 scope）。

### 3.2 为什么"成功"了

这其实**没有遵循 LLVM IR 的形式语义**：

> 形式语义上，`alias.scope = {S}, noalias = {}` 意味着"对所有声明对 S noalias 的指针都不 alias"。
> 因为我们的 `noalias` 是空集，**这条元数据在 LLVM 中端的 ScopedNoAliasAA 看来等于啥都没说**。

那为什么有效？因为我们利用的是 AMDGPU 后端 `SIInsertWaitcnts` 比对 `MachineMemOperand::AAInfo` 时的一个**轻量启发式**：

> 两条 machine inst 比对 `AAMDNodes` 时，若一边带 `AliasScope`、另一边为空，
> `SIInsertWaitcnts` 把它视为"它们来自不同的 alias 域，可以不必等 vmcnt"。

这条规则**没有写在 LLVM 任何官方文档里**，是经验观察 + 源码阅读推断的。

### 3.3 实测有效但脆弱

- 性能恢复到 ~2810 TFLOPS，vmcnt 降到 148。
- 但**对 buffer.load.lds 也打 scope 时反而更糟**（同 scope 的多条 buffer.load.lds 被视为互相 may-alias，~225 多余 vmcnt）。

### 3.4 隐患

这是一个 **hack**，三条风险：

1. **`MachineScheduler` 也读 `AAMDNodes`**：如果它采用同样的"一边有 scope、一边没有 → 视 noalias"启发式，会**真的重排访存**。当前 kernel 因为有 barrier 和强 SSA 数据依赖**碰巧**没出错；未来某个不带 barrier 的 LDS-通信 kernel **可能 miscompile**。
2. **LLVM/ROCm 升级**改了后端 AAMDNodes 处理 → 性能 fix 失效，或转成 miscompile。
3. **跨后端不可移植**：NVPTX/SPIR-V 的启发式可能不同。

**评价**：作为 perf fix 短期可接受，但**不应作为长期方案**。

---

## 4. 真正干净的方案：`Fly_Swizzle` + `composed_layout`

### 4.1 已存在的范本：`preshuffle_gemm_v2.py`

```python
swz = fx.SwizzleType.get(3, 3, 3)
sA = fx.make_view(smem_ptr, fx.make_composed_layout(
    fx.static(swz),
    fx.make_ordered_layout((tile_m, tile_k, 2), (1, 0, 2)),
))
```

实测 LLVM IR（同样用单一 `@__dynamic_shared__0`）：

```llvm
@__dynamic_shared__0 = external dso_local addrspace(3) global [0 x i8], align 1024

%108 = getelementptr bfloat, ptr addrspace(3) @__dynamic_shared__0, i32 %107
store <8 x bfloat> %50, ptr addrspace(3) %108, align 16
```

数字：

- **0 个 `ptrtoint`、0 个 `inttoptr`**
- 152 个直接 LDS GEP
- 0 个 `!alias.scope` 元数据
- vmcnt = 41，lgkmcnt = 126
- **BasicAA 完美工作，不需要任何 hack**

### 4.2 为什么这条路径不破坏 BasicAA

`Fly_Swizzle` 的 lowering 路径（FlyDSL 内部）：

1. **IntTuple 层**（`lib/Dialect/Fly/Utils/IntTupleUtils.cpp::applySwizzle`）：
   把 SwizzleAttr 应用在一个 **i32/i64 SSA value** 上，生成纯整数算术：
   ```
   %a = arith.andi %offset_in, <mask>
   %b = arith.shrui %a, <shift>
   %c = arith.xori %offset_in, %b      ; XOR-swizzle 结果
   ```
2. **Composed-layout 解构层**（`lib/Dialect/Fly/Transforms/LayoutLowering.cpp::DecompositionOpLowering`）：
   ```cpp
   Value offset = layoutBuilder.finalize(decomposed.offset);   // i32 SSA
   Value iter = AddOffsetOp::create(rewriter, loc, makeViewOp.getIter(), offset);
   //         ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
   //         fly.add_offset 最终 lower 成 LLVM::GEPOp，**不是 inttoptr**！
   ```
3. **FlyToROCDL 层**（`AddOffsetOpLowering`）：
   `fly.add_offset` → `LLVM::GEPOp`，保留 underlying object 信息。

结果：BasicAA 看到的是 `getelementptr ... @__dynamic_shared__0, i32 %xored_offset` —— 它知道 underlying object 是 `@__dynamic_shared__0`，知道这是 LDS（不同 addrspace），知道两条 GEP 是同一个 base + 不同 i32 SSA。即使两个 i32 是运行时值，`addrspace(3)` vs `addrspace(1)` 的地址空间 NoAlias 就足够 `SIInsertWaitcnts` 用了。

### 4.3 当前 `fp8_gemm_4wave` 用错了 API

| 当前 work tree 写法 | 干净写法 |
|---|---|
| `fx.CoordSwizzleType.get(3, 1, [0], 4, [1])`（变换 `(row,col)→(row',col')`） | `fx.SwizzleType.get(mask, base, shift)`（变换 i32 offset → i32 offset'） |
| `fx.crd2idx((row, col), _lds_swz_layout).to_py_value()` 在 Python 端**降到常数表** | `make_view(ptr, composed_layout(SwizzleType, outer))`，让 lowering pass 自动处理 |
| `fx.ptrtoint(lds_array.ptr) + offset → fx.inttoptr` 手工拼指针 | LayoutLowering 自动用 `fx.add_offset(ptr, i32_offset)`，最终是 GEP |

`CoordSwizzleType` 本身没问题（它在 `(row, col)` 域定义 XOR-swizzle），但**当前 kernel 调用 `to_py_value()` 把它降到 Python 常数**就完全绕过了 layout lowering 的好处。即使只能用 `CoordSwizzleType`，也应该让它走 `composed_layout` 路径，让 `DecompositionOpLowering` 来生成 `fly.add_offset`。

### 4.4 改造的工作量

`fp8_gemm_4wave` 的 LDS 访问散布在：

- `_lds_dst_at`（G→LDS 写）—— 改成基于 `composed_layout` view 的 partition
- `_vec_load_lds_i32x4`（LDS→reg 读）—— 同上
- `_load_lds` / `_load_one_lds` / `_load_rt` / `_load_one_rt` —— 调整调用方
- `_compute_global_swizzle` / `_compute_lds_swizzle` —— 不再需要 Python 端的常数表，直接通过 layout 索引

参考 `kernels/preshuffle_gemm_v2.py` 第 135-147 行的做法。改造完成后：

1. 删除 `lib/Conversion/FlyToROCDL/FlyToROCDL.cpp` 里 `annotateSharedAllocAliasScopes` + `cleanupAllocIdAttrs` + `AddOffsetOpLowering` 中 `fly.alloc_id` 透传
2. 删除 `lib/Dialect/Fly/Transforms/LayoutLowering.cpp` 里 `FoldAddOffsetChain`
3. 还原 `include/flydsl/Dialect/Fly/Transforms/MemrefLowering.td`
4. 删除 `python/flydsl/expr/struct.py` 里 `_tag_alloc_id` + `Arena.allocate` 的标签调用

整个 work tree 改动归零。Kernel 用 FlyDSL 的标准 layout API 表达。

---

## 5. 实验记录（用于后续复现）

### 5.1 测量方法

```bash
cd /root/Projects/flydsl
rm -rf ~/.flydsl/cache ~/.flydsl/debug

# 性能
python tests/kernels/test_fp8_gemm_4wave.py \
    -M 8192 -N 8192 -K 8192 --num_warmups 10 --num_iters 50

# IR dump（只编译，不跑）
FLYDSL_DUMP_IR=1 COMPILE_ONLY=1 \
    python tests/kernels/test_fp8_gemm_4wave.py \
    -M 8192 -N 8192 -K 8192 --num_warmups 0 --num_iters 1

# 关键 metric
grep -c "s_waitcnt vmcnt"      ~/.flydsl/debug/kernel_gemm_0/21_final_isa.s
grep -c "alias.scope"          ~/.flydsl/debug/kernel_gemm_0/20_llvm_ir.ll
grep -c "ptrtoint\|inttoptr"   ~/.flydsl/debug/kernel_gemm_0/20_llvm_ir.ll
```

### 5.2 实验矩阵

| 实验 | LDS 全局 | LDS ptr 表达 | alias-scope pass | vmcnt | TFLOPS |
|---|---|---|---|---:|---:|
| A. HEAD baseline | 8 sym | `ptrtoint+add+inttoptr` | OFF | 154 | 2750 |
| B. Work tree 全开 | 1 sym | `ptrtoint+add+inttoptr` | ON | 148 | 2810 |
| C. Work tree 关 pass | 1 sym | `ptrtoint+add+inttoptr` | OFF | 1141 | 690 |
| D. Work tree + natural add_offset | 1 sym | `fx.add_offset` (GEP) | ON | 148 | 2630 |
| E. preshuffle_gemm_v2 范式 | 1 sym | `fx.add_offset` (GEP) | OFF（不需要） | 41 | ✓ |

观察：
- **A vs B**：单 sym 必须配 alias-scope hack 才能恢复性能。
- **B vs C**：关掉 alias-scope pass，性能从 2810 跌到 690 —— pass 起了作用。
- **B vs D**：从 `ptrtoint` 改成 `fx.add_offset` 但仍依赖 alias-scope，性能反而掉一点（2630 vs 2810）。说明 D 路径下 lowering 阶段生成的辅助指令更多，影响了主循环大小。
- **E**：完全干净的 GEP 路径，**不需要**任何 alias-scope hack。

### 5.3 关键 LLVM IR 形态对照

**HEAD（8 sym, BasicAA 凭 underlying object 区分）**：
```llvm
@A_lds_cur_0 = external addrspace(3) global [16384 x i8]
...
%137 = add i64 ptrtoint (ptr addrspace(3) @A_lds_cur_0 to i64), %136
%138 = inttoptr i64 %137 to ptr addrspace(3)
%241 = getelementptr i8, ptr addrspace(3) @A_lds_cur_0, i64 %240
```

**Work tree（1 sym, BasicAA 失明）**：
```llvm
@__dynamic_shared__0 = external addrspace(3) global [0 x i8]
...
%X = add i32 ptrtoint (@__dynamic_shared__0 to i32), %off
%Y = inttoptr i32 %X to ptr addrspace(3)
load <16 x i8>, ptr addrspace(3) %Y, !alias.scope !{!fly.shared_alloc}, !noalias !{}
```

**preshuffle_gemm_v2（1 sym, BasicAA 看得清）**：
```llvm
@__dynamic_shared__0 = external addrspace(3) global [0 x i8]
...
%108 = getelementptr bfloat, ptr addrspace(3) @__dynamic_shared__0, i32 %107
store <8 x bfloat> %50, ptr addrspace(3) %108, align 16
```

---

## 6. 当前 work tree 修改清单（如果决定回滚）

按"完全干净 = 把 fp8_gemm_4wave 改用 preshuffle_gemm_v2 范式"的目标，**全部回滚**：

```
include/flydsl/Dialect/Fly/Transforms/MemrefLowering.td   ← 还原 TD pattern
kernels/fp8_gemm_4wave.py                                 ← 重写为 SwizzleType+composed_layout
lib/Conversion/FlyToROCDL/FlyToROCDL.cpp                  ← 删除 alias-scope pass
lib/Dialect/Fly/Transforms/LayoutLowering.cpp             ← 删除 FoldAddOffsetChain
python/flydsl/expr/primitive.py                           ← 还原 recast_iter 修复（仅 7 行，可保留）
python/flydsl/expr/struct.py                              ← 删除 _tag_alloc_id / Arena 标签
```

`primitive.py` 那个 `recast_iter` 修复实际上**修了一个独立的、预存在的 bug**（`Array.__peek_from_ptr__` 在 HEAD 上就会崩），跟 alias-scope 无关，可以独立保留。

---

## 7. 结论

1. **`fp8_gemm_4wave` 性能问题的本质**：单一 dyn-shared LDS 全局 + `ptrtoint/inttoptr` 让 BasicAA 看不见 LDS 访问的 underlying GEP 结构 → AMDGPU 后端 `SIInsertWaitcnts` 保守插入大量 `s_waitcnt vmcnt`。

2. **HEAD 之所以快，不是因为 ptrtoint hack 神奇**，而是因为 8 个独立 LDS 全局让 BasicAA 凭 underlying-object 分析得出 NoAlias。HEAD kernel 里的"break alias chain"注释是误传。

3. **当前 work tree 的 alias-scope pass 是 hack**：
   - 形式语义上没传达任何信息（`noalias = {}` 是空集）；
   - 实际依赖 AMDGPU `SIInsertWaitcnts` 一个未文档化的 AAMDNodes 启发式；
   - `MachineScheduler` 可能在类似启发式下重排访存，构造性场景可 miscompile；
   - 当前 kernel 因 barrier + SSA 依赖保护，正确性"碰巧安全"。

4. **真正干净的解法**：用 `Fly_Swizzle + composed_layout + fx.add_offset` 表达 LDS swizzle（类似 `preshuffle_gemm_v2.py`），让 lowering 生成纯 GEP，BasicAA 自然 work，无需任何 alias 元数据。

5. **推荐路径**：
   - **短期**（如要尽快 merge）：保留 work tree 当前修复，但在 `annotateSharedAllocAliasScopes` 处加大段注释说明这是 workaround、依赖未文档化启发式、`MachineScheduler` 风险点。打 issue tracking 长期方案。
   - **长期**：把 `fp8_gemm_4wave` 改写为 `Fly_Swizzle + composed_layout` 范式，回滚所有 alias-scope 基建。

---

## 8. 后续追加：`fp8_gemm_4wave_v2` 的对齐 (2025-05-14)

`fp8_gemm_4wave_v2.py`（layout API 重构 + 强制使用 `SharedAllocator`）相对 v1 在**保留单一 `@__dynamic_shared__0` 全局**的硬约束下出现 ~3% 性能差距。诊断结果与对齐方案：

### 8.1 性能差距的两阶段定位

**阶段 A（vmcnt 暴涨，~70% 损失）**：与第 §2/§3 节相同，单 dyn-shared LDS 全局让 BasicAA 失明，`SIInsertWaitcnts` 保守插入大量 `s_waitcnt vmcnt`。**已由 alias-scope hack 修复**（148 ↔ 154 持平）。

**阶段 B（v_accvgpr_mov_b32 +1118，~3% 损失）**：alias-scope 修复 vmcnt 后，发现 v2 ISA 仍比 v1 多 ~1414 行，主要来自：
- `v_accvgpr_mov_b32`: 3172 vs 2054 (+1118)
- `s_nop`: 947 vs 497 (+450)
- `ds_read_b128 → AGPR`: 1494 vs 1990 (-496)
- `ds_read_b128 → VGPR`: 554 vs 58 (+496)

LDS load 的 ~25% 经历"先到 VGPR，再 `v_accvgpr_write_b32` 转 AGPR"的迂回路径。AMDGPU 后端的 `RewriteAGPRCopyMFMA` pass 应该把这类链折叠成 `ds_read_b128 a[...]` 直读 AGPR，但在 v2 IR 上失败了。

### 8.2 关键对照：B form vs D form

回顾 §5.2 实验：
| 实验 | LDS ptr 表达 | TFLOPS |
|---|---|---:|
| B (旧 v1 `kernels/fp8_gemm_4wave.py`) | `add i64 ptrtoint(@LDS), %off` + `inttoptr` | **2810** |
| D (v2 重构 `kernels/fp8_gemm_4wave_v2.py`) | `getelementptr i8, @__dynamic_shared__0, i32 %off` | **2630** |

**B form 的优势**：`ptrtoint + addi i64 + inttoptr` 在 LLVM IR 中切断了 GEP 的 def-use chain，`RegisterAllocator` 把 LDS load 的 dst 类视为更宽松的 AV_* (AGPR/VGPR-compatible) reg class，让 `RewriteAGPRCopyMFMA` 后续能成功折叠到 `ds_read a[...]`。

**D form 的劣势**：`getelementptr` 的输出 reg class hint 跟随 GEP 链；当 base 是单一 `@__dynamic_shared__0` + i32 offset 时，dst 被绑定到 VGPR class，AGPR 折叠失败。

### 8.3 对齐修复

在 `AddOffsetOpLowering` 中，将所有 LDS（addrspace 3）的 `fly.add_offset` 强制走 B form：

```cpp
// lib/Conversion/FlyToROCDL/FlyToROCDL.cpp
if (flyPtrTy.getAddressSpace() == sharedAS) {
  Value baseAsInt = LLVM::PtrToIntOp::create(rewriter, loc, i64Ty, base);
  Value offset64 = arith::ExtSIOp::create(rewriter, loc, i64Ty, offsetVal);
  Value sumInt = arith::AddIOp::create(rewriter, loc, baseAsInt, offset64);
  newPtr = LLVM::IntToPtrOp::create(rewriter, loc, ptrTy, sumInt);
}
```

并扩展 `annotateSharedAllocAliasScopes` 让它认识 `LLVM::IntToPtrOp` 作为 root（之前只认 `LLVM::GEPOp`），保证 alias-scope hack 仍然 anchor 在新形态 root 上。

### 8.4 修复后实测

| 指标 | v1 | v2 baseline | **v2 (修复后)** |
|---|---:|---:|---:|
| TFLOPS | ~2598 | ~2505 | **~2662** ✅ |
| ISA 总行数 | 14834 | 16248 | **13163** |
| `v_accvgpr_mov_b32` | 2054 | 3172 | **1084** |
| `ds_read → AGPR` | 1990 | 1494 | **1986** |
| `s_waitcnt vmcnt` | 154 | 148 | **148** |
| 嵌套 GEP | 0 | 84 | **0** |

v2 性能反超 v1 **2.4%**。所有 4 项 fp8_gemm_4wave_v2 sanity test 通过；1038 项 kernel 测试 0 失败。

### 8.5 修复影响范围

- **不影响**任何 non-LDS 指针 lowering（global/buffer-fat-ptr/register 仍走 GEP / BufferFatPtr）。
- **不影响** v1 kernel 性能（v1 LDS 也走新 B form，保持 2585+ TFLOPS）。
- **影响** `tests/mlir/Conversion/dyn_shared.mlir`：CHECK 模式从 GEP 改成 ptrtoint+inttoptr（已更新）。

### 8.6 长期方案与短期方案的关系

§7 提到的"长期方案：换用 `Fly_Swizzle + composed_layout`"是另一条独立路径，会让 BasicAA 直接 work、不需要 alias-scope hack；它和这次的 B form 修复**并不冲突**，只是各自解决了不同的瓶颈：
- B form 修复：让 RegAlloc 对 LDS load → MFMA chain 选 AGPR 直读（解决阶段 B）。
- Swizzle composed_layout：让 BasicAA 不再需要 alias-scope hack（解决阶段 A）。

如果未来切换到 Swizzle composed_layout 范式，B form 修复仍然受益（AGPR 直读跟 alias 元数据无关）。
