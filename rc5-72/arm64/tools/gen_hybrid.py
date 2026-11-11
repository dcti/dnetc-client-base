#!/usr/bin/env python3
"""Generate rc5-72/arm64/r72-neon-hyb.S: MIX NEON HYBRID, the scalar+NEON RC5-72 core,
16 keys per batch (4 NEON pipes x 4 lanes), software-pipelined so the integer
units run the early key schedule for batch g+1 while the vector units run the
late key schedule and encryption for batch g (the structure of Didier Levet's
2003 PPC KKS7450 core, rebalanced for AArch64).

Per batch of 16 keys:
  scalar phase (4 subgroups of 4 keys, 4 independent chains each):
    key-schedule steps k=3..SPLIT-1, every S value stored transposed into the
    prep table (subgroup sg's key i -> row[k%26] + 16*sg + 4*i), then A and
    L0..L2 stored to handoff rows 26..29.  Runs interleaved with...
  vector phase: steps k=SPLIT..76 (unrolled, direct row offsets, no walking
    pointers), encryption, merged low-word check; k=77 deferred to the rare
    path.  Reads the table the scalar phase of the PREVIOUS batch wrote.
Two tables ping-pong; the scalar stream always preps "the other" table with
L0hi+16.  On an L0hi byte carry (1 batch in 16) that prep is discarded and
(after the mid/lo recompute) redone by the standalone copy of the scalar
phase; at the last batch it is simply discarded.  The steady loop has no
special cases.

Usage: gen_hybrid.py [--split=N] [--skew=N] [--testmask] > ../r72-neon-hyb.S
--split=N: first vector step (default 25).
--skew=N: vector pipe p runs N*p ops behind pipe 0 (default 2; 0: plain order).
--testmask: test build only (ELF): both result compares are done under a
  global u32 r72_testmask, which the test program must define, so that a test
  can provoke partial and full matches.
"""
import heapq, re, sys

TESTMASK = "--testmask" in sys.argv
SKEW = 2                     # vector pipe p runs SKEW*p ops behind pipe 0 (0: plain order)
for a in sys.argv:
    if a.startswith("--skew="): SKEW = int(a.split("=")[1])
NP = 4                       # vector pipes
STRIDE = 16 * NP             # bytes per S row (4 pipes x 4 lanes)
SPLIT = 25                   # first vector step k; scalar does k=3..SPLIT-1
# 25 measured best with the skewed vector order.
for a in sys.argv:
    if a.startswith("--split="): SPLIT = int(a.split("=")[1])
# Below 29 the vector side broadcasts the pass-1 constants (mov32 + dup,
# k=SPLIT..26) and the precompute values (dup from the spills, k=27..28);
# no extra base register, so the two streams stay disjoint. 4 <= SPLIT.
assert 4 <= SPLIT <= 52

# ---- vector register map ---------------------------------------------------
# T, U: temporaries; A: the key-schedule A of each pipe; B[p][j]: L[j] of
# pipe p; then the constants 31, ~31, cypher.lo, cypher.hi. v28-v31 unused.
# The encryption's C and D reuse the L[0]/L[2] registers (A and L[1] must
# survive for step k=77 on the rare path).
T = [f"v{i}" for i in range(0, 4)]
U = [f"v{i}" for i in range(4, 8)]
A = [f"v{i}" for i in range(8, 12)]
B = [[f"v{12 + 3 * p + j}" for j in range(3)] for p in range(NP)]
C31, CM32, CLO, CHI = "v24", "v25", "v26", "v27"
C = [B[0][0], B[0][2], B[1][0], B[1][2]]
D = [B[2][0], B[2][2], B[3][0], B[3][2]]
assert len(set(C + D + [B[p][1] for p in range(NP)])) == 12

# ---- scalar register map --------------------------------------------------
# The two streams are interleaved line by line, so they share nothing:
# vector GPRs: x0, w9, w13-w15 (read), x16, x10 (table anchor, live all body).
# Scalar GPRs below, plus x12 (base of the table being prepped) and x30 (store
# pointer into it). x11 (current table base) is dead after the handoff loads:
# the scalar stream reuses it as w11, and swap_tables sets it again.
SA  = ["w2", "w3", "w4", "w5"]        # A chain per key
SL  = [["w6", "w7", "w8", "w17"],     # L[0] per key
       ["w19", "w20", "w21", "w22"],  # L[1] per key
       ["w23", "w24", "w25", "w26"]]  # L[2] per key
ST  = ["w27", "w28", "w1"]            # temps, rotated per key (x1 spilled)
SC_K = "w11"                          # shared subkey constant / spill loads
# The prep's L0hi and the pre_* values travel through the frame, not in
# registers.

out = []
def e(s=""): out.append(s)
def ins(s): out.append("        " + s)
def s4(r): return r + ".4s"
def b16(r): return r + ".16b"
def q(r): return "q" + r[1:]
def each(fmt, **lists):
    for p in range(NP):
        ins(fmt.format(**{k: v[p] for k, v in lists.items()}) + (f"\t@{p}" if SKEW else ""))

def cap(fn, *a):
    global out
    saved, out = out, []
    fn(*a)
    got, out = out, saved
    return got

def merge_prop(a, b):
    """append stream a with stream b's lines distributed evenly through it"""
    bi = 0
    for i, line in enumerate(a):
        out.append(line)
        want = (i + 1) * len(b) // len(a)
        while bi < want:
            out.append(b[bi]); bi += 1
    out.extend(b[bi:])

def load4(dst, base, off):
    t0, t1 = ("\t@L01", "\t@L23") if SKEW else ("", "")
    ins(f"ldp     {q(dst[0])}, {q(dst[1])}, [{base}, #{off}]" + t0)
    ins(f"ldp     {q(dst[2])}, {q(dst[3])}, [{base}, #{off + 32}]" + t1)

def store4(src, base, off):
    t0, t1 = ("\t@S01", "\t@S23") if SKEW else ("", "")
    ins(f"stp     {q(src[0])}, {q(src[1])}, [{base}, #{off}]" + t0)
    ins(f"stp     {q(src[2])}, {q(src[3])}, [{base}, #{off + 32}]" + t1)

def vrotl(x, s, t1, t2):
    each("and     {u}, {s}, " + b16(C31), u=[b16(r) for r in t1], s=[b16(r) for r in s])
    each("orr     {t}, {s}, " + b16(CM32), t=[b16(r) for r in t2], s=[b16(r) for r in s])
    each("ushl    {u}, {x}, {u}", u=[s4(r) for r in t1], x=[s4(r) for r in x])
    each("ushl    {t}, {x}, {t}", t=[s4(r) for r in t2], x=[s4(r) for r in x])
    each("orr     {x}, {u}, {t}", x=[b16(r) for r in x], u=[b16(r) for r in t1], t=[b16(r) for r in t2])

def rotl3(dst, src):
    each("shl     {d}, {s}, #3", d=[s4(r) for r in dst], s=[s4(r) for r in src])
    each("sri     {d}, {s}, #29", d=[s4(r) for r in dst], s=[s4(r) for r in src])

def mov32(reg, val):
    ins(f"mov     {reg}, #0x{val & 0xFFFF:04X}")
    ins(f"movk    {reg}, #0x{val >> 16:04X}, lsl #16")

# ---- RC5-72 constants ------------------------------------------------------
M = 0xFFFFFFFF
P, Q = 0xB7E15163, 0x9E3779B9
def rotl(x, s):
    s &= 31
    return ((x << s) | (x >> (32 - s))) & M if s else x
S0 = rotl(P, 3)
K1 = (P + Q + S0) & M
K2 = (P + 2 * Q) & M
assert (S0, K1, K2) == (0xBF0A8B1D, 0x15235639, 0xF45044D5)

# ---- frame layout ----------------------------------------------------------
# sp+0    d8-d15 (64)
# sp+64   x19-x28 (80)
# sp+144  x1 spill (8), pad
# sp+160  pre_B_0, pre_S1, pre_B_1, pre_S2, pre_AB_2, prep L0hi (6 words)
# sp+184  x30 while scalar_prep runs
# sp+192  table 0: S rows 0..25, then A, L0, L1, L2 handoff rows (30 x 64)
# sp+2112 table 1 (same layout)
SP_X19, SP_X1, SP_PRE = 64, 144, 160
PRE_B0, PRE_S1, PRE_B1, PRE_S2, PRE_AB2, SP_HI = (SP_PRE + 4 * i for i in range(6))
SP_LR = SP_PRE + 24                 # x30 while scalar_prep runs (x30 = store pointer)
TAB0, TAB1, TABSZ = 192, 2112, 30 * STRIDE
ANCHOR = 13 * STRIDE                 # S rows 0..25 within ldp/stp q range of x10
ROW_A, ROW_L = 26, 27                # handoff rows: A, then L0..L2
FRAME = 4032                         # < 4096: no stack probe needed
assert SP_LR + 8 <= TAB0 and TAB0 + TABSZ <= TAB1 and TAB1 + TABSZ <= FRAME < 4096 and FRAME % 64 == 0
def roff(n): return (n - 13) * STRIDE       # S row offset from the anchor

# ---- vector stream ---------------------------------------------------------
def ks_step_u(k):
    """unrolled key-schedule step k, all pipes, direct row offsets.
    k <= 25: S_in is the pass-1 constant (mov32 + dup); k = 26: S0 (mov32 +
    dup); k = 27, 28: pre_S1 / pre_S2, dup from the spills; else the table row."""
    row, bx, by = k % 26, (k - 1) % 3, k % 3
    Bx = [B[p][bx] for p in range(NP)]
    By = [B[p][by] for p in range(NP)]
    ins(f"// k={k}: A = (A + S[{row}] + B{bx}) <<< 3; S[{row}] = A; B{by} = (B{by} + B{bx} + A) <<< (B{bx} + A)")
    if k <= 26:
        mov32("w16", S0 if k == 26 else (P + k * Q) & M)
        each("dup     {t}, w16", t=[s4(r) for r in T])
    elif k in (27, 28):
        ins(f"ldr     w16, [sp, #{PRE_S1 if k == 27 else PRE_S2}]")
        each("dup     {t}, w16", t=[s4(r) for r in T])
    else:
        load4(T, "x10", roff(row))
    each("add     {t}, {t}, {a}", t=[s4(r) for r in T], a=[s4(r) for r in A])
    each("add     {t}, {t}, {b}", t=[s4(r) for r in T], b=[s4(r) for r in Bx])
    rotl3(A, T)
    store4(A, "x10", roff(row))
    each("add     {t}, {b}, {a}", t=[s4(r) for r in T], b=[s4(r) for r in Bx], a=[s4(r) for r in A])
    each("add     {y}, {y}, {t}", y=[s4(r) for r in By], t=[s4(r) for r in T])
    vrotl(By, T, U, T)

def half_round(X, Y, off):
    each("eor     {x}, {x}, {y}", x=[b16(r) for r in X], y=[b16(r) for r in Y])
    vrotl(X, Y, U, T)
    load4(T, "x10", off)
    each("add     {x}, {x}, {t}", x=[s4(r) for r in X], t=[s4(r) for r in T])

def compare(dst, vecs, ref, vtmp):
    if not TESTMASK:
        each("cmeq    {t}, {v}, " + s4(ref), t=[s4(r) for r in dst], v=[s4(r) for r in vecs])
        return
    ins("// R72_TESTMASK (test build only): compare masked copies")
    ins("adrp    x16, r72_testmask")
    ins("add     x16, x16, :lo12:r72_testmask")
    ins(f"ld1r    {{{s4(vtmp)}}}, [x16]")
    each("and     {t}, {v}, " + b16(vtmp), t=[b16(r) for r in dst], v=[b16(r) for r in vecs])
    each("cmeq    {t}, {t}, " + s4(ref), t=[s4(r) for r in dst])

def movemask(dst_w, vecs, wreg, vtmp, tmpw):
    for p in range(NP):
        ins(f"and     {b16(vtmp)}, {b16(vecs[p])}, {b16(wreg)}")
        ins(f"addv    s{vtmp[1:]}, {s4(vtmp)}")
        if p == 0:
            ins(f"fmov    {dst_w}, s{vtmp[1:]}")
        else:
            ins(f"fmov    {tmpw}, s{vtmp[1:]}")
            ins(f"orr     {dst_w}, {dst_w}, {tmpw}, lsl #{4 * p}")

def popcnt(dst_w, src_w, vtmp):
    ins(f"fmov    s{vtmp[1:]}, {src_w}")
    ins(f"cnt     {vtmp}.8b, {vtmp}.8b")
    ins(f"addv    b{vtmp[1:]}, {vtmp}.8b")
    ins(f"fmov    {dst_w}, s{vtmp[1:]}")

def vector_body():
    """steps SPLIT..76, encryption, up to (not including) the check."""
    for k in range(SPLIT, 77):
        ks_step_u(k)
    ins("// encryption: C = plain.lo + S[0], D = plain.hi + S[1]")
    ins("ldr     w16, [x0, #4]")
    ins(f"dup     {s4(U[0])}, w16")
    ins("ldr     w16, [x0, #0]")
    ins(f"dup     {s4(U[1])}, w16")
    for row, dst, u in ((0, C, U[0]), (1, D, U[1])):
        load4(T, "x10", roff(row))
        each("add     {c}, {t}, " + s4(u), c=[s4(r) for r in dst], t=[s4(r) for r in T])
    for r in range(1, 12):
        ins(f"// round {r}: C = ((C ^ D) <<< D) + S[{2*r}]; D = ((D ^ C) <<< C) + S[{2*r+1}]")
        half_round(C, D, roff(2 * r))
        half_round(D, C, roff(2 * r + 1))
    ins("// final half-round; D only on the rare path")
    half_round(C, D, roff(24))

# ---- pipe-skewed order of the vector stream (--skew) -----------------------
# Every line of the vector stream is tagged with its pipe (each/load4/store4);
# pipe p's j-th op gets priority (j + SKEW*p)*4 + p, so adjacent instructions
# come from different pipes and different ops (the shifts are spread). The
# lines are then emitted in that priority order subject to every register and
# memory dependence (a topological sort), so any result computes the same
# values as the plain order. Step comments are dropped.
def _regs(ops):
    r = set()
    for o in ops:
        for m in re.finditer(r"\b([vqsdb])(\d+)\b", o): r.add("v" + m.group(2))
        for m in re.finditer(r"\b([wx])(\d+)\b", o): r.add("x" + m.group(2))
        if re.search(r"\bsp\b", o): r.add("sp")
    return r

def _deps(line):
    """(reads, writes, mem) for one vector-stream line; mem = (kind, base, lo, hi)"""
    op, rest = line.split(None, 1)
    ops = [x.strip() for x in re.split(r",(?![^\[]*\])", rest)]
    mem = None
    m = re.search(r"\[(\w+), #(-?\d+)\]", rest)
    if m:
        regs = [o for o in ops if not o.startswith("[")]
        size = {"q": 16, "w": 4, "x": 8}[regs[0][0]] * len(regs)
        mem = ("ld" if op.startswith("ld") else "st", m.group(1), int(m.group(2)), int(m.group(2)) + size)
    if op in ("ldr", "ldp"):
        return _regs([m.group(1)]), _regs(ops[:-1]), mem
    if op in ("str", "stp"):
        return _regs(ops), set(), mem
    if op in ("add", "orr", "and", "eor", "ushl", "shl", "dup", "mov", "cmeq"):
        return _regs(ops[1:]), _regs(ops[:1]), None
    if op in ("sri", "movk"):
        return _regs(ops), _regs(ops[:1]), None
    raise SystemExit(f"--skew: unhandled vector-stream line: {line}")

def skew_order(lines):
    body = []                             # (priority, index, text, reads, writes, mem)
    cnt = [0] * NP
    for l in lines:
        t = l.strip()
        if not t or t.startswith("//"): continue
        text, _, tag = l.partition("\t@")
        if tag[:1] in ("L", "S"):         # paired q load/store of pipes a, b
            a, b = int(tag[1]), int(tag[2])
            p = a if tag[0] == "L" else b # load: earliest reader; store: latest writer
            pri = (cnt[p] + SKEW * p) * NP + p
            cnt[a] += 1; cnt[b] += 1
        elif tag:
            p = int(tag); pri = (cnt[p] + SKEW * p) * NP + p; cnt[p] += 1
        else:
            pri = cnt[0] * NP           # shared (w16 constants): with pipe 0
        rd, wr, mem = _deps(text.strip())
        body.append((pri, len(body), text, rd, wr, mem))
    n = len(body)
    succ = [[] for _ in range(n)]; npred = [0] * n
    def edge(a, b):
        succ[a].append(b); npred[b] += 1
    last_w, readers, mems = {}, {}, []
    for i, (_, _, _, rd, wr, mem) in enumerate(body):
        for r in rd:
            if r in last_w: edge(last_w[r], i)                   # RAW
        for r in wr:
            if r in last_w: edge(last_w[r], i)                   # WAW
            for j in readers.get(r, []): edge(j, i)              # WAR
        for r in rd: readers.setdefault(r, []).append(i)
        for r in wr: last_w[r] = i; readers[r] = []
        if mem:
            for j, (k2, b2, lo2, hi2) in mems:
                if "st" in (mem[0], k2) and (b2 != mem[1] or (lo2 < mem[3] and mem[2] < hi2)):
                    edge(j, i)                                   # memory (other bases: conservative)
            mems.append((i, mem))
    ready = [(body[i][0], i) for i in range(n) if npred[i] == 0]
    heapq.heapify(ready)
    res = []
    while ready:
        _, i = heapq.heappop(ready)
        res.append(body[i][2])
        for j in succ[i]:
            npred[j] -= 1
            if npred[j] == 0: heapq.heappush(ready, (body[j][0], j))
    assert len(res) == n, "dependence cycle"
    return res

# ---- scalar stream ---------------------------------------------------------
# The scalar stream stores S/A/L words in transposed pairs (stp w, w: keys
# 2j and 2j+1 of a subgroup are adjacent words of a row) through x30, which
# points into the target table and is re-based every 8 rows (stp reaches
# -256..+252). x30 is free in the body; scalar_prep saves it around itself.
sc_base = [None]
def sc_store_pair(ra, rb, row, sg, j):
    off = None if sc_base[0] is None else (row - sc_base[0]) * STRIDE + 8 * j
    if off is None or not -256 <= off <= 252:
        sc_base[0] = row + 4
        ins(f"add     x30, x12, #{sc_base[0] * STRIDE + 16 * sg}")
        off = (row - sc_base[0]) * STRIDE + 8 * j
    ins(f"stp     {ra}, {rb}, [x30, #{off}]")

def scalar_stream(hi_lead):
    """the prep of one 16-key batch into [x12]: hi_lead emits the L0hi setup
    into the SP_HI frame slot. Clobbers only the scalar-pool registers."""
    hi_lead()
    for sg in range(NP):
        sc_base[0] = None
        ins(f"// ---- scalar subgroup {sg}: keys {4*sg}..{4*sg+3} of the next batch")
        ins(f"ldr     {SC_K}, [sp, #{PRE_S2}]")
        for i in range(4): ins(f"mov     {SA[i]}, {SC_K}")
        ins(f"ldr     {SC_K}, [sp, #{PRE_B0}]")
        for i in range(4): ins(f"mov     {SL[0][i]}, {SC_K}")
        ins(f"ldr     {SC_K}, [sp, #{PRE_B1}]")
        for i in range(4): ins(f"mov     {SL[1][i]}, {SC_K}")
        ins(f"// L2_i = (L0hi + {4*sg}+i + pre_AB_2) <<< pre_AB_2")
        ins(f"ldr     {SC_K}, [sp, #{SP_HI}]")
        for i in range(4):
            ins(f"add     {SL[2][i]}, {SC_K}, #{4 * sg + i}")
        ins(f"ldr     {SC_K}, [sp, #{PRE_AB2}]")
        ins(f"neg     {ST[0]}, {SC_K}")
        for i in range(4):
            ins(f"add     {SL[2][i]}, {SL[2][i]}, {SC_K}")
            ins(f"ror     {SL[2][i]}, {SL[2][i]}, {ST[0]}")
        for k in range(3, SPLIT):
            row, bx, by = k % 26, (k - 1) % 3, k % 3
            shared = None
            if k <= 25: shared = (P + k * Q) & M
            elif k == 26: shared = S0
            if shared is not None:
                ins(f"ldr     {SC_K}, L(sck{k})")
            elif k == 27:
                ins(f"ldr     {SC_K}, [sp, #{PRE_S1}]")
            elif k == 28:
                ins(f"ldr     {SC_K}, [sp, #{PRE_S2}]")
            for i in range(4):
                t = ST[i % 3]
                ins(f"add     {t}, {SA[i]}, {SL[bx][i]}")
                if k >= 29:
                    t2 = ST[(i + 1) % 3]
                    ins(f"ldr     {t2}, [x12, #{row * STRIDE + 16 * sg + 4 * i}]")
                    ins(f"add     {t}, {t}, {t2}")
                else:
                    ins(f"add     {t}, {t}, {SC_K}")
                ins(f"ror     {SA[i]}, {t}, #29")
                if i % 2: sc_store_pair(SA[i - 1], SA[i], row, sg, i // 2)
                ins(f"add     {t}, {SA[i]}, {SL[bx][i]}")
                ins(f"add     {SL[by][i]}, {SL[by][i]}, {t}")
                ins(f"neg     {t}, {t}")
                ins(f"ror     {SL[by][i]}, {SL[by][i]}, {t}")
        ins(f"// handoff: A and L0..L2, transposed")
        for h in range(2): sc_store_pair(SA[2 * h], SA[2 * h + 1], ROW_A, sg, h)
        for j in range(3):
            for h in range(2): sc_store_pair(SL[j][2 * h], SL[j][2 * h + 1], ROW_L + j, sg, h)

def hi_lead_next():
    ins(f"add     {ST[0]}, w13, #16            // prep is for the NEXT batch")
    ins(f"str     {ST[0]}, [sp, #{SP_HI}]")

# ==========================================================================
SKEWNOTE = (f"\n// The vector stream is skewed (pipe p runs {SKEW}*p ops behind pipe 0) so that"
            "\n// neighbouring instructions are different operations from different pipes;"
            "\n// this spreads the shifts on cores with a single vector shift unit." if SKEW else "")
e(f"""// RC5-72 scalar + NEON core for AArch64, 16 keys per batch (MIX NEON HYBRID)
// Created by Mix. Based on the structure of Didier Levet's PPC KKS7450
// core, rebalanced for AArch64.
//
// While 4 NEON pipes x 4 lanes run key-schedule steps k={SPLIT}..76 and the
// encryption for one batch, the scalar units run steps k=3..{SPLIT-1} of the next
// batch into a second S table; the two instruction streams are interleaved.
// Step k=77 and the second cipher word are only computed after a low-word
// match.{SKEWNOTE}
// x30 is used as a scratch pointer inside the loop. Needs MINIMUM_ITERATIONS
// to be a multiple of 16.
//
// Generated by rc5-72/arm64/tools/gen_hybrid.py; edit that, not this file."""
  + ("\n// TEST BUILD: R72_TESTMASK enabled. Do not ship." if TESTMASK else ""))
e("""
#if defined(__APPLE__)
#define FUNC(name) _##name
#define L(name) L##name
#else
#define FUNC(name) name
#define L(name) .L##name
#endif

        .text
        .p2align 4
        .globl  FUNC(rc5_72_unit_func_neon_hyb)
#if defined(__ELF__)
        .type   rc5_72_unit_func_neon_hyb, %function
#endif
FUNC(rc5_72_unit_func_neon_hyb):
// x0 = RC5_72UnitWork *, x1 = u32 *iterations, x2 = memblk (unused)
// Vector stream: temps v0-v7, A = v8-v11, L0..L2 of pipe p = v(12+3p)..
//   v(14+3p), v24 = 31, v25 = ~31, v26/v27 = cypher lo/hi;
//   x0, w9 (batches), w13/w14/w15 (L0), x16 scratch, x10 = current table
//   anchor (base + 832). x11 = current base (handoff loads only).
// Scalar stream: A = w2-w5, L0 = w6-w8/w17, L1 = w19-w22, L2 = w23-w26,
//   temps w27/w28/w1 (x1 is spilled), shared constant w11, x12 = base of
//   the table being prepped, x30 = store pointer into it.
//   The prep L0hi and the pre_* values live in the frame.
""")
ins("stp     x29, x30, [sp, #-16]!")
ins("mov     x29, sp")
ins(f"sub     x16, x29, #{FRAME}")
ins("and     sp, x16, #-64")
ins("stp     d8, d9, [sp, #0]")
ins("stp     d10, d11, [sp, #16]")
ins("stp     d12, d13, [sp, #32]")
ins("stp     d14, d15, [sp, #48]")
ins(f"stp     x19, x20, [sp, #{SP_X19}]")
ins(f"stp     x21, x22, [sp, #{SP_X19 + 16}]")
ins(f"stp     x23, x24, [sp, #{SP_X19 + 32}]")
ins(f"stp     x25, x26, [sp, #{SP_X19 + 48}]")
ins(f"stp     x27, x28, [sp, #{SP_X19 + 64}]")
ins(f"str     x1, [sp, #{SP_X1}]")
e()
ins("ldr     w9, [x1]")
ins("lsr     w9, w9, #4                 // 16 keys per batch")
ins("ldr     w13, [x0, #16]             // L0.hi")
ins("ldr     w14, [x0, #20]             // L0.mid")
ins("ldr     w15, [x0, #24]             // L0.lo")
e()
ins(f"movi    {s4(C31)}, #31")
ins(f"mvni    {s4(CM32)}, #31              // 0xFFFFFFE0")
ins("add     x16, x0, #12")
ins(f"ld1r    {{{s4(CLO)}}}, [x16]          // cypher.lo")
ins("add     x16, x0, #8")
ins(f"ld1r    {{{s4(CHI)}}}, [x16]          // cypher.hi")
e()
e("L(outer_loop):")
ins("// k=0: pre_B_0 = (L0lo + S0) <<< S0; k=1: pre_S1 = (P+Q+S0+pre_B_0) <<< 3")
mov32("w16", S0)
ins("add     w2, w15, w16")
ins("ror     w2, w2, #3")
ins(f"str     w2, [sp, #{PRE_B0}]")
mov32("w16", K1)
ins("add     w3, w2, w16")
ins("ror     w3, w3, #29")
ins(f"str     w3, [sp, #{PRE_S1}]")
e()
e("L(mid_loop):")
ins(f"ldr     w2, [sp, #{PRE_B0}]")
ins(f"ldr     w3, [sp, #{PRE_S1}]")
ins("add     w16, w2, w3")
ins("// k=1: pre_B_1 = (L0mid + pre_B_0 + pre_S1) <<< (pre_B_0 + pre_S1)")
ins("add     w17, w14, w16")
ins("neg     w2, w16")
ins("ror     w17, w17, w2")
ins(f"str     w17, [sp, #{PRE_B1}]")
ins("// k=2: pre_S2 = (P+2Q + pre_S1 + pre_B_1) <<< 3; pre_AB_2 = pre_S2 + pre_B_1")
mov32("w2", K2)
ins("add     w2, w2, w3")
ins("add     w2, w2, w17")
ins("ror     w2, w2, #29")
ins(f"str     w2, [sp, #{PRE_S2}]")
ins("add     w16, w17, w2")
ins(f"str     w16, [sp, #{PRE_AB2}]")
e()
ins("// prep the first batch (keys w13..w13+15) into table 0, then enter")
ins(f"add     x12, sp, #{TAB0}")
ins(f"str     w13, [sp, #{SP_HI}]")
ins("bl      L(scalar_prep)")
ins(f"add     x11, sp, #{TAB0}")
ins(f"add     x10, x11, #{ANCHOR}")
ins(f"add     x12, sp, #{TAB1}")
e()
ins(".p2align 4")
e("L(batch_loop):")
ins("// handoff: A and L0..L2 of this batch, transposed (prepped last batch)")
for p in range(NP):
    ins(f"ldr     {q(A[p])}, [x11, #{ROW_A * STRIDE + 16 * p}]")
for j in range(3):
    for p in range(NP):
        ins(f"ldr     {q(B[p][j])}, [x11, #{(ROW_L + j) * STRIDE + 16 * p}]")
e()
ins(f"// interleaved: vector k={SPLIT}..76+encryption (this batch, table [x10-{ANCHOR}])")
ins(f"//            x scalar prep k=3..{SPLIT-1} (next batch, table [x12])")
merge_prop(skew_order(cap(vector_body)) if SKEW else cap(vector_body),
           cap(scalar_stream, hi_lead_next))
e()
e("L(check_key):")
compare(T, C, CLO, U[0])
ins(f"orr     {b16(U[0])}, {b16(T[0])}, {b16(T[1])}")
ins(f"orr     {b16(U[1])}, {b16(T[2])}, {b16(T[3])}")
ins(f"orr     {b16(U[0])}, {b16(U[0])}, {b16(U[1])}")
ins(f"umaxv   s{U[0][1:]}, {s4(U[0])}")
ins(f"fmov    w16, s{U[0][1:]}")
ins("cbnz    w16, L(check_hi)")
e()
e("L(next_key):")
ins("subs    w9, w9, #1")
ins("b.eq    L(finished_nothing)")
ins("add     w13, w13, #16")
ins("tbnz    w13, #8, L(carry_hi)")
ins("str     w13, [x0, #16]")
e("L(swap_tables):")
ins("mov     x16, x12                   // other (just prepped) -> current")
ins(f"sub     x12, x10, #{ANCHOR}            // current -> other")
ins("mov     x11, x16")
ins(f"add     x10, x16, #{ANCHOR}")
ins("b       L(batch_loop)")
e()
e("L(carry_hi):")
ins("// the prep that ran alongside this batch assumed no carry out of L0hi:")
ins("// discard it, step mid/lo, recompute the pre_* values, re-prep the other")
ins("// table, and continue")
ins("and     w13, w13, #0xFF")
ins("str     w13, [x0, #16]")
ins("rev     w14, w14")
ins("adds    w14, w14, #1")
ins("rev     w14, w14")
ins("b.cc    L(redo_mid)")
ins("rev     w15, w15")
ins("add     w15, w15, #1")
ins("rev     w15, w15")
ins("// recompute pre_B_0/pre_S1 (lo changed), then fall through")
mov32("w16", S0)
ins("add     w2, w15, w16")
ins("ror     w2, w2, #3")
ins(f"str     w2, [sp, #{PRE_B0}]")
mov32("w16", K1)
ins("add     w3, w2, w16")
ins("ror     w3, w3, #29")
ins(f"str     w3, [sp, #{PRE_S1}]")
e("L(redo_mid):")
ins(f"ldr     w2, [sp, #{PRE_B0}]")
ins(f"ldr     w3, [sp, #{PRE_S1}]")
ins("add     w16, w2, w3")
ins("add     w17, w14, w16")
ins("neg     w2, w16")
ins("ror     w17, w17, w2")
ins(f"str     w17, [sp, #{PRE_B1}]")
mov32("w2", K2)
ins("add     w2, w2, w3")
ins("add     w2, w2, w17")
ins("ror     w2, w2, #29")
ins(f"str     w2, [sp, #{PRE_S2}]")
ins("add     w16, w17, w2")
ins(f"str     w16, [sp, #{PRE_AB2}]")
ins(f"str     w13, [sp, #{SP_HI}]")
ins("bl      L(scalar_prep)              // into [x12], the stale table")
ins("b       L(swap_tables)")
e()
e("L(check_hi):")
ins("// Rare path (about 1 batch in 2^28); scalar-pool registers (w2-w4) are")
ins("// dead here, and x10/x12 (the table pointers) must stay intact.")
ins("adr     x16, L(weights)")
ins(f"ldr     {q(U[0])}, [x16]")
movemask("w2", T, U[0], U[1], "w3")
ins("// w2 = 16-bit low-word match mask, bit n = key n")
e()
ins(f"// k=77: S[25] = (S[25] + A + B_1) <<< 3, then D = ((D ^ C) <<< C) + S[25]")
load4(U, "x10", roff(25))
each("add     {a}, {a}, {u}", a=[s4(r) for r in A], u=[s4(r) for r in U])
each("add     {a}, {a}, {b}", a=[s4(r) for r in A], b=[s4(B[p][1]) for p in range(NP)])
rotl3(T, A)
each("eor     {d}, {d}, {c}", d=[b16(r) for r in D], c=[b16(r) for r in C])
vrotl(D, C, U, A)
each("add     {d}, {d}, {t}", d=[s4(r) for r in D], t=[s4(r) for r in T])
compare(T, D, CHI, A[0])
ins("adr     x16, L(weights)")
ins(f"ldr     {q(U[0])}, [x16]")
movemask("w3", T, U[0], U[1], "w16")
ins("ands    w16, w3, w2                // keys matching both words")
ins("b.ne    L(save_found_key)")
e()
ins("// partial match(es) only: count them, record the last one")
popcnt("w4", "w2", U[0])
ins("ldr     w16, [x0, #28]")
ins("add     w16, w16, w4")
ins("str     w16, [x0, #28]")
ins("clz     w16, w2")
ins("mov     w4, #31")
ins("sub     w16, w4, w16")
ins("add     w16, w16, w13")
ins("str     w16, [x0, #32]")
ins("str     w14, [x0, #36]")
ins("str     w15, [x0, #40]")
ins("b       L(next_key)")
e()
e("L(save_found_key):")
ins("rbit    w16, w16")
ins("clz     w3, w16                    // w3 = first full match")
ins("mov     w4, #2")
ins("lsl     w4, w4, w3")
ins("sub     w4, w4, #1")
ins("and     w2, w2, w4")
popcnt("w4", "w2", U[0])
ins("ldr     w16, [x0, #28]")
ins("add     w16, w16, w4")
ins("str     w16, [x0, #28]")
ins("add     w16, w3, w13")
ins("str     w16, [x0, #32]")
ins("str     w14, [x0, #36]")
ins("str     w15, [x0, #40]")
ins("// *iterations = keys before the winning one")
ins(f"ldr     x8, [sp, #{SP_X1}]")
ins("ldr     w16, [x8]")
ins("lsr     w16, w16, #4")
ins("sub     w16, w16, w9")
ins("lsl     w16, w16, #4")
ins("add     w16, w16, w3")
ins("str     w16, [x8]")
ins("str     w14, [x0, #20]")
ins("str     w15, [x0, #24]")
ins("mov     w0, #2                     // RESULT_FOUND")
ins("b       L(finished)")
e()
e("L(finished_nothing):")
ins("add     w13, w13, #16")
ins("ubfx    w16, w13, #8, #1")
ins("and     w13, w13, #0xFF")
ins("rev     w14, w14")
ins("rev     w15, w15")
ins("adds    w14, w14, w16")
ins("adc     w15, w15, wzr")
ins("rev     w14, w14")
ins("rev     w15, w15")
ins("str     w13, [x0, #16]")
ins("str     w14, [x0, #20]")
ins("str     w15, [x0, #24]")
ins("mov     w0, #1                     // RESULT_NOTHING")
e()
e("L(finished):")
ins("ldp     d8, d9, [sp, #0]")
ins("ldp     d10, d11, [sp, #16]")
ins("ldp     d12, d13, [sp, #32]")
ins("ldp     d14, d15, [sp, #48]")
ins(f"ldp     x19, x20, [sp, #{SP_X19}]")
ins(f"ldp     x21, x22, [sp, #{SP_X19 + 16}]")
ins(f"ldp     x23, x24, [sp, #{SP_X19 + 32}]")
ins(f"ldp     x25, x26, [sp, #{SP_X19 + 48}]")
ins(f"ldp     x27, x28, [sp, #{SP_X19 + 64}]")
ins("mov     sp, x29")
ins("ldp     x29, x30, [sp], #16")
ins("ret")
e()
ins("// standalone scalar prep: the pipeline prologue, and the redo after an")
ins("// L0hi byte carry. In: x12 = target table base, [sp, SP_HI] = prep L0hi.")
e("L(scalar_prep):")
ins(f"str     x30, [sp, #{SP_LR}]")
scalar_stream(lambda: None)
ins(f"ldr     x30, [sp, #{SP_LR}]")
ins("ret")
e("#if defined(__ELF__)")
ins(".size   rc5_72_unit_func_neon_hyb, . - rc5_72_unit_func_neon_hyb")
e("#endif")
e()
ins(".p2align 4")
e("L(weights):")
ins(".word   1, 2, 4, 8")
e("// the scalar stream's pass-1 subkey constants, read PC-relative (ldr literal)")
for k in range(3, min(SPLIT, 27)):
    e(f"L(sck{k}):")
    ins(f".word   0x{((P + k * Q) & M if k <= 25 else S0):08X}")

e()
e("#if defined(__ELF__)")
ins('.section .note.GNU-stack,"",%progbits')
e("#endif")

sys.stdout.write("\n".join(re.sub(r"\t@\w+$", "", l) for l in out) + "\n")
