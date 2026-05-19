# BurnNotice: Design Analysis — Architecture, Science, and Adversarial Landscape

> **Status**: Superseded by `docs/analysis2.md`
> **This document**: Architecture evaluation, scientific critique, and adversarial research companion to the codebase audit

---

## Overview

`docs/analysis2.md` is the canonical source for:
- Complete codebase audit (all 11 flaws, pre/post-fix status)
- Full literature-grounded critique with citations
- Merged findings from both analysis passes

This document focuses on architectural assessment, design decision evaluation, and strategic guidance.

---

## 1. Architecture Assessment

### 1.1 What the Design Gets Right

The v0.3 design documents represent a substantial improvement over earlier drafts:

**Multi-probe redundancy**: Monitoring deception, conflict, policy OOD, truth confidence, and hallucination simultaneously creates defense-in-depth. If one probe is evaded, others may catch the threat.

**Multi-direction subspace extraction**: Moving beyond DIM to Top-K SVD and INLP (design.md §13.4) addresses the Wollschläger/Piras critique that refusal and deception are multi-dimensional phenomena.

**Adaptive calibration**: Dynamic threshold optimization via held-out ROC analysis (design.md §13.1) directly addresses the false positive compounding problem.

**Asynchronous scoring pipeline**: The lock-free ring buffer design (architecture.md §P0-1) prevents the synchronous blocking loop from stalling inference.

**Capability preservation benchmarks**: The $\mathcal{R}_{cap} \ge 0.98$ constraint on GSM8k is a meaningful safeguard against steering-induced capability degradation.

**Temporal scoring**: EMA-based sequence aggregation (design.md §13.5) addresses the limitation that deception is a sustained sequence, not a single-token event.

### 1.2 Critical Infrastructure Gaps

The architecture is built on four foundations, two of which are stubs:

| Foundation | Status | Impact if Missing |
|-----------|--------|-------------------|
| ONNX Inference Engine | **Stub** — Load/Forward/GetIntermediate all return nil | No activation capture possible |
| Longbow Vector Engine | **Stub** — all methods TODO | No vector storage, HNSW, SIMD, Arrow Flight |
| SQLite Ledger | **Working** — modernc.org/sqlite | Activity logging operational |
| Go Runtime | **Working** — goroutines, memory pool | Execution infrastructure operational |

The two working components (ledger, Go runtime) are supporting infrastructure. The two stubs (ONNX engine, Longbow) are the core of the system. This is the most critical architectural risk: the entire monitoring and intervention pipeline depends on infrastructure that has not been built.

### 1.3 The Longbow Dependency Risk — Updated Assessment

Longbow (github.com/23skdu/longbow) remains:
- Pre-v1.0 with rolling releases
- 13 stars, 1 fork, solo/small-team development
- No confirmed production deployments
- Not imported in `go.mod`; zero code written against it
- Competing against Qdrant (dominant self-hosted), Milvus (mature enterprise), Pinecone (managed simplicity)

The design documents now describe 6 Longbow-dependent features that would all need reimplementation if Longbow is unavailable:

1. Zero-copy Arrow Flight transport
2. SlabArena off-heap memory (bypasses Go GC)
3. SIMD kernels (AVX2/AVX-512/NEON)
4. HNSW nearest-neighbor indexing
5. COW publication (lock-free scoring)
6. SWIM gossip (distributed mesh)

A reasonable fallback strategy: implement storage on top of the existing SQLite ledger for scalar metadata, and use gonum for distance calculations (slower, but proven and dependency-free).

---

## 2. Scientific Foundations — What the Research Says

### 2.1 Core Claims vs. Research Reality

| Design Claim | Research Reality | Confidence |
|------------|-----------------|------------|
| Refusal mediated by a single direction | Multi-dimensional concept cones (Wollschläger) | **Rejected** |
| Deception detectable at 93%+ AUROC | 0.96–0.999 AUROC in narrow lab settings; topic-confused in practice (Apollo) | **Partial** |
| Activation steering restores alignment | Steering creates universal jailbreak vectors (Rogue Scalpel) | **Contradicted** |
| Proxy analyzer covers every backend | No cross-model validation exists | **Speculative** |
| Single-direction DIM is the primary extraction method | Top-K SVD outperforms DIM (Piras et al.) | **Incomplete** |
| 5-10% monitoring overhead | No empirical data for GPU inference pipeline | **Unvalidated** |

### 2.2 The Most Dangerous Research Finding

**Bailey et al. (2026), "Obfuscated Activations Bypass LLM Latent-Space Defenses" (ICLR 2026)** is not mentioned in any BurnNotice design document. It demonstrates that:

- Linear probes over activation space can be reduced from **100% recall to 0%** while maintaining **90% jailbreaking success**
- The attack crafts adversarial perturbations in activation space that evade the probe while preserving harmful output
- The "obfuscation tax" (reduced capability on legitimate tasks) is the only constraint

This directly attacks every probe-based component in BurnNotice. The design's adversarial robustness section (design.md §10.1) does not mention activation-space obfuscation attacks. The adversarial hardening section (design.md §13.2) proposes "steering gradient masking" and "decoy distributions" but provides no implementation path.

### 2.3 The Steering Attack Surface

BurnNotice's primary intervention mechanism is activation steering. The Rogue Scalpel (Korznikov et al., ICLR 2026) demonstrates that:

- Random steering vectors increase harmful compliance from 0% to 1–13%
- Combining 20 random vectors creates a **universal attack** that generalizes to unseen harmful requests
- SAE features used for steering are comparably harmful

This means the steering engine is simultaneously the system's most powerful safety tool and its largest attack surface. An attacker who learns how BurnNotice steers can invert the vectors to amplify deception.

---

## 3. Remediation Roadmap

### Phase 1: Build the Core (Weeks 1–4)

**Goal**: Make the system end-to-end functional with simulated activations, then real ones.

1. Implement ONNX inference engine (`internal/engine/onnx.go`)
2. Fix whitening rank deficiency: add PCA-subspace projection before covariance computation
3. Validate proxy analyzer approach with controlled experiments (or remove it)
4. Add capability preservation benchmark to steering tests

### Phase 2: Adversarial Hardening (Weeks 5–8)

**Goal**: Make the system robust to the known attack classes.

5. Implement and test against Bailey et al.'s obfuscation attack
6. Implement steering vector extraction detection (or formal analysis proving it's impossible)
7. Add KV-cache steering accumulation tracking; test multi-turn coherence
8. Implement multi-direction subspace extraction (Top-K SVD + INLP) as default

### Phase 3: Production Hardening (Weeks 9–12)

**Goal**: Make the system production-ready.

9. Integrate Longbow or implement fallback vector storage
10. Add alert fatigue management to SSE hub
11. Implement async scoring pipeline from architecture.md §P0-1
12. Run deception probe experiments on held-out dataset; measure actual AUROC

### Phase 4: Validation and Benchmarking (Weeks 13–16)

**Goal**: Establish empirical evidence for design claims.

13. Benchmark steering effectiveness vs. capability preservation (GSM8k, MMLU)
14. Test multi-turn steering accumulation with realistic agentic workloads
15. Benchmark activation capture overhead on GPU inference
16. Publish benchmark results; revise targets based on empirical data

---

## 4. References

Full citation list available in `docs/analysis2.md` Appendix A.