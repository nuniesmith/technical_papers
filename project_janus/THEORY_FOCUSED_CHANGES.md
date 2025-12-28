# Theory-Focused Changes to janus.tex

## Summary

The `janus.tex` document has been refactored to focus on **mathematical theory and algorithmic specifications** rather than implementation code examples. This makes it ideal for:

1. **Theoretical verification** - Compare algorithms against implementation
2. **Design documentation** - Mathematical foundations without code clutter
3. **Academic review** - Focus on novel contributions and theory
4. **Logic validation** - Use equations to verify code correctness

---

## Major Changes

### 1. Removed Long Code Examples

**Before:** 100+ lines of Rust code scattered throughout Part 5  
**After:** Algorithmic specifications with mathematical notation

### 2. Replaced with Mathematical Formulations

All implementation details now expressed as:
- Mathematical equations
- Algorithmic pseudocode
- Theoretical specifications
- Complexity analysis

---

## Detailed Changes by Section

### Part 5: Rust Implementation

#### Section 3.2: Core Data Structures
- **Removed:** 15 lines of Rust struct definitions
- **Replaced with:**
  - Mathematical state representation: $\mathcal{S}_t = (\tau_t, \mathbf{f}_t, \mathcal{O}_t, \mathbf{c}_t)$
  - Order book formalization: $\mathcal{B}_t = \{(p_i, q_i)\}$, $\mathcal{A}_t = \{(p_j, q_j)\}$

#### Section 3.3: GAF Transformation Module
- **Removed:** 25 lines of Rust implementation
- **Replaced with:**
  - Algorithmic pseudocode (8 steps)
  - Complexity analysis: $\mathcal{O}(W^2)$
  - Mathematical operations: normalization, angle computation, matrix construction

#### Section 3.4: LTN Constraint Evaluation
- **Removed:** 20 lines of Rust code for constraint struct and methods
- **Replaced with:**
  - Constraint structure: $\mathcal{C}_k = (P_k, w_k)$
  - Evaluation function: $\text{Eval}(\mathcal{C}_k, \mathcal{S}_t) = w_k \cdot P_k(\mathcal{S}_t)$
  - T-norm operations (already defined in Part 2)
  - Total satisfaction formula

#### Section 3.5: Async Service Architecture
- **Removed:** 15 lines of Tokio async code
- **Replaced with:**
  - Request processing pipeline (enumerated steps)
  - Throughput equation: $\text{Throughput} = \frac{N_{\text{workers}} \times 1000}{T_{\text{avg}}}$
  - Performance characteristics (bulleted list)

#### Section 4.1: Prioritized Experience Replay
- **Removed:** 30 lines of buffer implementation code
- **Replaced with:**
  - Buffer state representation: $\mathcal{B} = \{(e_i, p_i)\}_{i=1}^{N}$
  - Hyperparameters ($\alpha$, $\beta$, $C$)
  - Sampling algorithm (pseudocode)
  - Importance weight formula: $w_i = \left(\frac{1}{N \cdot P(i)}\right)^\beta$

#### Section 4.2: Schema Consolidation
- **Removed:** 35 lines of async Rust code with Qdrant client
- **Replaced with:**
  - Schema update algorithm (pseudocode with 8 steps)
  - Schema metadata specification
  - K-means objective function: $\min_{\mathcal{C}} \sum_{k=1}^{K} \sum_{i \in C_k} \|\mathbf{h}_i - \mathbf{z}_k\|^2$

#### Section 5.1: Docker Compose Setup
- **Removed:** 35 lines of YAML configuration
- **Replaced with:**
  - Service topology (3 services enumerated)
  - Communication flow equation
  - Volume management strategy
  - Resource limits (memory, CPU)

### Part 3: Backward Service

#### Section 3.1: Schema Storage
- **Removed:** JSON code example
- **Replaced with:**
  - Schema representation: $\mathcal{S}_k = (\text{id}_k, \mathbf{z}_k, \mathcal{M}_k)$
  - Metadata definitions (count, average reward, volatility)
  - Storage invariant (L2 normalization)

### Part 4: Neuromorphic Architecture

#### Section 2.3: Basal Ganglia
- **Removed:** 2 Rust function snippets
- **Replaced with:**
  - Direct pathway equation: $\mathbf{d}_{\text{direct}} = \text{ReLU}(\mathbf{W}_{\text{direct}} \mathbf{h} + \mathbf{b}_{\text{direct}})$
  - Indirect pathway equation
  - Action selection: $\mathbf{a}_t = \text{softmax}(\mathbf{d}_{\text{direct}} - \lambda \cdot \mathbf{d}_{\text{indirect}})$

#### Section 2.5: Amygdala
- **Removed:** 4 lines of if-statement code
- **Replaced with:**
  - Mahalanobis distance formula: $D_M(\mathbf{s}_t) = \sqrt{(\mathbf{s}_t - \boldsymbol{\mu})^T \boldsymbol{\Sigma}^{-1} (\mathbf{s}_t - \boldsymbol{\mu})}$
  - Circuit breaker condition (piecewise function)
  - Additional threat signals (volatility spike, drawdown, liquidity)

### Part 1: Main Architecture
- **Changed:** Removed missing `\input{../main_content.tex}`
- **Replaced with:** Overview placeholder with bullet points
- **Note:** Focus is now on Parts 2-5 which contain the detailed theory

---

## Document Statistics

### Before Theory Focus
- **Pages:** 49
- **File size:** 383 KB
- **Code listings:** ~15 code blocks
- **Total code lines:** ~250+

### After Theory Focus
- **Pages:** 26
- **File size:** 278 KB
- **Code listings:** 0 (all removed)
- **Mathematical equations:** 50+
- **Algorithms:** 5 pseudocode blocks

---

## Benefits of Theory-Focused Approach

### 1. **Code Verification**
Use mathematical specifications to verify implementation:
```
Theory: G_{ij} = cos(φ_i + φ_j)
Code:   gaf[i * window_size + j] = (angles[i] + angles[j]).cos()
✓ Match confirmed
```

### 2. **Language Agnostic**
Mathematical notation works for:
- Rust implementation
- Python prototyping
- Julia numerical computing
- Any future language choice

### 3. **Academic Rigor**
Equations provide:
- Precise definitions
- Complexity analysis
- Theoretical guarantees
- Peer review foundation

### 4. **Maintenance**
Theory remains stable while code evolves:
- Refactoring doesn't invalidate documentation
- Multiple implementations can coexist
- Clear contract between theory and practice

---

## How to Use This Document

### For Implementation Verification

1. **Read the algorithm** - Understand the mathematical specification
2. **Check your code** - Ensure it implements the exact operations
3. **Verify complexity** - Confirm your implementation meets bounds
4. **Test edge cases** - Use equations to derive test scenarios

### Example Workflow

**Theory (Section 3.3):**
```
Algorithm: GAF Computation
1. Normalize X to [-1, 1]
2. φ_i ← arccos(x̃_i)
3. For i,j: G_ij ← cos(φ_i + φ_j)
Complexity: O(W²)
```

**Code Verification:**
```rust
// 1. Normalize ✓
let normalized = normalize(time_series)?;

// 2. Compute angles ✓
let angles: Vec<f32> = normalized
    .iter()
    .map(|&x| x.acos())
    .collect();

// 3. Build matrix ✓
for i in 0..W {
    for j in 0..W {
        gaf[i*W + j] = (angles[i] + angles[j]).cos();
    }
}
// Complexity: O(W²) ✓
```

### For New Features

1. **Start with theory** - Define mathematical specification first
2. **Derive algorithm** - Convert to pseudocode
3. **Implement** - Write code following algorithm exactly
4. **Validate** - Test against theoretical properties

---

## Sections with Pure Theory (No Code Changes Needed)

These sections already contained only mathematical theory:

- **Part 2, Section 1.1:** Gramian Angular Fields equations
- **Part 2, Section 1.3:** ViViT transformer architecture
- **Part 2, Section 2:** Logic Tensor Networks (all subsections)
- **Part 2, Section 3:** Multimodal Fusion equations
- **Part 3, Section 1:** Memory Hierarchy mathematics
- **Part 3, Section 2:** UMAP objective functions

---

## Compilation

The theory-focused document compiles successfully:

```bash
cd project_janus
pdflatex janus.tex
pdflatex janus.tex  # Second pass for TOC
```

**Output:** `janus.pdf` (26 pages, 278 KB)

---

## Next Steps

### Recommended Additions

1. **Proof sketches** - Add theoretical guarantees where applicable
2. **Complexity analysis** - Expand on computational costs
3. **Convergence proofs** - For iterative algorithms (K-means, UMAP)
4. **Stability analysis** - For numerical methods

### Future Enhancements

- [ ] Add algorithm correctness proofs
- [ ] Include convergence rate analysis
- [ ] Expand on approximation bounds
- [ ] Add information-theoretic analysis (entropy, mutual information)
- [ ] Include Lyapunov stability for control systems

---

## Contact

For questions about the theory-focused refactoring:
- **Author:** Jordan Smith
- **Repository:** [github.com/nuniesmith/technical_papers](https://github.com/nuniesmith/technical_papers)
- **Document:** `project_janus/janus.tex`

---

**Last Updated:** December 28, 2024  
**Version:** 2.0 (Theory-Focused)