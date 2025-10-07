# Memory Embedding Storage Work Breakdown

This note captures a quick estimate and recommended task breakdown for augmenting the
existing filesystem memory repository with vector embedding support while continuing to
store raw text.  It reflects the current architecture in `context_memory.py` and outlines
how we could implement hybrid retrieval in incremental steps.

## Current State
- `MemoryItem` already stores a `plain_text` field populated by `_extract_plain_text`,
  ensuring every record has a human-readable body.
- `MemoryRepo.search_semantic` is defined but returns lexical results in the
  `FsMemoryRepo` implementation because no embedding backend exists yet.
- The context builder performs ranking/merging by combining recency and text heuristics,
  so vector results would need to slot into the same pipeline once available.

## Proposed Stages
1. **Schema & Persistence Plumbing (Small/Medium)**
   - Extend `MemoryItem` (or companion metadata) to accept an optional embedding vector
     while keeping `plain_text` mandatory.
   - Update serialization in `FsMemoryRepo` to persist embeddings, gating them behind a
     feature flag so text-only operation still works.

2. **Embedding Generation Service (Medium)**
   - Introduce a utility that takes new/updated items, calls the chosen embedding model,
     and caches the resulting vector alongside the item.
   - Add backfill tooling for legacy records.

3. **Vector Index & Hybrid Retrieval (Medium/Large)**
   - Integrate a lightweight vector index (FAISS, chroma, or PG vector) depending on the
     deployment target.
   - Implement `search_semantic` to query the vector index and blend results with the
     existing text search.

4. **Testing & Monitoring (Small/Medium)**
   - Unit tests covering serialization, embedding generation, and ranking merge logic.
   - Add metrics/logging hooks to observe embedding coverage and query performance.

## Recommendation
Treat this as a small sequence of deliverables rather than one monolithic task.  Shipping
stage 1 on its own keeps compatibility with text-only storage.  Each subsequent stage can
be tackled independently, enabling incremental validation and rollback points.
