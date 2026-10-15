#ifndef _WIN32
#define _POSIX_C_SOURCE 200809L
#endif

/*
 * Make Seven DF-PN v9.8.3 Proof-Pressure Gated Sticky Resume solver (unlimited-tile research variant)
 *
 * Rules implemented:
 *   - 7 columns; playable height 5, 6, or 7 via --rows.
 *   - Tiles 1, 2, 3 are unlimited.
 *   - Tile 3 may only land on the original red cells.
 *   - Sliding contiguous subsequences summing exactly to 7 are wins.
 *
 * Search:
 *   - two-phase DF-PN for exact W/L/D
 *   - v9.8.2 shared-memory SPDFPN-style bounded-job parallel mode
 *   - shared canonical exact PN/DN + coarse finite PN/DN/work exchange
 *   - v9.8.2 immutable persistent exact-proposition database (outcomes + partial W/L facts)
 *   - active-subtree occupancy weighted virtual scheduling (scheduler-only)
 *   - exact immediate-threat / forced-defense / double-threat pruning
 *   - exact canonical sibling-state deduplication
 *   - exact endgame dependency dead-prover
 *   - v6 first-pick ordering diagnostics
 *   - v8 exact root/depth-2 proof-frontier work profiling
 *   - v9.8.2 deep move-prefix proof-frontier profiling through configurable depth
 *   - v9.8.2 per-prefix sibling work share / PN-DN / stall / TSS / branching diagnostics
 *   - v8 root-stall accounting (expansions since PN/DN last changed)
 *   - v9.8.2 automatic deep-node PN/DN-stall detection
 *   - v9.8.2 persistent focused/recursive deep-reroot DF-PN for stalled deep prefixes
 *   - exact non-replacing reroot overlay + prompt recursive unwind to root
 *   - v9.8.2 exact-descendant promotion from nested reroot into global shared TT
 *   - v9.8.2 adaptive MPN-chain exact harvesting after persistent outer UNKNOWN slices
 *   - v9.8.2 shallow-first harvest ladder: +4 before +5 before +6 on the current MPN chain
 *   - v9.8.2 two-slice sticky budget on +4; one slice on +5/+6
 *   - v9.8.2 evidence gate on +6: minimum accumulated work AND stall/work stability
 *   - v9.8.2 proof-pressure gate on the second +4 slice: resume only when min(pn,dn)<=128 by default
 *   - immediate main-root ancestor reaggregation after promoted exact results
 *   - optional v9.4 work-quantum DF-PN retained for A/B testing
 *   - v9 dedicated bounded exact DFS + independent exact-result TT
 *   - v9 conservative low-yield TSS retry throttling inside hot descendants
 *   - optional dependency-first quiet ordering with persistent top-4 TT hints
 *   - exact prover-dead detection (no remaining viable sum-to-7 segment)
 *   - sound/incomplete dependency proof oracle for quiet setup chains
 *   - adaptive sound/incomplete TSS with attacker threat-signature compression
 *   - TSS proof memoization and immediate propagation into DF-PN TT values
 *   - optional bounded exact tactical pre-solver (off by default)
 *   - optional quiet-setup bridge proof oracle (off by default)
 *   - fixed-perspective solving for independent opening analysis
 *   - exact-state, symmetry-canonicalized, 2-way transposition table
 *   - periodic wall-clock benchmark logging to console and CSV
 *
 * Build (MSYS2 UCRT64 / MinGW GCC):
 *   gcc -O3 -std=c23 -march=native -flto -Wall -Wextra -fopenmp dfpn_solver_v9_8_2.c -o Make7DFPNv9_8_2.exe
 */

#include <ctype.h>
#include <errno.h>
#include <inttypes.h>
#include <stdbool.h>
#include <signal.h>
#include <stdint.h>
#include <stdatomic.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#ifdef _OPENMP
#include <omp.h>
#endif

#ifdef _WIN32
#include <windows.h>
#else
#include <time.h>
#endif

#define COLS 7
#define MAX_ROWS 7
#define STRIDE 8
#define MAX_MOVES 21
#define THREE_MASK UINT64_C(0x4102008201004)
#define PN_INF UINT32_C(0x3fffffff)
#define MAX_SEGMENTS 384
#define CHILD_CACHE_MAX_MOVES MAX_MOVES
#define CHILD_CACHE_DEFAULT_MB 32.0
#define DEEP_FRONTIER_HARD_MAX_DEPTH 10
#define DEEP_FRONTIER_DEFAULT_DEPTH 6
#define DEEP_FRONTIER_DEFAULT_NODES 1000000u
#define REROOT_OVERLAY_SIZE 1024u
#define REROOT_DEFAULT_TT_GB 0.25
#define REROOT_DEFAULT_CACHE_MB 16.0
#define REROOT_SESSION_MAX_SLOTS 4u
#define REROOT_SESSION_DEFAULT_SLOTS 2u
#define REROOT_DEFAULT_FOCUS_SLICES 4u
#define REROOT_HARVEST_AFTER_SLICES 2u
#define REROOT_HARVEST_MIN_DEPTH 4u
#define REROOT_HARVEST_MAX_DEPTH 6u
#define REROOT_HARVEST_ATTEMPTS 2u
#define REROOT_HARVEST_FOCUS_SLICES 2u
#define REROOT_HARVEST_MIN_WORK UINT64_C(1000)
#define REROOT_HARVEST_DEEP_MIN_WORK UINT64_C(50000)
#define REROOT_HARVEST_DEEP_MIN_STABILITY 0.50
#define REROOT_HARVEST_RESUME_PRESSURE 128u
#define REROOT_HARVEST_TIME_SEC 5.0
#define SP_REROOT_TRACK_SLOTS 32u
#define SP_HOT_TRACK_SLOTS 64u
#define SP_DEFAULT_HOT_AGE_SEC 20.0
#define SP_DEFAULT_HOT_WORK UINT64_C(1000000)
#define SP_DEFAULT_HOT_MAX_CHANGE 0.10
#define SP_DEFAULT_HOT_PROBE_SEC 0.50
#define SP_DEFAULT_HOT_GAP_SEC 5.0
#define SP_DEFAULT_HOT_MIN_BUSY 2u
#define SP_DEFAULT_HOT_REGION_DEPTH 3u

/* v9.8.2 shared-memory SPDFPN-style parallel engine. */
#define SP_MAX_THREADS 32u
#define SP_PATH_MAX 50u
#define SP_DEFAULT_THREADS 1u
#define SP_DEFAULT_SHARED_TT_GB 1.0
#define SP_DEFAULT_WORKER_TT_MB 128.0
#define SP_DEFAULT_WORKER_CACHE_MB 8.0
#define SP_DEFAULT_JOB_WORK 4096u
#define SP_DEFAULT_MIN_JOB_DEPTH 1u
#define SP_DEFAULT_PUBLISH_WORK 0u
#define SP_DEFAULT_PUBLISH_DEPTH 8u
#define SP_DEFAULT_BUSY_PENALTY 0.02

#define TT_FLAG_EXPANDED UINT8_C(1)
#define TT_FLAG_TSS_TRIED UINT8_C(2)
#define TT_FLAG_TSS_PROVEN_GREEN UINT8_C(4)
#define TT_FLAG_TSS_PROVEN_YELLOW UINT8_C(8)
#define TT_FLAG_DEP_DEAD_CHECKED UINT8_C(16)
#define TT_FLAG_DEP_ORACLE_TRIED UINT8_C(32)
#define TT_FLAG_DEP_HINT_READY UINT8_C(64)
#define TT_FLAG_DEP_FULL_HINT_READY UINT8_C(128)

static const int CENTER_ORDER[COLS] = {3, 2, 4, 1, 5, 0, 6};
static const int CENTER_RANK[COLS]  = {5, 3, 1, 0, 2, 4, 6};

typedef enum {
    WIN_NONE = 0,
    WIN_OTHER = 1,
    WIN_2221 = 2,
    WIN_322 = 3
} WinClass;

typedef struct {
    uint64_t p[2];
    uint64_t twos;
    uint64_t threes;
    uint8_t height[COLS];       /* next original-board row in each column */
    uint8_t turn;               /* 0 Green, 1 Yellow */
    uint8_t ply;
} Board;

#define EXACT_DB_POSITION_MAX 128u

#define EXACT_PROP_G_PROVED      UINT8_C(1)
#define EXACT_PROP_G_DISPROVED   UINT8_C(2)
#define EXACT_PROP_Y_PROVED      UINT8_C(4)
#define EXACT_PROP_Y_DISPROVED   UINT8_C(8)


typedef struct {
    uint64_t k[4];
    uint8_t prop_mask;           /* exact phase facts; see EXACT_PROP_* below */
    uint8_t valid;
    uint16_t reserved16;
    char position[EXACT_DB_POSITION_MAX];
} ExactDBEntry;

/* v9.8.2 fast path: store both board orientations as raw exact keys.
   Runtime lookup therefore avoids mirror_bits(), tuple comparison, and the
   expensive canonical-key hash that the normal TT will compute immediately
   afterwards anyway.  The canonical table above is retained only for
   load-time duplicate/conflict validation and reporting. */
typedef struct {
    uint64_t k[4];
    uint8_t prop_mask;
    uint8_t valid;
    uint16_t reserved16;
} ExactDBRawEntry;

typedef struct {
    ExactDBEntry *slot;
    size_t sets, mask;
    ExactDBRawEntry *raw_slot;
    size_t raw_sets, raw_mask, raw_count;
    ExactDBRawEntry *raw_list;
    size_t raw_list_cap;
    uint64_t ply_mask;           /* exact facts exist only at these absolute plies */
    size_t count;                /* unique canonical states */
    size_t outcome_facts;
    size_t proposition_facts;
    size_t duplicates;
    size_t skipped_rows;
    const char *path;
} ExactDB;

typedef struct {
    uint8_t colmask;
    uint8_t best_class;
    uint8_t best_tile;
} ThreatInfo;

typedef struct {
    uint8_t len;
    uint8_t cell[7];            /* bit index = col*8+row */
    uint64_t mask;
} WinSegment;

typedef struct {
    uint16_t direct[2][56];     /* empty cells in still-viable winning segments */
    uint16_t support[2][56];    /* lower cells needed to make targets landable */
    uint16_t near_score[2][56];       /* emphasis for 1-2 empty-cell segments */
    uint16_t viable[2];
    uint16_t one_empty[2];
    uint16_t two_empty[2];
    uint64_t direct_mask[2];
    uint64_t support_mask[2];
} DependencyMap;

typedef struct {
    Board board;
    uint8_t move;
    uint8_t priority;
    int16_t dep_score;          /* dependency tie-break / quiet-move ordering */
    int8_t terminal_winner;     /* -1 unknown, 0/1 winner, 2 draw */
} Child;

static WinSegment g_segments[MAX_SEGMENTS];
static unsigned g_segment_count = 0;
static uint8_t g_segment_floor = 255;
static uint8_t g_cell_seg_count[56];
static uint16_t g_cell_segs[56][64];

typedef struct {
    uint64_t k0, k1, k2, k3;
    uint32_t pn, dn;
    uint32_t generation;
    uint32_t examined_mask;     /* canonical move bits selected for recursive DF-PN */
    uint8_t best_move;          /* canonical orientation */
    uint8_t dep_hint_move[4];   /* cached top-4 dependency order, canonical */
    uint8_t flags;
    uint8_t tss_level;          /* next adaptive threat-chain level to try */
    uint8_t bridge_level;       /* next selective bridge-proof level to try */
} TTEntry;

typedef struct {
    TTEntry way[2];
} TTBucket;

typedef struct {
    TTBucket *bucket;
    size_t sets;
    size_t mask;
    double allocated_gb;
} TT;

/*
 * v7 child-generation cache.  This is separate from the proof-number TT: it
 * stores only the exact, safely-pruned child move set for a state.  A full
 * canonical key is retained, so hash collisions can only cause misses, never
 * an incorrect child list.  Moves are stored in canonical orientation.
 */
typedef struct {
    uint64_t k0, k1, k2, k3;
    uint32_t generation;
    uint8_t count;
    uint8_t quiet_start;           /* first quiet child in cached base order */
    int8_t status;                 /* -1 normal, 0/1 winner, 2 draw */
    uint8_t move[CHILD_CACHE_MAX_MOVES];
    int8_t terminal[CHILD_CACHE_MAX_MOVES];
} ChildCacheEntry;

typedef struct {
    ChildCacheEntry *entry;
    size_t sets;
    size_t mask;
    double allocated_mb;
} ChildCache;

/* v9 independent exact-result cache. UNKNOWN is never stored. */
typedef struct {
    uint64_t k0, k1, k2, k3;
    int8_t result;
    uint8_t valid;
    uint16_t pad16;
    uint32_t pad32;
} HotExactEntry;

typedef struct {
    HotExactEntry *entry;
    size_t sets;
    size_t mask;
    double allocated_mb;
} HotExactTT;

/* v8 proof-frontier diagnostics.  These counters never affect search choice. */
typedef struct {
    bool valid;
    uint8_t move;
    Board board;
    uint64_t descents;
    uint64_t work_calls;
    uint64_t work_expansions;
    uint64_t work_tss_calls;
    uint64_t work_tss_nodes;
    uint64_t work_tss_proofs;
    uint64_t work_collapses;
    uint64_t work_tt_hits;
    uint64_t work_child_cache_hits;
    uint64_t hot_exact_attempts;
    uint64_t hot_exact_nodes;
    uint64_t hot_last_attempt_work;
    uint8_t hot_attempt_level;
    int8_t hot_result;
    bool hot_result_valid;
    bool tss_gate_active;
} FrontierReply;

typedef struct {
    bool valid;
    uint8_t move;
    Board board;
    uint64_t descents;
    uint64_t work_calls;
    uint64_t work_expansions;
    uint64_t work_tss_calls;
    uint64_t work_tss_nodes;
    uint64_t work_tss_proofs;
    uint64_t work_collapses;
    uint64_t work_tt_hits;
    uint64_t work_child_cache_hits;
    uint8_t reply_count;
    FrontierReply reply[MAX_MOVES];
} FrontierChild;

typedef struct {
    bool initialized;
    bool root_is_or;
    uint8_t child_count;
    FrontierChild child[MAX_MOVES];
    uint32_t last_root_pn, last_root_dn;
    uint64_t expansions_at_root_change;
    bool snapshot_valid;
    uint32_t prev_snapshot_pn, prev_snapshot_dn;
    uint64_t prev_snapshot_expansions;
} FrontierProfile;

typedef struct {
    uint64_t calls, expansions, tss_calls, tss_nodes, tss_proofs;
    uint64_t collapses, tt_hits, child_cache_hits;
} WorkStamp;

/* v9.8.2 deep proof-frontier profiler.  This is diagnostic-only and never
   changes move choice, thresholds, TT values, or exact-result semantics.
   Each node represents a concrete move-prefix from the current phase root. */
typedef struct {
    bool valid;
    uint8_t depth;                 /* moves from profiling root; root = 0 */
    uint8_t move;                  /* move from parent; root = 0 */
    uint8_t branching;             /* exact generated child count at this node */
    uint8_t pair_valid;
    Board board;
    int32_t parent;
    int32_t first_child;
    int32_t next_sibling;
    uint64_t descents;
    uint64_t work_calls;
    uint64_t work_expansions;
    uint64_t work_tss_calls;
    uint64_t work_tss_nodes;
    uint64_t work_tss_proofs;
    uint64_t work_collapses;
    uint64_t work_tt_hits;
    uint64_t work_child_cache_hits;
    uint32_t last_pn, last_dn;
    uint64_t work_at_pair_change;
    uint8_t reroot_attempts;
    uint8_t reroot_result_valid;
    uint8_t reroot_result;          /* 1 prover can force win, 0 disproven */
    uint8_t harvest_attempts;         /* adaptive exact-harvest slices for this node */
    uint64_t reroot_last_attempt_work;
} DeepFrontierNode;

typedef struct {
    DeepFrontierNode *node;
    uint32_t count;
    uint32_t capacity;
    uint64_t dropped_nodes;
    unsigned max_depth;
    unsigned top;
} DeepFrontierProfile;

typedef struct {
    bool valid;
    uint8_t prover;
    uint8_t reserved[3];
    uint64_t k[4];
    uint32_t pn, dn;
} RerootProof;

typedef struct {
    uint64_t dfpn_calls;
    uint64_t expansions;
    uint64_t child_generations;
    uint64_t moves_generated;
    uint64_t child_cache_probes;
    uint64_t child_cache_hits;
    uint64_t child_cache_stores;
    uint64_t child_cache_restored;
    uint64_t tt_probes;
    uint64_t tt_hits;
    uint64_t tt_replacements;
    uint64_t tss_calls;
    uint64_t tss_nodes;
    uint64_t tss_proofs;
    uint64_t tss_retries;
    uint64_t tss_deep_proofs;
    uint64_t tss_cache_hits;
    uint64_t tss_forcing_candidates;
    uint64_t tss_candidates_kept;
    uint64_t tss_signature_pruned;
    uint64_t sibling_dedup_pruned;
    uint64_t dep_maps;
    uint64_t dep_segments_tested;
    uint64_t dep_viable_green;
    uint64_t dep_viable_yellow;
    uint64_t dep_ordered_moves;
    uint64_t dep_dead_prunes;
    uint64_t dep_oracle_calls;
    uint64_t dep_oracle_nodes;
    uint64_t dep_oracle_proofs;
    uint64_t dep_oracle_candidates;
    uint64_t dep_oracle_defender_replies;
    uint64_t dep_oracle_budget_exhausted;
    uint64_t bridge_calls;
    uint64_t bridge_nodes;
    uint64_t bridge_proofs;
    uint64_t bridge_candidates;
    uint64_t bridge_defender_replies;
    uint64_t bridge_budget_exhausted;
    uint64_t oracle_collapses;
    uint64_t expanded_children;
    uint64_t child_descents;
    uint64_t unique_children_examined;
    uint64_t unique_top3_examined;
    uint64_t unique_rank_sum;
    uint64_t first_pick_nodes;
    uint64_t first_pick_top3;
    uint64_t first_pick_rank_sum;
    uint64_t dep_light_nodes;
    uint64_t dep_light_segments_tested;
    uint64_t dep_hint_uses;
    uint64_t dep_dead_checks;
    uint64_t tactical_calls;
    uint64_t tactical_nodes;
    uint64_t tactical_children_solved;
    uint64_t tactical_wins;
    uint64_t tactical_losses;
    uint64_t tactical_draws;
    uint64_t tactical_budget_exhausted;
    uint64_t hot_exact_triggers;
    uint64_t hot_exact_attempts;
    uint64_t hot_exact_nodes;
    uint64_t hot_exact_tt_hits;
    uint64_t hot_exact_proofs;
    uint64_t hot_exact_wins;
    uint64_t hot_exact_losses;
    uint64_t hot_exact_draws;
    uint64_t hot_exact_budget_exhausted;
    uint64_t hot_exact_depth_cutoffs;
    uint64_t hot_tss_gate_activations;
    uint64_t hot_tss_gated_nodes;
    uint64_t work_quantum_yields;
    uint64_t work_quantum_yield_expansions;
    uint64_t work_quantum_max_call_exp;
    uint64_t root_mpc_switches;
    uint64_t deep_reroot_triggers;
    uint64_t deep_reroot_attempts;
    uint64_t deep_reroot_proofs;
    uint64_t deep_reroot_proved;
    uint64_t deep_reroot_disproved;
    uint64_t deep_reroot_unknown;
    uint64_t deep_reroot_sub_calls;
    uint64_t deep_reroot_sub_expansions;
    uint64_t deep_reroot_sub_ms;
    uint64_t deep_reroot_overlay_hits;
    uint64_t deep_reroot_fresh_sessions;
    uint64_t deep_reroot_resumes;
    uint64_t deep_reroot_session_evictions;
    uint64_t deep_reroot_sub_tt_probes;
    uint64_t deep_reroot_sub_tt_hits;
    uint64_t deep_reroot_slices;
    uint64_t deep_reroot_slice_unknown;
    uint64_t deep_reroot_focus_extra_slices;
    uint64_t deep_reroot_focus_exhausted;
    uint64_t forced_defenses;
    uint64_t double_threat_wins;
    uint64_t double_threat_losses;
    uint64_t unsafe_pruned;
    uint64_t symmetry_pruned;
    uint64_t immediate_terminals;
    uint32_t max_ply;
} Stats;

/* v9.8.2 persistent/resumable reroot session.  The sub-context is heap-owned
   so its TT, child cache, deep-frontier state, exact overlay, and any nested
   reroot sessions survive UNKNOWN time slices.  The parent still imports only
   exact pn==0/dn==0 results. */
typedef struct {
    bool valid;
    uint8_t prover;
    uint8_t reserved[2];
    uint64_t k[4];
    Board root;
    void *sub_context;
    uint64_t uses;
    uint64_t last_used_tick;
} RerootSession;

typedef struct ParallelShared ParallelShared;

typedef struct {
    TT tt;
    ChildCache child_cache;
    HotExactTT hot_tt;
    FrontierProfile frontier;
    DeepFrontierProfile deep_frontier;
    Stats stats;
    uint32_t generation;
    uint8_t prover;
    uint8_t floor_row;
    bool tss_enabled;
    bool tss_compress;
    unsigned tss_depth;
    uint64_t tss_budget;
    unsigned tss_start_ply;
    unsigned tss_levels;
    unsigned tss_depth_step;
    unsigned tss_budget_mult;
    bool tactical_enabled;
    unsigned tactical_start_ply;
    unsigned tactical_depth;
    unsigned tactical_candidates;
    uint64_t tactical_budget;
    bool bridge_enabled;
    unsigned bridge_start_ply;
    unsigned bridge_layers;
    unsigned bridge_candidates;
    unsigned bridge_levels;
    uint64_t bridge_budget;
    unsigned bridge_budget_mult;
    bool dependency_enabled;
    bool dep_full_enabled;
    bool dep_oracle_enabled;
    bool legacy_quiet_order;
    bool or_dependency_only;
    bool child_cache_enabled;
    bool frontier_enabled;
    unsigned frontier_top;
    bool deep_frontier_enabled;
    unsigned deep_frontier_depth;
    unsigned deep_frontier_top;
    uint32_t deep_frontier_max_nodes;
    bool hot_exact_enabled;
    double hot_exact_tt_mb;
    unsigned hot_exact_depth;
    unsigned hot_exact_depth_step;
    unsigned hot_exact_levels;
    uint64_t hot_exact_budget;
    unsigned hot_exact_budget_mult;
    uint64_t hot_exact_min_total_exp;
    uint64_t hot_exact_stall_exp;
    uint64_t hot_exact_min_reply_exp;
    uint64_t hot_exact_retry_work;
    double hot_exact_child_share;
    double hot_exact_reply_share;
    bool hot_tss_gate_enabled;
    uint64_t hot_tss_min_calls;
    double hot_tss_max_proof_rate;
    unsigned hot_tss_level_cap;
    bool hot_tss_gate_valid;
    Board hot_tss_gate_board;

    /* v9.8.2 work quantum.  Zero means classic/unlimited DF-PN.  The quantum
       counts exact DF-PN expansions performed during one non-root recursive
       invocation; oracle-internal TSS/tactical nodes are deliberately excluded. */
    uint64_t work_quantum;

    /* v9.8.3 exact lemma closure oracle.  It only consumes terminal states,
       immutable exact-DB facts, or already-exact TT facts.  UNKNOWN leaves
       remain UNKNOWN; finite PN/DN values are never used as proof evidence. */
    unsigned exact_lemma_depth;
    uint64_t exact_lemma_budget;

    /* v9.8.2 deep-reroot DF-PN.  When a deep proof-frontier prefix dominates
       after a long root stall, solve that concrete board as an independent
       DF-PN root. UNKNOWN sessions retain their TT/context and resume later.  Only exact binary
       proposition results (pn==0 or dn==0 for the current prover) are kept. */
    bool deep_reroot_enabled;
    double deep_reroot_memory_gb;
    double deep_reroot_child_cache_mb;
    double deep_reroot_time_sec;
    uint64_t deep_reroot_min_total_exp;
    uint64_t deep_reroot_stall_exp;
    uint64_t deep_reroot_min_work;
    uint64_t deep_reroot_retry_work;
    /* v9.8.2 parallel-aware hot-region detector.  Serial reroot keeps the
       legacy root/node-stall fields above; SPDFPN uses these wall-time/work
       gates so harmless root PN/DN motion does not suppress reroot forever. */
    double deep_reroot_hot_age_sec;
    uint64_t deep_reroot_hot_work;
    double deep_reroot_hot_max_change;
    double deep_reroot_hot_probe_sec;
    double deep_reroot_hot_gap_sec;
    unsigned deep_reroot_hot_min_busy;
    unsigned deep_reroot_hot_region_depth;
    double deep_reroot_root_share;
    double deep_reroot_inner_share;
    unsigned deep_reroot_min_depth;
    unsigned deep_reroot_max_attempts;
    unsigned deep_reroot_max_nesting;
    unsigned deep_reroot_session_slots;
    uint32_t deep_reroot_profile_nodes;
    unsigned deep_reroot_focus_slices;
    /* v9.8.2 shallow-first recursive reroot / exact harvesting.  After an outer
       persistent reroot has consumed N UNKNOWN slices, intentionally descend
       another 4..6 plies along its current private-TT MPN chain and run a
       short persistent nested reroot.  Exact descendants are promoted through
       the exact-only shared-TT promotion channel; finite estimates never escape. */
    bool deep_reroot_harvest_enabled;
    unsigned deep_reroot_harvest_after_slices;
    unsigned deep_reroot_harvest_min_depth;
    unsigned deep_reroot_harvest_max_depth;
    unsigned deep_reroot_harvest_attempts;
    unsigned deep_reroot_harvest_focus_slices;
    uint64_t deep_reroot_harvest_min_work;
    uint64_t deep_reroot_harvest_deep_min_work;
    double deep_reroot_harvest_deep_min_stability;
    uint32_t deep_reroot_harvest_resume_pressure;
    double deep_reroot_harvest_time_sec;
    unsigned deep_reroot_level;
    bool deep_reroot_tss;
    bool deep_reroot_unwind;
    RerootProof reroot_overlay[REROOT_OVERLAY_SIZE];
    RerootSession reroot_session[REROOT_SESSION_MAX_SLOTS];
    uint64_t reroot_session_tick;

    /* v9.8.2 immutable exact-proposition database.  Facts may be full outcomes
       (Green win / Yellow win / draw) or individual Green/Yellow phase
       PROVED/DISPROVED propositions. UNKNOWN/finite PN-DN values are never imported. */
    const ExactDB *exact_db;

    /* v9.8.2 shared-memory parallel job engine.  Worker contexts keep private
       full DF-PN metadata but exchange canonical PN/DN/work through one
       shared table.  Only finite search estimates are shared; exact zero
       proof/disproof values remain exact. */
    ParallelShared *parallel_shared;
    /* Cancellation-only pointer used by independent reroot subsolvers.  Unlike
       parallel_shared it does not import/publish PN/DN; it only notices that
       the main shared proof has already finished. */
    ParallelShared *parallel_stop_shared;
    /* v9.8.2 exact-descendant promotion channel.  Reroot subsolvers never
       import finite shared PN/DN through this pointer; they may only publish
       mathematically exact pn==0/dn==0 descendants and request ancestor
       reaggregation on the main SP tree. */
    ParallelShared *parallel_exact_shared;
    Board parallel_promote_path[SP_PATH_MAX];
    unsigned parallel_promote_path_len;
    unsigned parallel_worker_id;
    bool parallel_worker;
    uint64_t parallel_job_start_calls;
    uint64_t parallel_job_budget;
    uint64_t parallel_publish_interval;
    uint64_t parallel_next_publish_calls;
    unsigned parallel_publish_depth;
    Board parallel_path[SP_PATH_MAX];

    /* v9.8.2 live frontier attribution.  Committed frontier counters are only
       updated after a recursive descent returns.  A hard subtree can remain
       inside one descent for millions of expansions, so the hot scheduler must
       also see the work currently in flight. */
    int active_frontier_ci;
    int active_frontier_ri;
    uint64_t active_child_start_exp;
    uint64_t active_reply_start_exp;

    /* v9.8.2 active deep-prefix stack for in-flight accounting. */
    int32_t deep_path_node[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    int32_t deep_active_node[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    WorkStamp deep_active_start[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];

    unsigned dep_start_ply;
    unsigned dep_full_start_ply;
    unsigned dep_full_tss_level;
    unsigned dep_dead_empty;
    unsigned dep_deadcheck_ply;
    unsigned dep_oracle_start_ply;
    unsigned dep_oracle_layers;
    unsigned dep_oracle_candidates;
    uint64_t dep_oracle_budget;
    double start_time;
    double phase_start;
    double deadline;
    double report_interval;
    double next_report;
    bool stop;
    const char *phase_name;
    const char *log_path;
    const char *frontier_log_path;
    const char *deep_frontier_log_path;
    FILE *log_fp;
    FILE *frontier_fp;
    FILE *deep_frontier_fp;
    uint32_t root_pn, root_dn;
    uint8_t root_best_move;
} Context;

typedef struct {
    double memory_gb;
    double child_cache_mb;
    double time_limit;
    double report_interval;
    unsigned rows;
    bool tss_enabled;
    bool tss_compress;
    unsigned tss_depth;
    uint64_t tss_budget;
    unsigned tss_start_ply;
    unsigned tss_levels;
    unsigned tss_depth_step;
    unsigned tss_budget_mult;
    bool tactical_enabled;
    unsigned tactical_start_ply;
    unsigned tactical_depth;
    unsigned tactical_candidates;
    uint64_t tactical_budget;
    bool bridge_enabled;
    unsigned bridge_start_ply;
    unsigned bridge_layers;
    unsigned bridge_candidates;
    unsigned bridge_levels;
    uint64_t bridge_budget;
    unsigned bridge_budget_mult;
    bool dependency_enabled;
    bool dep_full_enabled;
    bool dep_oracle_enabled;
    bool legacy_quiet_order;
    bool or_dependency_only;
    bool child_cache_enabled;
    bool frontier_enabled;
    unsigned frontier_top;
    bool deep_frontier_enabled;
    unsigned deep_frontier_depth;
    unsigned deep_frontier_top;
    uint32_t deep_frontier_max_nodes;
    bool hot_exact_enabled;
    double hot_exact_tt_mb;
    unsigned hot_exact_depth;
    unsigned hot_exact_depth_step;
    unsigned hot_exact_levels;
    uint64_t hot_exact_budget;
    unsigned hot_exact_budget_mult;
    uint64_t hot_exact_min_total_exp;
    uint64_t hot_exact_stall_exp;
    uint64_t hot_exact_min_reply_exp;
    uint64_t hot_exact_retry_work;
    double hot_exact_child_share;
    double hot_exact_reply_share;
    bool hot_tss_gate_enabled;
    uint64_t hot_tss_min_calls;
    double hot_tss_max_proof_rate;
    unsigned hot_tss_level_cap;
    uint64_t work_quantum;
    unsigned exact_lemma_depth;
    uint64_t exact_lemma_budget;
    bool deep_reroot_enabled;
    double deep_reroot_memory_gb;
    double deep_reroot_child_cache_mb;
    double deep_reroot_time_sec;
    uint64_t deep_reroot_min_total_exp;
    uint64_t deep_reroot_stall_exp;
    uint64_t deep_reroot_min_work;
    uint64_t deep_reroot_retry_work;
    /* v9.8.2 parallel-aware hot-region detector.  Serial reroot keeps the
       legacy root/node-stall fields above; SPDFPN uses these wall-time/work
       gates so harmless root PN/DN motion does not suppress reroot forever. */
    double deep_reroot_hot_age_sec;
    uint64_t deep_reroot_hot_work;
    double deep_reroot_hot_max_change;
    double deep_reroot_hot_probe_sec;
    double deep_reroot_hot_gap_sec;
    unsigned deep_reroot_hot_min_busy;
    unsigned deep_reroot_hot_region_depth;
    double deep_reroot_root_share;
    double deep_reroot_inner_share;
    unsigned deep_reroot_min_depth;
    unsigned deep_reroot_max_attempts;
    unsigned deep_reroot_max_nesting;
    unsigned deep_reroot_session_slots;
    uint32_t deep_reroot_profile_nodes;
    unsigned deep_reroot_focus_slices;
    bool deep_reroot_harvest_enabled;
    unsigned deep_reroot_harvest_after_slices;
    unsigned deep_reroot_harvest_min_depth;
    unsigned deep_reroot_harvest_max_depth;
    unsigned deep_reroot_harvest_attempts;
    unsigned deep_reroot_harvest_focus_slices;
    uint64_t deep_reroot_harvest_min_work;
    uint64_t deep_reroot_harvest_deep_min_work;
    double deep_reroot_harvest_deep_min_stability;
    uint32_t deep_reroot_harvest_resume_pressure;
    double deep_reroot_harvest_time_sec;
    bool deep_reroot_tss;
    unsigned dep_start_ply;
    unsigned dep_full_start_ply;
    unsigned dep_full_tss_level;
    unsigned dep_dead_empty;
    unsigned dep_deadcheck_ply;
    unsigned dep_oracle_start_ply;
    unsigned dep_oracle_layers;
    unsigned dep_oracle_candidates;
    uint64_t dep_oracle_budget;
    unsigned parallel_threads;
    double parallel_shared_tt_gb;
    double parallel_worker_tt_mb;
    double parallel_worker_cache_mb;
    uint64_t parallel_job_work;
    unsigned parallel_min_job_depth;
    uint64_t parallel_publish_work;
    unsigned parallel_publish_depth;
    double parallel_busy_penalty;
    bool legacy_serial;
    int perspective;            /* -1 side to move, 0 Green, 1 Yellow */
    int phase_only;             /* 0 full game, 1 W only, 2 L only */
    const char *position;
    const char *exact_db_path;
    const char *log_path;
    const char *frontier_log_path;
    const char *deep_frontier_log_path;
    const char *summary_path;
    const char *label;
    bool self_test;
    bool list_legal_moves;
} Options;

typedef struct {
    int verdict;                /* 1 proven, 0 disproven, -1 unknown */
    double elapsed;
    Stats stats;
    uint32_t root_pn, root_dn;
    uint8_t root_best_move;
} PhaseSummary;

/* v9.8.2 shared PN/DN/work table used by all parallel workers.  The legacy
   per-worker TT keeps rich ordering/oracle metadata; this shared layer is the
   canonical cross-thread exchange for proof/disproof estimates and work. */
typedef struct {
    uint64_t k0, k1, k2, k3;
    uint32_t pn, dn;
    uint32_t generation;
    uint32_t pad32;
    uint64_t work;
    /* Shared-path work at the last accepted PN/DN change.  In parallel mode
       this is the exact analogue of the serial deep-frontier node-stall clock. */
    uint64_t pair_change_work;
} SPSharedEntry;

typedef struct {
    atomic_flag lock;
    SPSharedEntry way[2];
} SPSharedBucket;

typedef struct {
    SPSharedBucket *bucket;
    size_t sets, mask;
    double allocated_gb;
    uint32_t generation;
} SPSharedTT;


typedef struct {
    bool active;
    unsigned worker_id;
    unsigned path_len;
    Board path[SP_PATH_MAX];
    uint32_t thpn, thdn;
    uint32_t initial_pn, initial_dn;
} SPActiveJob;

typedef struct {
    bool valid;
    uint8_t prover;
    uint8_t attempts;
    uint8_t reserved[5];
    uint64_t k[4];
    Board board;
    uint64_t last_attempt_work;
    uint64_t last_used_tick;
} SPRerootTrack;

typedef struct {
    bool valid;
    uint8_t prover;
    uint8_t depth;
    uint8_t reserved[6];
    uint64_t k[4];
    Board board;
    double first_seen_time;
    double last_seen_time;
    double exposure_sec;
    uint64_t last_probe_id;
    uint64_t start_work;
    uint64_t last_work;
    uint32_t start_pn, start_dn;
    uint32_t last_pn, last_dn;
    uint64_t last_used_tick;
} SPHotTrack;

typedef struct {
    bool valid;
    unsigned depth;
    unsigned path_len;
    Board target;
    Board path[SP_PATH_MAX];
    uint8_t moves[SP_PATH_MAX];
    uint64_t target_work;
    uint64_t target_stall_work; /* deep target work since its last accepted PN/DN change */
    uint64_t hot_region_work;   /* aggregate anchor work accumulated during hot exposure */
    uint64_t root_stall_exp;    /* legacy diagnostic; zero for parallel hot detector */
    double hot_age_sec;
    double hot_delta_pn;
    double hot_delta_dn;
    unsigned hot_busy;
    unsigned hot_region_depth;
    double root_share;
    double inner_share;
    unsigned attempt;
} SPRerootRequest;

struct ParallelShared {
    SPSharedTT tt;
    const ExactDB *exact_db;
    atomic_uint_fast64_t exact_db_hits;
    atomic_uint_fast64_t exact_db_conflicts;
    atomic_flag job_lock;
    atomic_int solved;           /* 0 unresolved, 1 proposition proved, 2 disproved */
    atomic_uint_fast64_t jobs_started;
    atomic_uint_fast64_t jobs_finished;
    atomic_uint_fast64_t shared_probes;
    atomic_uint_fast64_t shared_hits;
    atomic_uint_fast64_t shared_publishes;
    atomic_uint_fast64_t scheduler_descents;
    atomic_uint_fast64_t scheduler_waits;
    atomic_uint_fast64_t assign_fail_exact;
    atomic_uint_fast64_t assign_fail_threshold;
    atomic_uint_fast64_t assign_fail_terminal;
    atomic_uint_fast64_t assign_fail_pathmax;
    atomic_uint_fast64_t total_worker_calls;
    atomic_uint_fast64_t total_worker_expansions;
    atomic_uint_fast64_t periodic_sync_events;
    atomic_uint_fast64_t periodic_sync_nodes;
    atomic_uint_fast64_t periodic_sync_read_hits;
    atomic_uint_fast64_t periodic_sync_updates;
    atomic_uint_fast64_t concentrated_assignments;

    /* v9.8.2 hybrid SPDFPN + persistent focused deep reroot.  One worker may
       leave the normal job pool to reroot a stalled shared MPN prefix while
       all remaining workers continue SPDFPN.  UNKNOWN reroot state stays in
       a persistent private session; only exact pn==0/dn==0 is injected into
       the shared TT. */
    atomic_uint_fast64_t reroot_triggers;
    atomic_uint_fast64_t reroot_episodes;
    atomic_uint_fast64_t reroot_exact;
    atomic_uint_fast64_t reroot_unknown;
    atomic_uint_fast64_t reroot_sub_calls;
    atomic_uint_fast64_t reroot_sub_expansions;
    atomic_uint_fast64_t reroot_sub_ms;
    atomic_uint_fast64_t reroot_fresh_slices;
    atomic_uint_fast64_t reroot_resume_slices;
    atomic_uint_fast64_t reroot_slices;
    /* v9.8.2 exact-harvesting diagnostics.  trigger = ladder target selection
       request after outer UNKNOWN slices; attempt = one nested persistent
       slice; exact/unknown are nested outcomes. */
    atomic_uint_fast64_t reroot_harvest_triggers;
    atomic_uint_fast64_t reroot_harvest_attempts;
    atomic_uint_fast64_t reroot_harvest_exact;
    atomic_uint_fast64_t reroot_harvest_unknown;
    atomic_uint_fast64_t reroot_harvest_sub_expansions;
    atomic_uint_fast64_t reroot_harvest_sub_ms;
    atomic_uint_fast64_t reroot_harvest_fresh;
    atomic_uint_fast64_t reroot_harvest_resume;
    atomic_uint_fast64_t reroot_harvest_focus_exhausted;
    atomic_uint_fast64_t reroot_harvest_budget_exhausted;
    atomic_uint_fast64_t reroot_harvest_deep_gate_skips;
    atomic_uint_fast64_t reroot_harvest_pressure_skips;
    /* v9.8.2 exact-descendant promotion diagnostics:
       found / inserted into shared exact TT / ancestor pairs changed. */
    atomic_uint_fast64_t reroot_promote_found;
    atomic_uint_fast64_t reroot_promote_inserted;
    atomic_uint_fast64_t reroot_promote_ancestor_changed;
    atomic_uint_fast64_t reroot_promote_conflicts;
    bool reroot_pending;
    bool reroot_active;
    unsigned reroot_owner;
    SPRerootRequest reroot_request;
    SPRerootTrack reroot_track[SP_REROOT_TRACK_SLOTS];
    uint64_t reroot_track_tick;
    SPHotTrack hot_track[SP_HOT_TRACK_SLOTS];
    uint64_t hot_track_tick;
    uint64_t hot_probe_id;
    double hot_next_probe;
    bool hot_diag_valid;
    bool hot_diag_candidate;
    unsigned hot_diag_depth;
    unsigned hot_diag_region_depth;
    unsigned hot_diag_busy;
    double hot_diag_age_sec;
    uint64_t hot_diag_work;
    double hot_diag_delta_pn;
    double hot_diag_delta_dn;
    uint32_t hot_diag_pn, hot_diag_dn;
    Board hot_diag_board;
    bool reroot_root_valid;
    uint32_t reroot_root_pn, reroot_root_dn;
    uint64_t reroot_root_change_exp;

    Board root;
    uint8_t prover;
    unsigned threads;
    uint64_t max_work_per_job;
    unsigned min_job_depth;
    uint64_t publish_work;
    unsigned publish_depth;
    double busy_penalty;
    double deadline;
    double start_time;
    double report_interval;
    double next_report;
    SPActiveJob active[SP_MAX_THREADS];
};

static volatile sig_atomic_t g_interrupted = 0;
static void on_interrupt(int sig) { (void)sig; g_interrupted = 1; }

static double wall_time(void)
{
#ifdef _WIN32
    static LARGE_INTEGER freq;
    static bool initialized = false;
    LARGE_INTEGER now;
    if (!initialized) {
        QueryPerformanceFrequency(&freq);
        initialized = true;
    }
    QueryPerformanceCounter(&now);
    return (double)now.QuadPart / (double)freq.QuadPart;
#else
    struct timespec ts;
    clock_gettime(CLOCK_MONOTONIC, &ts);
    return (double)ts.tv_sec + (double)ts.tv_nsec * 1e-9;
#endif
}

static uint32_t sat_add(uint32_t a, uint32_t b)
{
    if (a >= PN_INF || b >= PN_INF || a > PN_INF - b) return PN_INF;
    return a + b;
}

static uint32_t sat_inc(uint32_t a)
{
    return a >= PN_INF - 1 ? PN_INF : a + 1;
}

static uint32_t threshold_sub_add(uint32_t threshold, uint32_t total, uint32_t part)
{
    if (threshold >= PN_INF) return PN_INF;
    if (total > threshold) return part;
    uint32_t base = threshold - total;
    return sat_add(base, part);
}

static inline uint64_t bit_at(int col, int row)
{
    return UINT64_C(1) << (col * STRIDE + row);
}

static inline unsigned popcount7(uint8_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcount((unsigned)x);
#else
    unsigned n = 0;
    while (x) { n += x & 1u; x >>= 1; }
    return n;
#endif
}

static inline uint8_t move_encode(int tile, int col)
{
    return (uint8_t)((tile << 3) | col);
}

static inline int move_tile(uint8_t m) { return (int)(m >> 3); }
static inline int move_col(uint8_t m)  { return (int)(m & 7u); }

static uint8_t mirror_move(uint8_t m)
{
    if (!m) return 0;
    return move_encode(move_tile(m), 6 - move_col(m));
}

static const char *player_name(int p) { return p ? "Yellow" : "Green"; }

static void board_init(Board *b, uint8_t floor_row)
{
    memset(b, 0, sizeof(*b));
    for (int c = 0; c < COLS; ++c) b->height[c] = floor_row;
}

static inline bool legal_tile_at(const Board *b, int col, int tile)
{
    if (col < 0 || col >= COLS || tile < 1 || tile > 3) return false;
    int row = b->height[col];
    if (row >= MAX_ROWS) return false;
    if (tile == 3 && !(bit_at(col, row) & THREE_MASK)) return false;
    return true;
}

static bool board_drop(Board *b, int col, int tile)
{
    if (!legal_tile_at(b, col, tile)) return false;
    int row = b->height[col];
    uint64_t bit = bit_at(col, row);
    b->p[b->turn] |= bit;
    if (tile == 2) b->twos |= bit;
    else if (tile == 3) b->threes |= bit;
    b->height[col]++;
    b->turn ^= 1u;
    b->ply++;
    return true;
}

static inline int tile_value_at(const Board *b, int col, int row)
{
    uint64_t bit = bit_at(col, row);
    if (b->threes & bit) return 3;
    if (b->twos & bit) return 2;
    return 1;
}

static inline int owner_at(const Board *b, int col, int row)
{
    uint64_t bit = bit_at(col, row);
    if (b->p[0] & bit) return 0;
    if (b->p[1] & bit) return 1;
    return -1;
}

static bool board_full(const Board *b)
{
    for (int c = 0; c < COLS; ++c) if (b->height[c] < MAX_ROWS) return false;
    return true;
}

static unsigned board_empty_cells(const Board *b)
{
    unsigned n = 0;
    for (int c = 0; c < COLS; ++c) n += (unsigned)(MAX_ROWS - b->height[c]);
    return n;
}


/* ---------------- dependency-aware winning-structure engine ----------------
 *
 * A WinSegment is a straight contiguous segment of 3..7 playable cells.  It is
 * "viable" for a player if (i) it contains no opponent stone and (ii) the
 * player's already-fixed tile values plus legal values for all empty cells can
 * still sum to exactly seven.  This deliberately ignores future opponent play
 * and gravity, so viability is an optimistic superset: if NO viable segment
 * remains, that player can never win.  That makes the dead-prover test exact.
 *
 * For ordering, every empty cell in a viable segment is a direct dependency.
 * Empty cells below it in the same column are support dependencies.  These
 * scores are heuristics only; they never delete a legal DF-PN child.
 */
static void segments_init(uint8_t floor_row)
{
    if (g_segment_floor == floor_row && g_segment_count) return;
    g_segment_floor = floor_row;
    g_segment_count = 0;
    memset(g_cell_seg_count, 0, sizeof(g_cell_seg_count));
    static const int dx[4] = {1, 0, 1, 1};
    static const int dy[4] = {0, 1, 1, -1};
    for (int c0 = 0; c0 < COLS; ++c0) {
        for (int r0 = floor_row; r0 < MAX_ROWS; ++r0) {
            for (int d = 0; d < 4; ++d) {
                for (int len = 3; len <= 7; ++len) {
                    int c1 = c0 + dx[d] * (len - 1);
                    int r1 = r0 + dy[d] * (len - 1);
                    if (c1 < 0 || c1 >= COLS || r1 < floor_row || r1 >= MAX_ROWS) continue;
                    if (g_segment_count >= MAX_SEGMENTS) {
                        fprintf(stderr, "Internal error: too many win segments.\n");
                        abort();
                    }
                    WinSegment *sg = &g_segments[g_segment_count++];
                    memset(sg, 0, sizeof(*sg));
                    sg->len = (uint8_t)len;
                    for (int k = 0; k < len; ++k) {
                        int c = c0 + dx[d] * k;
                        int r = r0 + dy[d] * k;
                        uint8_t idx = (uint8_t)(c * STRIDE + r);
                        sg->cell[k] = idx;
                        sg->mask |= UINT64_C(1) << idx;
                        uint8_t nmem = g_cell_seg_count[idx];
                        if (nmem >= 64) { fprintf(stderr, "Internal error: cell segment list overflow.\n"); abort(); }
                        g_cell_segs[idx][nmem] = (uint16_t)g_segment_count;
                        g_cell_seg_count[idx] = (uint8_t)(nmem + 1);
                    }
                }
            }
        }
    }
}

static inline unsigned popcount64(uint64_t x)
{
#if defined(__GNUC__) || defined(__clang__)
    return (unsigned)__builtin_popcountll(x);
#else
    unsigned n = 0; while (x) { x &= x - 1; ++n; } return n;
#endif
}

static bool segment_viable_detail(const Board *b, int player, const WinSegment *sg,
                                   unsigned *empty_out, unsigned *own_sum_out)
{
    uint64_t opp = b->p[player ^ 1] & sg->mask;
    if (opp) return false;
    uint64_t own = b->p[player] & sg->mask;
    unsigned nown = popcount64(own);
    unsigned own_sum = nown + popcount64(own & b->twos) + 2u * popcount64(own & b->threes);
    if (own_sum > 7) return false;
    uint64_t empty = sg->mask & ~(b->p[0] | b->p[1]);
    unsigned empties = popcount64(empty);
    if (empty_out) *empty_out = empties;
    if (own_sum_out) *own_sum_out = own_sum;
    if (empties == 0) return own_sum == 7;

    /* Every empty cell can be 1 or 2; red empties can additionally be 3.
       Hence all totals from empties through 2*empties+red_empties are reachable. */
    unsigned red_empties = popcount64(empty & THREE_MASK);
    unsigned need = 7u - own_sum;
    return need >= empties && need <= 2u * empties + red_empties;
}

static bool player_has_viable_segment(Context *ctx, const Board *b, int player)
{
    segments_init(ctx->floor_row);
    for (unsigned i = 0; i < g_segment_count; ++i) {
        ctx->stats.dep_segments_tested++;
        if (segment_viable_detail(b, player, &g_segments[i], NULL, NULL)) return true;
    }
    return false;
}

static inline uint16_t add_u16_sat(uint16_t a, unsigned b)
{
    unsigned x = (unsigned)a + b;
    return (uint16_t)(x > 65535u ? 65535u : x);
}

static void dependency_map_build(Context *ctx, const Board *b, DependencyMap *dm)
{
    memset(dm, 0, sizeof(*dm));
    segments_init(ctx->floor_row);
    ctx->stats.dep_maps++;

    for (int player = 0; player < 2; ++player) {
        for (unsigned i = 0; i < g_segment_count; ++i) {
            unsigned empties = 0, own_sum = 0;
            ctx->stats.dep_segments_tested++;
            const WinSegment *sg = &g_segments[i];
            if (!segment_viable_detail(b, player, sg, &empties, &own_sum)) continue;
            if (empties == 0) continue; /* already a terminal win; handled elsewhere */
            dm->viable[player]++;
            if (player == 0) ctx->stats.dep_viable_green++;
            else ctx->stats.dep_viable_yellow++;
            if (empties == 1) dm->one_empty[player]++;
            if (empties == 2) dm->two_empty[player]++;

            unsigned base;
            if (empties == 1) base = 64u;
            else if (empties == 2) base = 28u;
            else if (empties == 3) base = 12u;
            else if (empties == 4) base = 5u;
            else base = 2u;
            /* More already-fixed value means a more mature structure. */
            base += own_sum * 2u;

            for (unsigned k = 0; k < sg->len; ++k) {
                int idx = sg->cell[k];
                int c = idx / STRIDE, r = idx % STRIDE;
                if (owner_at(b, c, r) >= 0) continue;
                dm->direct[player][idx] = add_u16_sat(dm->direct[player][idx], base);
                dm->direct_mask[player] |= UINT64_C(1) << idx;
                if (empties <= 2) dm->near_score[player][idx] = add_u16_sat(dm->near_score[player][idx], base * 2u);

                int landing = b->height[c];
                for (int sr = landing; sr < r; ++sr) {
                    int sidx = c * STRIDE + sr;
                    unsigned dist = (unsigned)(r - sr);
                    unsigned sw = base / (dist + 1u);
                    if (sw == 0) sw = 1;
                    dm->support[player][sidx] = add_u16_sat(dm->support[player][sidx], sw);
                    dm->support_mask[player] |= UINT64_C(1) << sidx;
                }
            }
        }
    }
}

static int segment_weight_after_move(const Board *child, int player, uint64_t landing_bit)
{
    if (!landing_bit) return 0;
#if defined(__GNUC__) || defined(__clang__)
    int idx = __builtin_ctzll(landing_bit);
#else
    int idx = 0; while (((landing_bit >> idx) & 1u) == 0) ++idx;
#endif
    int score = 0;
    unsigned nseg = g_cell_seg_count[idx];
    for (unsigned j = 0; j < nseg; ++j) {
        const WinSegment *sg = &g_segments[g_cell_segs[idx][j]];
        unsigned empties = 0, own_sum = 0;
        if (!segment_viable_detail(child, player, sg, &empties, &own_sum)) continue;
        if (empties == 0) score += 500;
        else if (empties == 1) score += 120 + (int)own_sum * 4;
        else if (empties == 2) score += 45 + (int)own_sum * 3;
        else if (empties == 3) score += 16 + (int)own_sum * 2;
        else score += 4 + (int)own_sum;
    }
    return score;
}

/*
 * v6 lightweight dependency score.  Unlike the full DependencyMap this only
 * inspects winning segments touching the landing cell and the next two support
 * cells above it.  It is an ordering heuristic only and never removes a child.
 */
static int local_viable_weight(Context *ctx, const Board *b, int player, int idx)
{
    int score = 0;
    unsigned nseg = g_cell_seg_count[idx];
    for (unsigned j = 0; j < nseg; ++j) {
        const WinSegment *sg = &g_segments[g_cell_segs[idx][j]];
        unsigned empties = 0, own_sum = 0;
        ctx->stats.dep_light_segments_tested++;
        if (!segment_viable_detail(b, player, sg, &empties, &own_sum)) continue;
        if (empties == 0) score += 300;
        else if (empties == 1) score += 96 + (int)own_sum * 5;
        else if (empties == 2) score += 38 + (int)own_sum * 3;
        else if (empties == 3) score += 14 + (int)own_sum * 2;
        else score += 3 + (int)own_sum;
    }
    return score;
}

static int dependency_light_score(Context *ctx, const Board *b, const Board *child,
                                  int mover, int col, int tile)
{
    segments_init(ctx->floor_row);
    int row = b->height[col];
    int idx = col * STRIDE + row;
    int opp = mover ^ 1;
    int score = 0;

    /* Direct attacking value after the move and defensive value before it. */
    score += segment_weight_after_move(child, mover, bit_at(col, row)) * 4;
    score += local_viable_weight(ctx, b, opp, idx) * 3;

    /* Cheap support relevance: a stone can enable the next one/two cells above. */
    for (int d = 1; d <= 2 && row + d < MAX_ROWS; ++d) {
        int up = col * STRIDE + row + d;
        int div = d + 1;
        score += local_viable_weight(ctx, b, mover, up) / div;
        score += local_viable_weight(ctx, b, opp, up) / (div + 1);
    }

    score += (tile == 2 ? 12 : tile == 3 ? 8 : 3);
    score += (6 - CENTER_RANK[col]);
    if (score > 32767) score = 32767;
    return score;
}

static int dependency_move_score(const Board *b, const Board *child, int mover,
                                 int col, int tile, const DependencyMap *dm)
{
    (void)tile;
    int row = b->height[col];
    int idx = col * STRIDE + row;
    int opp = mover ^ 1;
    int score = 0;

    score += (int)dm->direct[mover][idx] * 5;
    score += (int)dm->near_score[mover][idx] * 4;
    score += (int)dm->support[mover][idx] * 3;
    /* Occupying an opponent's direct target permanently destroys every
       opponent segment that requires this exact cell. */
    score += (int)dm->direct[opp][idx] * 4;
    score += (int)dm->near_score[opp][idx] * 3;

    uint64_t landing_bit = bit_at(col, row);
    score += segment_weight_after_move(child, mover, landing_bit);

    /* Stable secondary preference: unrestricted 2, then red-cell 3, then 1. */
    score += (tile == 2 ? 10 : tile == 3 ? 6 : 2);
    score += (6 - CENTER_RANK[col]);
    if (score > 32767) score = 32767;
    return score;
}

static uint64_t mirror_bits(uint64_t x)
{
    uint64_t r = 0;
    for (int c = 0; c < COLS; ++c) {
        uint64_t col = (x >> (c * STRIDE)) & UINT64_C(0x7f);
        r |= col << ((6 - c) * STRIDE);
    }
    return r;
}

static bool board_symmetric(const Board *b)
{
    return b->p[0] == mirror_bits(b->p[0]) &&
           b->p[1] == mirror_bits(b->p[1]) &&
           b->twos == mirror_bits(b->twos) &&
           b->threes == mirror_bits(b->threes);
}

static int tuple_cmp4(const uint64_t a[4], const uint64_t b[4])
{
    for (int i = 0; i < 4; ++i) {
        if (a[i] < b[i]) return -1;
        if (a[i] > b[i]) return 1;
    }
    return 0;
}

static void canonical_key(const Board *b, uint64_t key[4], bool *mirrored)
{
    uint64_t a[4] = {b->p[0], b->p[1], b->twos, b->threes};
    uint64_t m[4] = {mirror_bits(b->p[0]), mirror_bits(b->p[1]),
                     mirror_bits(b->twos), mirror_bits(b->threes)};
    a[0] |= (uint64_t)b->turn << 63;
    m[0] |= (uint64_t)b->turn << 63;
    if (tuple_cmp4(m, a) < 0) {
        memcpy(key, m, sizeof(m));
        *mirrored = true;
    } else {
        memcpy(key, a, sizeof(a));
        *mirrored = false;
    }
}

static uint64_t mix64(uint64_t x)
{
    x ^= x >> 30;
    x *= UINT64_C(0xbf58476d1ce4e5b9);
    x ^= x >> 27;
    x *= UINT64_C(0x94d049bb133111eb);
    x ^= x >> 31;
    return x;
}

static uint64_t hash_key4(const uint64_t k[4])
{
    uint64_t h = UINT64_C(0x9e3779b97f4a7c15);
    h ^= mix64(k[0] + UINT64_C(0x243f6a8885a308d3));
    h = (h << 17) | (h >> 47);
    h ^= mix64(k[1] + UINT64_C(0x13198a2e03707344));
    h = (h << 23) | (h >> 41);
    h ^= mix64(k[2] + UINT64_C(0xa4093822299f31d0));
    h = (h << 29) | (h >> 35);
    h ^= mix64(k[3] + UINT64_C(0x082efa98ec4e6c89));
    return mix64(h);
}


static inline bool exact_db_lookup_pair(const ExactDB *db, const Board *b, uint8_t prover,
                                        uint32_t *pn, uint32_t *dn);

static inline void sp_bucket_lock(SPSharedBucket *b)
{
    while (atomic_flag_test_and_set_explicit(&b->lock, memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        __builtin_ia32_pause();
#endif
    }
}

static inline void sp_bucket_unlock(SPSharedBucket *b)
{
    atomic_flag_clear_explicit(&b->lock, memory_order_release);
}

static inline void sp_job_lock(ParallelShared *sp)
{
    while (atomic_flag_test_and_set_explicit(&sp->job_lock, memory_order_acquire)) {
#if defined(__x86_64__) || defined(__i386__) || defined(_M_X64) || defined(_M_IX86)
        __builtin_ia32_pause();
#endif
    }
}

static inline void sp_job_unlock(ParallelShared *sp)
{
    atomic_flag_clear_explicit(&sp->job_lock, memory_order_release);
}

static void sp_thread_yield(void)
{
#ifdef _WIN32
    SwitchToThread();
#else
    struct timespec ts = {0, 1000000};
    nanosleep(&ts, NULL);
#endif
}

static bool sp_shared_tt_init(SPSharedTT *tt, double requested_gb)
{
    memset(tt, 0, sizeof(*tt));
    if (requested_gb <= 0.0) return false;
    long double bytes_ld = (long double)requested_gb * 1024.0L * 1024.0L * 1024.0L;
    size_t sets = 1024;
    while (sets <= SIZE_MAX / 2) {
        size_t nxt = sets << 1;
        long double need = (long double)nxt * (long double)sizeof(SPSharedBucket);
        if (need > bytes_ld) break;
        sets = nxt;
    }
    tt->bucket = (SPSharedBucket *)calloc(sets, sizeof(SPSharedBucket));
    if (!tt->bucket) return false;
    for (size_t i = 0; i < sets; ++i) atomic_flag_clear(&tt->bucket[i].lock);
    tt->sets = sets;
    tt->mask = sets - 1;
    tt->allocated_gb = ((double)sets * (double)sizeof(SPSharedBucket)) /
                       (1024.0 * 1024.0 * 1024.0);
    tt->generation = 1;
    return true;
}

static void sp_shared_tt_destroy(SPSharedTT *tt)
{
    free(tt->bucket);
    memset(tt, 0, sizeof(*tt));
}

static inline bool sp_key_equal(const SPSharedEntry *e, const uint64_t k[4], uint32_t generation)
{
    return e->generation == generation && e->k0 == k[0] && e->k1 == k[1] &&
           e->k2 == k[2] && e->k3 == k[3];
}

static inline uint64_t sp_pair_quality(uint32_t pn, uint32_t dn)
{
    if (pn == 0 || dn == 0) return UINT64_MAX;
    return (uint64_t)pn + (uint64_t)dn;
}

static bool sp_shared_read(ParallelShared *sp, const Board *b,
                           uint32_t *pn, uint32_t *dn, uint64_t *work)
{
    if (!sp) return false;
    if (sp->exact_db && exact_db_lookup_pair(sp->exact_db, b, sp->prover, pn, dn)) {
        if (work) *work = 0;
        atomic_fetch_add_explicit(&sp->exact_db_hits, 1, memory_order_relaxed);
        return true;
    }
    if (!sp->tt.bucket) return false;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    atomic_fetch_add_explicit(&sp->shared_probes, 1, memory_order_relaxed);
    SPSharedBucket *bucket = &sp->tt.bucket[(size_t)(hash_key4(k) & sp->tt.mask)];
    bool found = false;
    sp_bucket_lock(bucket);
    for (int w = 0; w < 2; ++w) {
        SPSharedEntry *e = &bucket->way[w];
        if (sp_key_equal(e, k, sp->tt.generation)) {
            *pn = e->pn; *dn = e->dn;
            if (work) *work = e->work;
            found = true;
            break;
        }
    }
    sp_bucket_unlock(bucket);
    if (found) atomic_fetch_add_explicit(&sp->shared_hits, 1, memory_order_relaxed);
    return found;
}

static bool sp_shared_read_meta(ParallelShared *sp, const Board *b,
                                uint32_t *pn, uint32_t *dn, uint64_t *work,
                                uint64_t *pair_change_work)
{
    if (!sp) return false;
    uint32_t xpn=1, xdn=1;
    if (sp->exact_db && exact_db_lookup_pair(sp->exact_db, b, sp->prover, &xpn, &xdn)) {
        if (pn) *pn=xpn;
        if (dn) *dn=xdn;
        if (work) *work=0;
        if (pair_change_work) *pair_change_work=0;
        atomic_fetch_add_explicit(&sp->exact_db_hits, 1, memory_order_relaxed);
        return true;
    }
    if (!sp->tt.bucket) return false;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    atomic_fetch_add_explicit(&sp->shared_probes, 1, memory_order_relaxed);
    SPSharedBucket *bucket = &sp->tt.bucket[(size_t)(hash_key4(k) & sp->tt.mask)];
    bool found = false;
    sp_bucket_lock(bucket);
    for (int w = 0; w < 2; ++w) {
        SPSharedEntry *e = &bucket->way[w];
        if (sp_key_equal(e, k, sp->tt.generation)) {
            if (pn) *pn = e->pn;
            if (dn) *dn = e->dn;
            if (work) *work = e->work;
            if (pair_change_work) *pair_change_work = e->pair_change_work;
            found = true;
            break;
        }
    }
    sp_bucket_unlock(bucket);
    if (found) atomic_fetch_add_explicit(&sp->shared_hits, 1, memory_order_relaxed);
    return found;
}

static void sp_shared_publish(ParallelShared *sp, const Board *b,
                              uint32_t pn, uint32_t dn)
{
    if (!sp || !sp->tt.bucket) return;
    uint32_t dbpn=1, dbdn=1;
    if (sp->exact_db && exact_db_lookup_pair(sp->exact_db, b, sp->prover, &dbpn, &dbdn)) {
        if ((pn==0 || dn==0) && (pn!=dbpn || dn!=dbdn))
            atomic_fetch_add_explicit(&sp->exact_db_conflicts, 1, memory_order_relaxed);
        pn=dbpn; dn=dbdn;
    }
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    SPSharedBucket *bucket = &sp->tt.bucket[(size_t)(hash_key4(k) & sp->tt.mask)];
    sp_bucket_lock(bucket);
    SPSharedEntry *e = NULL;
    for (int w = 0; w < 2; ++w) {
        if (sp_key_equal(&bucket->way[w], k, sp->tt.generation)) { e = &bucket->way[w]; break; }
    }
    if (!e) {
        int victim = -1;
        for (int w = 0; w < 2; ++w) {
            if (bucket->way[w].generation != sp->tt.generation) { victim = w; break; }
        }
        if (victim < 0) {
            bool ex0 = bucket->way[0].pn == 0 || bucket->way[0].dn == 0;
            bool ex1 = bucket->way[1].pn == 0 || bucket->way[1].dn == 0;
            if (ex0 != ex1) victim = ex0 ? 1 : 0;
            else victim = sp_pair_quality(bucket->way[0].pn, bucket->way[0].dn) <=
                          sp_pair_quality(bucket->way[1].pn, bucket->way[1].dn) ? 0 : 1;
        }
        e = &bucket->way[victim];
        e->k0=k[0]; e->k1=k[1]; e->k2=k[2]; e->k3=k[3];
        e->generation = sp->tt.generation;
        e->pn = pn; e->dn = dn; e->work = 0; e->pair_change_work = 0;
    } else {
        bool old_exact = e->pn == 0 || e->dn == 0;
        bool new_exact = pn == 0 || dn == 0;
        if (!old_exact && (new_exact || sp_pair_quality(pn,dn) >= sp_pair_quality(e->pn,e->dn))) {
            if (e->pn != pn || e->dn != dn) e->pair_change_work = e->work;
            e->pn = pn; e->dn = dn;
        }
    }
    sp_bucket_unlock(bucket);
    atomic_fetch_add_explicit(&sp->shared_publishes, 1, memory_order_relaxed);
}

static void sp_shared_add_work(ParallelShared *sp, const Board *b, uint64_t delta)
{
    if (!sp || !sp->tt.bucket || delta == 0) return;
    uint32_t dbpn=1, dbdn=1;
    if (sp->exact_db && exact_db_lookup_pair(sp->exact_db, b, sp->prover, &dbpn, &dbdn)) return;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    SPSharedBucket *bucket = &sp->tt.bucket[(size_t)(hash_key4(k) & sp->tt.mask)];
    sp_bucket_lock(bucket);
    SPSharedEntry *e = NULL;
    for (int w=0; w<2; ++w) if (sp_key_equal(&bucket->way[w],k,sp->tt.generation)) { e=&bucket->way[w]; break; }
    if (!e) {
        int victim = bucket->way[0].generation != sp->tt.generation ? 0 :
                     bucket->way[1].generation != sp->tt.generation ? 1 :
                     (sp_pair_quality(bucket->way[0].pn,bucket->way[0].dn) <=
                      sp_pair_quality(bucket->way[1].pn,bucket->way[1].dn) ? 0 : 1);
        e=&bucket->way[victim];
        e->k0=k[0]; e->k1=k[1]; e->k2=k[2]; e->k3=k[3];
        e->generation=sp->tt.generation; e->pn=1; e->dn=1; e->work=0; e->pair_change_work=0;
    }
    if (UINT64_MAX - e->work < delta) e->work = UINT64_MAX;
    else e->work += delta;
    sp_bucket_unlock(bucket);
}

static void sp_context_publish(Context *ctx, const Board *b, uint32_t pn, uint32_t dn)
{
    if (!ctx || !ctx->parallel_shared) return;
    /* Worker-side finite PN/DN is intentionally published only at bounded-job
       boundaries.  Publishing every interior recomputation turns the shared TT
       into a lock-contention hotspot at 8-16 cores.  Exact zero values remain
       immediately shared; finite job-end values and scheduler ancestor
       recomputations are published explicitly by the parallel engine. */
    if (ctx->parallel_worker && pn != 0 && dn != 0) return;
    sp_shared_publish(ctx->parallel_shared, b, pn, dn);
}

static bool reroot_overlay_lookup(const Context *ctx, const Board *b, uint32_t *pn, uint32_t *dn)
{
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    size_t idx = (size_t)(hash_key4(k) & (REROOT_OVERLAY_SIZE - 1u));
    const RerootProof *rp = &ctx->reroot_overlay[idx];
    if (!rp->valid || rp->prover != ctx->prover) return false;
    if (rp->k[0] != k[0] || rp->k[1] != k[1] || rp->k[2] != k[2] || rp->k[3] != k[3]) return false;
    *pn = rp->pn; *dn = rp->dn;
    return true;
}

static void reroot_overlay_store(Context *ctx, const Board *b, uint32_t pn, uint32_t dn)
{
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    size_t idx = (size_t)(hash_key4(k) & (REROOT_OVERLAY_SIZE - 1u));
    RerootProof *rp = &ctx->reroot_overlay[idx];
    memset(rp, 0, sizeof(*rp));
    rp->valid = true;
    rp->prover = ctx->prover;
    memcpy(rp->k, k, sizeof(k));
    rp->pn = pn;
    rp->dn = dn;
}

static bool key_equal(const TTEntry *e, const uint64_t k[4], uint32_t generation)
{
    return e->generation == generation && e->k0 == k[0] && e->k1 == k[1] &&
           e->k2 == k[2] && e->k3 == k[3];
}

static bool tt_init(TT *tt, double requested_gb)
{
    memset(tt, 0, sizeof(*tt));
    if (requested_gb < 0.05) requested_gb = 0.05;
    long double bytes_ld = (long double)requested_gb * 1073741824.0L;
    size_t max_sets = (size_t)(bytes_ld / (long double)sizeof(TTBucket));
    if (max_sets < 1024) max_sets = 1024;
    size_t sets = 1;
    while (sets <= max_sets / 2) sets <<= 1;
    while (sets >= 1024) {
        TTBucket *p = (TTBucket *)calloc(sets, sizeof(TTBucket));
        if (p) {
            tt->bucket = p;
            tt->sets = sets;
            tt->mask = sets - 1;
            tt->allocated_gb = (double)((long double)sets * sizeof(TTBucket) / 1073741824.0L);
            return true;
        }
        sets >>= 1;
    }
    return false;
}

static void tt_destroy(TT *tt)
{
    free(tt->bucket);
    memset(tt, 0, sizeof(*tt));
}

static bool child_cache_init(ChildCache *cc, double requested_mb)
{
    memset(cc, 0, sizeof(*cc));
    if (requested_mb <= 0.0) return true;
    long double bytes_ld = (long double)requested_mb * 1048576.0L;
    size_t max_sets = (size_t)(bytes_ld / (long double)sizeof(ChildCacheEntry));
    if (max_sets < 1024) max_sets = 1024;
    size_t sets = 1;
    while (sets <= max_sets / 2) sets <<= 1;
    while (sets >= 1024) {
        ChildCacheEntry *p = (ChildCacheEntry *)calloc(sets, sizeof(ChildCacheEntry));
        if (p) {
            cc->entry = p;
            cc->sets = sets;
            cc->mask = sets - 1;
            cc->allocated_mb = (double)((long double)sets * sizeof(ChildCacheEntry) / 1048576.0L);
            return true;
        }
        sets >>= 1;
    }
    return false;
}

static void child_cache_destroy(ChildCache *cc)
{
    free(cc->entry);
    memset(cc, 0, sizeof(*cc));
}

static bool hot_tt_init(HotExactTT *ht, double requested_mb)
{
    memset(ht, 0, sizeof(*ht));
    if (requested_mb <= 0.0) return true;
    long double bytes_ld = (long double)requested_mb * 1048576.0L;
    size_t max_sets = (size_t)(bytes_ld / (long double)sizeof(HotExactEntry));
    if (max_sets < 1024) max_sets = 1024;
    size_t sets = 1;
    while (sets <= max_sets / 2) sets <<= 1;
    while (sets >= 1024) {
        HotExactEntry *e = (HotExactEntry *)calloc(sets, sizeof(HotExactEntry));
        if (e) {
            ht->entry = e; ht->sets = sets; ht->mask = sets - 1;
            ht->allocated_mb = (double)((long double)sets * sizeof(HotExactEntry) / 1048576.0L);
            return true;
        }
        sets >>= 1;
    }
    return false;
}

static void hot_tt_destroy(HotExactTT *ht)
{
    free(ht->entry);
    memset(ht, 0, sizeof(*ht));
}

static bool hot_tt_lookup(Context *ctx, const Board *b, int *result)
{
    if (!ctx->hot_tt.entry) return false;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    HotExactEntry *e = &ctx->hot_tt.entry[(size_t)(hash_key4(k) & ctx->hot_tt.mask)];
    if (e->valid && e->k0 == k[0] && e->k1 == k[1] && e->k2 == k[2] && e->k3 == k[3]) {
        ctx->stats.hot_exact_tt_hits++;
        *result = e->result;
        return true;
    }
    return false;
}

static void hot_tt_store(Context *ctx, const Board *b, int result)
{
    if (!ctx->hot_tt.entry || result < 0 || result > 2) return;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored); (void)mirrored;
    HotExactEntry *e = &ctx->hot_tt.entry[(size_t)(hash_key4(k) & ctx->hot_tt.mask)];
    e->k0 = k[0]; e->k1 = k[1]; e->k2 = k[2]; e->k3 = k[3];
    e->result = (int8_t)result; e->valid = 1;
}

static ChildCacheEntry *child_cache_lookup(Context *ctx, const Board *b, bool *mirrored)
{
    if (!ctx->child_cache_enabled || !ctx->child_cache.entry) return NULL;
    uint64_t k[4];
    canonical_key(b, k, mirrored);
    ctx->stats.child_cache_probes++;
    ChildCacheEntry *e = &ctx->child_cache.entry[(size_t)(hash_key4(k) & ctx->child_cache.mask)];
    if (e->generation == ctx->generation && e->k0 == k[0] && e->k1 == k[1] &&
        e->k2 == k[2] && e->k3 == k[3]) {
        ctx->stats.child_cache_hits++;
        return e;
    }
    return NULL;
}

static void child_cache_store(Context *ctx, const Board *b, int status,
                              const Child children[MAX_MOVES], int n, int quiet_start)
{
    if (!ctx->child_cache_enabled || !ctx->child_cache.entry) return;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored);
    ChildCacheEntry *e = &ctx->child_cache.entry[(size_t)(hash_key4(k) & ctx->child_cache.mask)];
    e->k0 = k[0]; e->k1 = k[1]; e->k2 = k[2]; e->k3 = k[3];
    e->generation = ctx->generation;
    e->status = (int8_t)status;
    e->count = (uint8_t)(n < 0 ? 0 : n);
    e->quiet_start = (uint8_t)(quiet_start < 0 ? 0 : quiet_start);
    for (int i = 0; i < n; ++i) {
        e->move[i] = mirrored ? mirror_move(children[i].move) : children[i].move;
        e->terminal[i] = children[i].terminal_winner;
    }
    ctx->stats.child_cache_stores++;
}

static void promote_move_stable(Child children[MAX_MOVES], int n, uint8_t move)
{
    if (!move || n <= 1) return;
    int at = -1;
    for (int i = 0; i < n; ++i) if (children[i].move == move) { at = i; break; }
    if (at <= 0) return;
    Child x = children[at];
    memmove(&children[1], &children[0], (size_t)at * sizeof(children[0]));
    children[0] = x;
}

static void promote_dep_hints_in_quiet(Child children[MAX_MOVES], int n, int quiet_start,
                                       const uint8_t hints[4])
{
    if (quiet_start < 0) quiet_start = 0;
    if (quiet_start >= n) return;
    int insert = quiet_start;
    for (int r = 0; r < 4 && insert < n; ++r) {
        if (!hints[r]) continue;
        int at = -1;
        for (int i = insert; i < n; ++i) if (children[i].move == hints[r]) { at = i; break; }
        if (at < 0) continue;
        Child x = children[at];
        memmove(&children[insert + 1], &children[insert], (size_t)(at - insert) * sizeof(children[0]));
        children[insert++] = x;
    }
}

static TTEntry *tt_access(Context *ctx, const Board *b, bool create, bool *was_hit, bool *mirrored)
{
    uint64_t k[4];
    canonical_key(b, k, mirrored);
    size_t idx = (size_t)(hash_key4(k) & ctx->tt.mask);
    TTBucket *bucket = &ctx->tt.bucket[idx];
    ctx->stats.tt_probes++;
    for (int w = 0; w < 2; ++w) {
        if (key_equal(&bucket->way[w], k, ctx->generation)) {
            ctx->stats.tt_hits++;
            *was_hit = true;
            return &bucket->way[w];
        }
    }
    *was_hit = false;
    if (!create) return NULL;

    int victim = -1;
    for (int w = 0; w < 2; ++w) {
        if (bucket->way[w].generation != ctx->generation) { victim = w; break; }
    }
    if (victim < 0) {
        /* Prefer replacing an unresolved entry with the larger work estimate. */
        uint64_t s0 = (uint64_t)bucket->way[0].pn + bucket->way[0].dn;
        uint64_t s1 = (uint64_t)bucket->way[1].pn + bucket->way[1].dn;
        victim = (s0 >= s1) ? 0 : 1;
        if ((bucket->way[0].pn == 0 || bucket->way[0].dn == 0) &&
            !(bucket->way[1].pn == 0 || bucket->way[1].dn == 0)) victim = 1;
        else if ((bucket->way[1].pn == 0 || bucket->way[1].dn == 0) &&
                 !(bucket->way[0].pn == 0 || bucket->way[0].dn == 0)) victim = 0;
        ctx->stats.tt_replacements++;
    }
    TTEntry *e = &bucket->way[victim];
    e->k0 = k[0]; e->k1 = k[1]; e->k2 = k[2]; e->k3 = k[3];
    e->pn = 1; e->dn = 1;
    e->generation = ctx->generation;
    e->examined_mask = 0;
    e->best_move = 0;
    memset(e->dep_hint_move, 0, sizeof(e->dep_hint_move));
    e->flags = 0;
    e->tss_level = 0;
    e->bridge_level = 0;
    return e;
}

static void tt_pair(Context *ctx, const Board *b, uint32_t *pn, uint32_t *dn)
{
    if (ctx->exact_db && exact_db_lookup_pair(ctx->exact_db, b, ctx->prover, pn, dn)) {
        sp_context_publish(ctx, b, *pn, *dn);
        return;
    }
    if (reroot_overlay_lookup(ctx, b, pn, dn)) {
        ctx->stats.deep_reroot_overlay_hits++;
        sp_context_publish(ctx, b, *pn, *dn);
        return;
    }
    bool hit, mirrored;
    TTEntry *e = tt_access(ctx, b, false, &hit, &mirrored);
    (void)mirrored;
    uint32_t lpn = 1, ldn = 1;
    if (hit && e) { lpn = e->pn; ldn = e->dn; }

    if (ctx->parallel_shared) {
        if (lpn == 0 || ldn == 0) {
            sp_context_publish(ctx, b, lpn, ldn);
            *pn=lpn; *dn=ldn; return;
        }
        /* A private TT hit is deliberately trusted for the duration of one
           bounded job.  Shared finite state is sampled on local misses and at
           job boundaries; this preserves exact sharing while avoiding a locked
           shared probe for every child lookup. */
        if (!hit) {
            uint32_t spn=1, sdn=1; uint64_t work=0;
            bool shit = sp_shared_read(ctx->parallel_shared, b, &spn, &sdn, &work);
            (void)work;
            if (shit) { *pn=spn; *dn=sdn; return; }
        }
    }
    *pn=lpn; *dn=ldn;
}

/* Read-only TT lookup used by v8 diagnostics.  It deliberately does not
   modify TT statistics, replacement state, or search ordering. */
static bool tt_pair_peek(const Context *ctx, const Board *b, uint32_t *pn, uint32_t *dn)
{
    if (reroot_overlay_lookup(ctx, b, pn, dn)) return true;
    if (!ctx->tt.bucket || !ctx->tt.sets) { *pn = 1; *dn = 1; return false; }
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored);
    (void)mirrored;
    const TTBucket *bucket = &ctx->tt.bucket[(size_t)(hash_key4(k) & ctx->tt.mask)];
    for (int w = 0; w < 2; ++w) {
        const TTEntry *e = &bucket->way[w];
        if (key_equal(e, k, ctx->generation)) { *pn = e->pn; *dn = e->dn; return true; }
    }
    *pn = 1; *dn = 1;
    return false;
}

/* v9.8.2 read-only counterpart of tt_best_move().  Exact harvesting uses it
   to reconstruct the reroot session's CURRENT most-proving chain without
   perturbing TT statistics or allocating/replacing an entry. */
static bool tt_best_move_peek(const Context *ctx, const Board *b, uint8_t *move_out)
{
    if (move_out) *move_out = 0;
    if (!ctx || !b || !ctx->tt.bucket || !ctx->tt.sets) return false;
    uint64_t k[4]; bool mirrored;
    canonical_key(b, k, &mirrored);
    const TTBucket *bucket = &ctx->tt.bucket[(size_t)(hash_key4(k) & ctx->tt.mask)];
    for (int w = 0; w < 2; ++w) {
        const TTEntry *e = &bucket->way[w];
        if (!key_equal(e, k, ctx->generation) || !e->best_move) continue;
        if (move_out) *move_out = mirrored ? mirror_move(e->best_move) : e->best_move;
        return true;
    }
    return false;
}


/* v9.8.2 bidirectional hot-path synchronization.  Workers keep their rich
   private TT for low-contention search, but periodically import a more
   advanced shared PN/DN estimate and republish the currently active path.
   Exact zero values remain the only game-theoretic conclusions; finite values
   only steer subsequent DF-PN work. */
static int sp_worker_import_shared(Context *ctx, const Board *b)
{
    if (!ctx || !ctx->parallel_worker || !ctx->parallel_shared) return 0;
    uint32_t spn=1, sdn=1; uint64_t work=0;
    if (!sp_shared_read(ctx->parallel_shared,b,&spn,&sdn,&work)) return 0;
    (void)work;
    bool hit=false, mirrored=false;
    TTEntry *e=tt_access(ctx,b,true,&hit,&mirrored);
    (void)hit; (void)mirrored;
    bool local_exact=(e->pn==0 || e->dn==0);
    bool shared_exact=(spn==0 || sdn==0);
    if (!local_exact && (shared_exact || sp_pair_quality(spn,sdn) > sp_pair_quality(e->pn,e->dn))) {
        e->pn=spn; e->dn=sdn;
        return 2; /* shared hit + local update */
    }
    return 1;     /* shared hit, local value already at least as advanced */
}

static void sp_periodic_hot_sync(Context *ctx, unsigned depth)
{
    if (!ctx || !ctx->parallel_worker || !ctx->parallel_shared ||
        ctx->parallel_publish_interval==0) return;
    uint64_t calls=ctx->stats.dfpn_calls;
    if (calls < ctx->parallel_next_publish_calls) return;
    ctx->parallel_next_publish_calls = calls + ctx->parallel_publish_interval;

    unsigned hi=depth < SP_PATH_MAX ? depth : SP_PATH_MAX-1u;
    unsigned width=ctx->parallel_publish_depth ? ctx->parallel_publish_depth : 1u;
    unsigned lo=(hi+1u>width)?(hi+1u-width):0u;
    if (lo==0u && depth>0u) lo=1u; /* worker job root starts at depth 1 */
    uint64_t synced=0;
    for (unsigned d=lo; d<=hi; ++d) {
        Board *pb=&ctx->parallel_path[d];
        int imported=sp_worker_import_shared(ctx,pb);
        if (imported) atomic_fetch_add_explicit(&ctx->parallel_shared->periodic_sync_read_hits,1,memory_order_relaxed);
        if (imported==2) atomic_fetch_add_explicit(&ctx->parallel_shared->periodic_sync_updates,1,memory_order_relaxed);
        uint32_t pn=1,dn=1;
        if (tt_pair_peek(ctx,pb,&pn,&dn)) {
            sp_shared_publish(ctx->parallel_shared,pb,pn,dn);
            synced++;
        }
    }
    atomic_fetch_add_explicit(&ctx->parallel_shared->periodic_sync_events,1,memory_order_relaxed);
    atomic_fetch_add_explicit(&ctx->parallel_shared->periodic_sync_nodes,synced,memory_order_relaxed);
}

static bool same_canonical_state(const Board *a, const Board *b)
{
    uint64_t ka[4], kb[4]; bool ma, mb;
    canonical_key(a, ka, &ma); canonical_key(b, kb, &mb);
    (void)ma; (void)mb;
    return ka[0] == kb[0] && ka[1] == kb[1] && ka[2] == kb[2] && ka[3] == kb[3];
}

static uint8_t tt_best_move(Context *ctx, TTEntry *e, bool mirrored)
{
    (void)ctx;
    return mirrored ? mirror_move(e->best_move) : e->best_move;
}

static void tt_set_best(TTEntry *e, bool mirrored, uint8_t move)
{
    e->best_move = mirrored ? mirror_move(move) : move;
}

static uint8_t tt_dep_hint_at(TTEntry *e, bool mirrored, unsigned rank)
{
    if (rank >= 4) return 0;
    uint8_t m = e->dep_hint_move[rank];
    return mirrored ? mirror_move(m) : m;
}

static void tt_set_dep_hints(TTEntry *e, bool mirrored, const uint8_t moves[4], bool full)
{
    for (unsigned i = 0; i < 4; ++i)
        e->dep_hint_move[i] = (mirrored && moves[i]) ? mirror_move(moves[i]) : moves[i];
    e->flags |= TT_FLAG_DEP_HINT_READY;
    if (full) e->flags |= TT_FLAG_DEP_FULL_HINT_READY;
}

static inline uint8_t tss_proven_flag(uint8_t attacker)
{
    return attacker ? TT_FLAG_TSS_PROVEN_YELLOW : TT_FLAG_TSS_PROVEN_GREEN;
}

/*
 * Reuse only proofs attached to states already resident in the main TT.
 * We deliberately do not allocate TT entries from inside TSS: this avoids
 * letting the selective oracle evict exact DF-PN work.
 */
static bool tss_cached_proof(Context *ctx, const Board *b, uint8_t attacker)
{
    bool hit, mirrored;
    TTEntry *e = tt_access(ctx, b, false, &hit, &mirrored);
    (void)mirrored;
    if (hit && e && (e->flags & tss_proven_flag(attacker))) {
        ctx->stats.tss_cache_hits++;
        return true;
    }
    return false;
}

static void tss_mark_proven(Context *ctx, const Board *b, uint8_t attacker)
{
    bool hit, mirrored;
    TTEntry *e = tt_access(ctx, b, false, &hit, &mirrored);
    (void)mirrored;
    if (hit && e) {
        e->flags |= tss_proven_flag(attacker);
        /* A successful TSS proof is an exact forced-win proof.  Push it
           directly into the main DF-PN numbers so sibling aggregation can
           consume it without selecting this child again. */
        if (attacker == ctx->prover) { e->pn = 0; e->dn = PN_INF; }
        else { e->pn = PN_INF; e->dn = 0; }
    }
}

static WinClass classify_counts(int ones, int twos, int threes)
{
    if (ones == 0 && twos == 2 && threes == 1) return WIN_322;
    if (ones == 1 && twos == 3 && threes == 0) return WIN_2221;
    return WIN_OTHER;
}

/* Exact local sliding-window test for a hypothetical move by player. */
static WinClass winning_move_class(const Board *b, int player, int col, int tile)
{
    if (!legal_tile_at(b, col, tile)) return WIN_NONE;
    const int row = b->height[col];
    static const int dx[4] = {1, 0, 1, 1};
    static const int dy[4] = {0, 1, 1, -1};
    WinClass best = WIN_NONE;

    for (int d = 0; d < 4; ++d) {
        int vals[13];
        int len = 0, center = 0;
        int left_vals[6], nl = 0;

        for (int step = 1; step <= 6; ++step) {
            int c = col - dx[d] * step;
            int r = row - dy[d] * step;
            if (c < 0 || c >= COLS || r < 0 || r >= MAX_ROWS) break;
            if (owner_at(b, c, r) != player) break;
            left_vals[nl++] = tile_value_at(b, c, r);
        }
        for (int i = nl - 1; i >= 0; --i) vals[len++] = left_vals[i];
        center = len;
        vals[len++] = tile;

        for (int step = 1; step <= 6; ++step) {
            int c = col + dx[d] * step;
            int r = row + dy[d] * step;
            if (c < 0 || c >= COLS || r < 0 || r >= MAX_ROWS) break;
            if (owner_at(b, c, r) != player) break;
            vals[len++] = tile_value_at(b, c, r);
        }

        for (int l = 0; l <= center; ++l) {
            int sum = 0, ones = 0, twos = 0, threes = 0;
            for (int r = l; r < len; ++r) {
                int v = vals[r];
                sum += v;
                ones += (v == 1);
                twos += (v == 2);
                threes += (v == 3);
                if (sum > 7) break;
                if (r >= center && l <= center && sum == 7) {
                    WinClass cl = classify_counts(ones, twos, threes);
                    if (cl > best) best = cl;
                }
            }
        }
    }
    return best;
}

/* Slow, independent all-lines test used for parsing/self-test only. */
static bool has_win_for_player(const Board *b, int player)
{
    static const int dx[4] = {1, 0, 1, 1};
    static const int dy[4] = {0, 1, 1, -1};
    for (int c0 = 0; c0 < COLS; ++c0) {
        for (int r0 = 0; r0 < MAX_ROWS; ++r0) {
            if (owner_at(b, c0, r0) != player) continue;
            for (int d = 0; d < 4; ++d) {
                int sum = 0;
                for (int step = 0; step < 7; ++step) {
                    int c = c0 + dx[d] * step;
                    int r = r0 + dy[d] * step;
                    if (c < 0 || c >= COLS || r < 0 || r >= MAX_ROWS) break;
                    if (owner_at(b, c, r) != player) break;
                    sum += tile_value_at(b, c, r);
                    if (sum == 7) return true;
                    if (sum > 7) break;
                }
            }
        }
    }
    return false;
}

static ThreatInfo immediate_info(const Board *b, int player)
{
    ThreatInfo t = {0, WIN_NONE, 0};
    for (int oi = 0; oi < COLS; ++oi) {
        int c = CENTER_ORDER[oi];
        int row = b->height[c];
        if (row >= MAX_ROWS) continue;
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, c, tile)) continue;
            WinClass cl = winning_move_class(b, player, c, tile);
            if (cl != WIN_NONE) {
                t.colmask |= (uint8_t)(1u << c);
                if (cl > t.best_class) {
                    t.best_class = (uint8_t)cl;
                    t.best_tile = (uint8_t)tile;
                }
            }
        }
    }
    return t;
}

static void board_print(const Board *b, uint8_t floor_row)
{
    printf("\n      A  B  C  D  E  F  G\n");
    for (int r = MAX_ROWS - 1; r >= floor_row; --r) {
        printf("%2d   ", r - floor_row + 1);
        for (int c = 0; c < COLS; ++c) {
            int owner = owner_at(b, c, r);
            if (owner >= 0) {
                printf("%c%d ", owner ? 'Y' : 'G', tile_value_at(b, c, r));
            } else {
                printf(" %c ", (bit_at(c, r) & THREE_MASK) ? '=' : '-');
            }
        }
        putchar('\n');
    }
    printf("Turn: %s | tiles unlimited | playable board: %ux7\n\n",
           player_name(b->turn), (unsigned)(MAX_ROWS - floor_row));
}

static bool sequence_is_syntax(const char *s)
{
    size_t n = strlen(s);
    if ((n & 1u) != 0) return false;
    for (size_t i = 0; i < n; i += 2) {
        if (s[i] < '1' || s[i] > '3') return false;
        char c = (char)toupper((unsigned char)s[i + 1]);
        if (c < 'A' || c > 'G') return false;
    }
    return true;
}

/* Returns -1 for ongoing, 0/1 for winner, 2 for draw, -2 invalid. */
static int parse_position(Board *b, uint8_t floor_row, const char *seq)
{
    board_init(b, floor_row);
    if (!seq || !*seq) return -1;
    if (!sequence_is_syntax(seq)) {
        fprintf(stderr, "Invalid position string. Use pairs such as 2D1C3A.\n");
        return -2;
    }
    size_t n = strlen(seq);
    for (size_t i = 0; i < n; i += 2) {
        int tile = seq[i] - '0';
        int col = toupper((unsigned char)seq[i + 1]) - 'A';
        int mover = b->turn;
        WinClass cl = winning_move_class(b, mover, col, tile);
        if (!board_drop(b, col, tile)) {
            fprintf(stderr, "Illegal move at characters %zu-%zu: %c%c\n", i + 1, i + 2, seq[i], seq[i + 1]);
            return -2;
        }
        if (cl != WIN_NONE) {
            if (i + 2 != n) {
                fprintf(stderr, "Position string continues after %s already won.\n", player_name(mover));
                return -2;
            }
            return mover;
        }
    }
    if (board_full(b)) return 2;
    return -1;
}

static bool exact_outcome_parse(const char *s, uint8_t *out)
{
    if (!s || !out) return false;
    char u[32]; size_t n=strlen(s); if (n>=sizeof(u)) return false;
    for (size_t i=0;i<=n;++i) u[i]=(char)toupper((unsigned char)s[i]);
    if (!strcmp(u,"GREEN_WIN") || !strcmp(u,"GWIN")) { *out=0; return true; }
    if (!strcmp(u,"YELLOW_WIN") || !strcmp(u,"YWIN")) { *out=1; return true; }
    if (!strcmp(u,"DRAW")) { *out=2; return true; }
    return false;
}

static inline uint8_t exact_prop_proved_bit(uint8_t prover)
{
    return prover==0 ? EXACT_PROP_G_PROVED : EXACT_PROP_Y_PROVED;
}

static inline uint8_t exact_prop_disproved_bit(uint8_t prover)
{
    return prover==0 ? EXACT_PROP_G_DISPROVED : EXACT_PROP_Y_DISPROVED;
}

static inline uint8_t exact_outcome_prop_mask(uint8_t outcome)
{
    if (outcome==0) return EXACT_PROP_G_PROVED | EXACT_PROP_Y_DISPROVED;
    if (outcome==1) return EXACT_PROP_G_DISPROVED | EXACT_PROP_Y_PROVED;
    return EXACT_PROP_G_DISPROVED | EXACT_PROP_Y_DISPROVED;
}

static inline bool exact_prop_mask_conflict(uint8_t m)
{
    return ((m & EXACT_PROP_G_PROVED) && (m & EXACT_PROP_G_DISPROVED)) ||
           ((m & EXACT_PROP_Y_PROVED) && (m & EXACT_PROP_Y_DISPROVED));
}

static bool exact_prop_parse_prover(const char *s, uint8_t *prover)
{
    if (!s || !prover) return false;
    char u[24]; size_t n=strlen(s); if (n>=sizeof(u)) return false;
    for (size_t i=0;i<=n;++i) u[i]=(char)toupper((unsigned char)s[i]);
    if (!strcmp(u,"GREEN") || !strcmp(u,"G")) { *prover=0; return true; }
    if (!strcmp(u,"YELLOW") || !strcmp(u,"Y")) { *prover=1; return true; }
    return false;
}

static bool exact_prop_parse_verdict(const char *s, bool *proved)
{
    if (!s || !proved) return false;
    char u[24]; size_t n=strlen(s); if (n>=sizeof(u)) return false;
    for (size_t i=0;i<=n;++i) u[i]=(char)toupper((unsigned char)s[i]);
    if (!strcmp(u,"PROVED")) { *proved=true; return true; }
    if (!strcmp(u,"DISPROVED")) { *proved=false; return true; }
    return false;
}

static void exact_db_destroy(ExactDB *db)
{
    if (!db) return;
    free(db->slot);
    free(db->raw_slot);
    free(db->raw_list);
    memset(db,0,sizeof(*db));
}

static inline uint64_t rotl64_fast(uint64_t x, unsigned r)
{
    return (x << r) | (x >> (64u-r));
}

/* Fast immutable proposition DB lookup.  As in v9.8.1, the runtime path uses
   raw orientation keys so it never duplicates normal-TT canonicalization. */
static inline uint64_t exact_db_raw_hash4(const uint64_t k[4])
{
    uint64_t h = k[0] * UINT64_C(0x9e3779b185ebca87);
    h ^= rotl64_fast(k[1] * UINT64_C(0xc2b2ae3d27d4eb4f), 17);
    h ^= rotl64_fast(k[2] * UINT64_C(0x165667b19e3779f9), 33);
    h ^= rotl64_fast(k[3] * UINT64_C(0x85ebca77c2b2ae63), 49);
    h ^= h >> 29;
    h *= UINT64_C(0x94d049bb133111eb);
    return h ^ (h >> 31);
}

static bool exact_db_init(ExactDB *db, size_t expected)
{
    memset(db,0,sizeof(*db));
    size_t sets=16;
    while (sets < expected*4 + 16) sets <<= 1;
    db->slot=(ExactDBEntry*)calloc(sets,sizeof(ExactDBEntry));
    if (!db->slot) return false;
    db->sets=sets; db->mask=sets-1;

    size_t raw_sets=16;
    while (raw_sets < expected*8 + 16) raw_sets <<= 1;
    db->raw_slot=(ExactDBRawEntry*)calloc(raw_sets,sizeof(ExactDBRawEntry));
    db->raw_list_cap=expected*2 + 8;
    db->raw_list=(ExactDBRawEntry*)calloc(db->raw_list_cap,sizeof(ExactDBRawEntry));
    if (!db->raw_slot || !db->raw_list) {
        exact_db_destroy(db);
        return false;
    }
    db->raw_sets=raw_sets; db->raw_mask=raw_sets-1;
    return true;
}

static ExactDBEntry *exact_db_find_slot(ExactDB *db, const uint64_t k[4], bool *found)
{
    if (found) *found=false;
    if (!db || !db->slot) return NULL;
    size_t idx=(size_t)(hash_key4(k)&db->mask);
    for (size_t step=0; step<db->sets; ++step) {
        ExactDBEntry *e=&db->slot[(idx+step)&db->mask];
        if (!e->valid) return e;
        if (e->k[0]==k[0] && e->k[1]==k[1] && e->k[2]==k[2] && e->k[3]==k[3]) {
            if (found) *found=true;
            return e;
        }
    }
    return NULL;
}

static inline void exact_db_raw_key(const Board *b, uint64_t k[4])
{
    k[0]=b->p[0] | ((uint64_t)b->turn << 63);
    k[1]=b->p[1];
    k[2]=b->twos;
    k[3]=b->threes;
}

static bool exact_db_insert_raw(ExactDB *db, const uint64_t k[4], uint8_t add_mask)
{
    if (!db || !db->raw_slot || !add_mask) return false;
    size_t idx=(size_t)(exact_db_raw_hash4(k)&db->raw_mask);
    for (size_t step=0; step<db->raw_sets; ++step) {
        ExactDBRawEntry *e=&db->raw_slot[(idx+step)&db->raw_mask];
        if (!e->valid) {
            memcpy(e->k,k,sizeof(e->k)); e->prop_mask=add_mask; e->valid=1;
            if (db->raw_count < db->raw_list_cap) db->raw_list[db->raw_count]=*e;
            db->raw_count++;
            return true;
        }
        if (e->k[0]==k[0] && e->k[1]==k[1] && e->k[2]==k[2] && e->k[3]==k[3]) {
            uint8_t merged=(uint8_t)(e->prop_mask | add_mask);
            if (exact_prop_mask_conflict(merged)) return false;
            e->prop_mask=merged;
            /* Keep tiny-DB linear-list copy coherent after proposition merges. */
            size_t lim=db->raw_count < db->raw_list_cap ? db->raw_count : db->raw_list_cap;
            for (size_t i=0;i<lim;++i) {
                ExactDBRawEntry *r=&db->raw_list[i];
                if (r->valid && r->k[0]==k[0] && r->k[1]==k[1] && r->k[2]==k[2] && r->k[3]==k[3]) {
                    r->prop_mask=merged; break;
                }
            }
            return true;
        }
    }
    return false;
}

static inline bool exact_db_lookup_mask(const ExactDB *db, const Board *b, uint8_t *prop_mask)
{
    if (!db || !db->raw_slot || !b) return false;
    if (b->ply >= 64 || !(db->ply_mask & (UINT64_C(1) << b->ply))) return false;

    uint64_t k[4]; exact_db_raw_key(b,k);
    if (db->raw_count <= 8 && db->raw_count <= db->raw_list_cap) {
        for (size_t i=0;i<db->raw_count;++i) {
            const ExactDBRawEntry *e=&db->raw_list[i];
            if (e->k[0]==k[0] && e->k[1]==k[1] && e->k[2]==k[2] && e->k[3]==k[3]) {
                if (prop_mask) *prop_mask=e->prop_mask;
                return true;
            }
        }
        return false;
    }

    size_t idx=(size_t)(exact_db_raw_hash4(k)&db->raw_mask);
    for (size_t step=0; step<db->raw_sets; ++step) {
        const ExactDBRawEntry *e=&db->raw_slot[(idx+step)&db->raw_mask];
        if (!e->valid) return false;
        if (e->k[0]==k[0] && e->k[1]==k[1] && e->k[2]==k[2] && e->k[3]==k[3]) {
            if (prop_mask) *prop_mask=e->prop_mask;
            return true;
        }
    }
    return false;
}

static inline bool exact_db_lookup_pair(const ExactDB *db, const Board *b, uint8_t prover,
                                        uint32_t *pn, uint32_t *dn)
{
    uint8_t m=0;
    if (!exact_db_lookup_mask(db,b,&m)) return false;
    if (m & exact_prop_proved_bit(prover)) { *pn=0; *dn=PN_INF; return true; }
    if (m & exact_prop_disproved_bit(prover)) { *pn=PN_INF; *dn=0; return true; }
    return false;
}

/* Backward compatible DB formats:
     ROWS POSITION GREEN_WIN|YELLOW_WIN|DRAW [source]
     ROWS POSITION GREEN|YELLOW PROVED|DISPROVED [source]
   Outcome facts imply both exact phase propositions.  Proposition facts may be
   accumulated independently and are merged only if logically consistent. */
static bool exact_db_load(ExactDB *db, const char *path, unsigned rows)
{
    FILE *fp=fopen(path,"r");
    if (!fp) { fprintf(stderr,"Could not open exact proposition DB %s\n",path); return false; }
    size_t expected=32; char line[1024];
    while (fgets(line,sizeof(line),fp)) if (line[0]!='#' && line[0]!='\n' && line[0]!='\r') expected++;
    rewind(fp);
    if (!exact_db_init(db,expected)) { fclose(fp); fprintf(stderr,"Could not allocate exact proposition DB.\n"); return false; }
    db->path=path; unsigned lineno=0; uint8_t floor_row=(uint8_t)(7-rows);
    while (fgets(line,sizeof(line),fp)) {
        lineno++;
        char *q=line; while (*q && isspace((unsigned char)*q)) q++;
        if (!*q || *q=='#') continue;

        unsigned erows=0; char pos[EXACT_DB_POSITION_MAX], t3[32], t4[32];
        int nf=sscanf(q,"%u %127s %31s %31s",&erows,pos,t3,t4);
        if (nf < 3 || erows<5 || erows>7) {
            fprintf(stderr,"Exact proposition DB parse error %s:%u\n",path,lineno);
            exact_db_destroy(db); fclose(fp); return false;
        }
        if (erows!=rows) { db->skipped_rows++; continue; }

        uint8_t add_mask=0; bool is_outcome=false; uint8_t outcome=0;
        if (exact_outcome_parse(t3,&outcome)) {
            add_mask=exact_outcome_prop_mask(outcome);
            is_outcome=true;
        } else {
            uint8_t prover=0; bool proved=false;
            if (nf < 4 || !exact_prop_parse_prover(t3,&prover) || !exact_prop_parse_verdict(t4,&proved)) {
                fprintf(stderr,"Exact proposition DB parse error %s:%u; expected either\n"
                               "  ROWS POSITION GREEN_WIN|YELLOW_WIN|DRAW [source]\n"
                               "or ROWS POSITION GREEN|YELLOW PROVED|DISPROVED [source]\n",path,lineno);
                exact_db_destroy(db); fclose(fp); return false;
            }
            add_mask = proved ? exact_prop_proved_bit(prover) : exact_prop_disproved_bit(prover);
        }

        Board b; int st=parse_position(&b,floor_row,pos);
        if (st==-2) { fprintf(stderr,"Exact proposition DB invalid position %s:%u: %s\n",path,lineno,pos); exact_db_destroy(db); fclose(fp); return false; }
        if (st>=0) {
            uint8_t terminal_mask=exact_outcome_prop_mask((uint8_t)st);
            if (add_mask & (uint8_t)~terminal_mask) {
                fprintf(stderr,"Exact proposition DB terminal conflict %s:%u: %s fact incompatible with terminal=%s\n",
                        path,lineno,pos,st==2?"DRAW":player_name(st));
                exact_db_destroy(db); fclose(fp); return false;
            }
        }

        uint64_t k[4]; bool mirrored; canonical_key(&b,k,&mirrored); (void)mirrored;
        bool found=false; ExactDBEntry *e=exact_db_find_slot(db,k,&found);
        if (!e) { fprintf(stderr,"Exact proposition DB table full.\n"); exact_db_destroy(db); fclose(fp); return false; }
        if (found) {
            uint8_t merged=(uint8_t)(e->prop_mask | add_mask);
            if (exact_prop_mask_conflict(merged)) {
                fprintf(stderr,"Exact proposition DB canonical conflict %s:%u: %s conflicts with %s\n",path,lineno,pos,e->position);
                exact_db_destroy(db); fclose(fp); return false;
            }
            if (merged==e->prop_mask) db->duplicates++;
            e->prop_mask=merged;
        } else {
            memcpy(e->k,k,sizeof(k)); e->prop_mask=add_mask; e->valid=1;
            snprintf(e->position,sizeof(e->position),"%s",pos); db->count++;
        }
        if (is_outcome) db->outcome_facts++; else db->proposition_facts++;

        if (b.ply < 64) db->ply_mask |= UINT64_C(1) << b.ply;
        uint64_t raw[4], raw_m[4];
        exact_db_raw_key(&b,raw);
        raw_m[0]=mirror_bits(b.p[0]) | ((uint64_t)b.turn << 63);
        raw_m[1]=mirror_bits(b.p[1]);
        raw_m[2]=mirror_bits(b.twos);
        raw_m[3]=mirror_bits(b.threes);
        if (!exact_db_insert_raw(db,raw,add_mask) || !exact_db_insert_raw(db,raw_m,add_mask)) {
            fprintf(stderr,"Exact proposition DB raw-key conflict/table full %s:%u: %s\n",path,lineno,pos);
            exact_db_destroy(db); fclose(fp); return false;
        }
    }
    fclose(fp);
    printf("[exact-db] states=%zu rawKeys=%zu outcomeFacts=%zu propositionFacts=%zu plyMask=0x%016" PRIx64
           " rows=%u skippedRows=%zu duplicates=%zu lookup=raw-fast-proposition path=%s\n",
           db->count,db->raw_count,db->outcome_facts,db->proposition_facts,db->ply_mask,
           rows,db->skipped_rows,db->duplicates,path);
    return true;
}

static size_t exact_db_report_phase(ParallelShared *sp, const ExactDB *db, uint8_t prover, const char *phase)
{
    if (!sp || !db || !db->slot) return 0;
    size_t known=0, proved=0, disproved=0;
    uint8_t pb=exact_prop_proved_bit(prover), dbit=exact_prop_disproved_bit(prover);
    for (size_t i=0;i<db->sets;++i) {
        const ExactDBEntry *e=&db->slot[i]; if (!e->valid) continue;
        if (e->prop_mask & pb) { proved++; known++; }
        else if (e->prop_mask & dbit) { disproved++; known++; }
    }
    printf("[exact-db] phase=%s prover=%s states=%zu known=%zu propositionProved=%zu propositionDisproved=%zu immutableLookup=on\n",
           phase,player_name(prover),db->count,known,proved,disproved);
    return known;
}

static void format_duration(double sec, char out[32])
{
    if (sec < 0) sec = 0;
    uint64_t s = (uint64_t)sec;
    uint64_t h = s / 3600;
    uint64_t m = (s % 3600) / 60;
    uint64_t ss = s % 60;
    snprintf(out, 32, "%02" PRIu64 ":%02" PRIu64 ":%02" PRIu64, h, m, ss);
}

static WorkStamp work_stamp(const Context *ctx)
{
    WorkStamp s = {
        ctx->stats.dfpn_calls, ctx->stats.expansions, ctx->stats.tss_calls,
        ctx->stats.tss_nodes, ctx->stats.tss_proofs, ctx->stats.oracle_collapses,
        ctx->stats.tt_hits, ctx->stats.child_cache_hits
    };
    return s;
}

static void add_work_delta(uint64_t *calls, uint64_t *expansions,
                           uint64_t *tss_calls, uint64_t *tss_nodes,
                           uint64_t *tss_proofs, uint64_t *collapses,
                           uint64_t *tt_hits, uint64_t *cc_hits,
                           WorkStamp before, WorkStamp after)
{
    *calls += after.calls - before.calls;
    *expansions += after.expansions - before.expansions;
    *tss_calls += after.tss_calls - before.tss_calls;
    *tss_nodes += after.tss_nodes - before.tss_nodes;
    *tss_proofs += after.tss_proofs - before.tss_proofs;
    *collapses += after.collapses - before.collapses;
    *tt_hits += after.tt_hits - before.tt_hits;
    *cc_hits += after.child_cache_hits - before.child_cache_hits;
}

static void move_text(uint8_t move, char out[8])
{
    if (!move) { strcpy(out, "--"); return; }
    snprintf(out, 8, "%d%c", move_tile(move), 'A' + move_col(move));
}

static uint64_t frontier_stall_exp(const Context *ctx);

/* ---------------- v9.8.2 deep proof-frontier profiler ---------------- */

static bool deep_frontier_init(Context *ctx, const Board *root)
{
    if (!ctx->deep_frontier_enabled) return true;
    DeepFrontierProfile *dp = &ctx->deep_frontier;
    memset(dp, 0, sizeof(*dp));
    dp->max_depth = ctx->deep_frontier_depth;
    dp->top = ctx->deep_frontier_top;
    dp->capacity = ctx->deep_frontier_max_nodes;
    if (dp->capacity < 2) dp->capacity = 2;
    dp->node = (DeepFrontierNode *)calloc(dp->capacity, sizeof(*dp->node));
    if (!dp->node) return false;
    dp->count = 1;
    DeepFrontierNode *r = &dp->node[0];
    r->valid = true;
    r->depth = 0;
    r->move = 0;
    r->board = *root;
    r->parent = -1;
    r->first_child = -1;
    r->next_sibling = -1;
    for (unsigned d = 0; d <= DEEP_FRONTIER_HARD_MAX_DEPTH; ++d) {
        ctx->deep_path_node[d] = -1;
        ctx->deep_active_node[d] = -1;
        memset(&ctx->deep_active_start[d], 0, sizeof(ctx->deep_active_start[d]));
    }
    ctx->deep_path_node[0] = 0;
    return true;
}

static void deep_frontier_destroy(Context *ctx)
{
    free(ctx->deep_frontier.node);
    memset(&ctx->deep_frontier, 0, sizeof(ctx->deep_frontier));
}

static void deep_frontier_reset_phase(Context *ctx, const Board *root)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node) return;
    DeepFrontierProfile *dp = &ctx->deep_frontier;
    memset(dp->node, 0, (size_t)(dp->count ? dp->count : 1u) * sizeof(*dp->node));
    dp->count = 1;
    dp->dropped_nodes = 0;
    DeepFrontierNode *r = &dp->node[0];
    r->valid = true;
    r->depth = 0;
    r->board = *root;
    r->parent = -1;
    r->first_child = -1;
    r->next_sibling = -1;
    for (unsigned d = 0; d <= DEEP_FRONTIER_HARD_MAX_DEPTH; ++d) {
        ctx->deep_path_node[d] = -1;
        ctx->deep_active_node[d] = -1;
        memset(&ctx->deep_active_start[d], 0, sizeof(ctx->deep_active_start[d]));
    }
    ctx->deep_path_node[0] = 0;
}

static int32_t deep_frontier_child(Context *ctx, int32_t parent, const Child *ch, unsigned child_depth)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || !ch ||
        parent < 0 || child_depth > ctx->deep_frontier_depth) return -1;
    DeepFrontierProfile *dp = &ctx->deep_frontier;
    if ((uint32_t)parent >= dp->count) return -1;
    for (int32_t i = dp->node[parent].first_child; i >= 0; i = dp->node[i].next_sibling) {
        if (dp->node[i].valid && dp->node[i].move == ch->move) return i;
    }
    if (dp->count >= dp->capacity) {
        dp->dropped_nodes++;
        return -1;
    }
    int32_t idx = (int32_t)dp->count++;
    DeepFrontierNode *n = &dp->node[idx];
    memset(n, 0, sizeof(*n));
    n->valid = true;
    n->depth = (uint8_t)child_depth;
    n->move = ch->move;
    n->board = ch->board;
    n->parent = parent;
    n->first_child = -1;
    n->next_sibling = dp->node[parent].first_child;
    dp->node[parent].first_child = idx;
    return idx;
}

static WorkStamp deep_frontier_effective_stamp(const Context *ctx, int32_t idx)
{
    WorkStamp z = {0};
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || idx < 0 ||
        (uint32_t)idx >= ctx->deep_frontier.count) return z;
    const DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    WorkStamp s = { n->work_calls, n->work_expansions, n->work_tss_calls,
                    n->work_tss_nodes, n->work_tss_proofs, n->work_collapses,
                    n->work_tt_hits, n->work_child_cache_hits };
    unsigned d = n->depth;
    if (d <= DEEP_FRONTIER_HARD_MAX_DEPTH && ctx->deep_active_node[d] == idx) {
        WorkStamp now = work_stamp(ctx);
        WorkStamp before = ctx->deep_active_start[d];
        s.calls += now.calls - before.calls;
        s.expansions += now.expansions - before.expansions;
        s.tss_calls += now.tss_calls - before.tss_calls;
        s.tss_nodes += now.tss_nodes - before.tss_nodes;
        s.tss_proofs += now.tss_proofs - before.tss_proofs;
        s.collapses += now.collapses - before.collapses;
        s.tt_hits += now.tt_hits - before.tt_hits;
        s.child_cache_hits += now.child_cache_hits - before.child_cache_hits;
    }
    return s;
}

static void deep_frontier_add_work(Context *ctx, int32_t idx, WorkStamp before, WorkStamp after)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || idx < 0 ||
        (uint32_t)idx >= ctx->deep_frontier.count) return;
    DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    n->descents++;
    add_work_delta(&n->work_calls, &n->work_expansions,
                   &n->work_tss_calls, &n->work_tss_nodes, &n->work_tss_proofs,
                   &n->work_collapses, &n->work_tt_hits, &n->work_child_cache_hits,
                   before, after);
}

static void deep_frontier_note_branching(Context *ctx, unsigned depth, int n)
{
    if (!ctx->deep_frontier_enabled || depth > ctx->deep_frontier_depth ||
        depth > DEEP_FRONTIER_HARD_MAX_DEPTH) return;
    int32_t idx = ctx->deep_path_node[depth];
    if (idx < 0 || (uint32_t)idx >= ctx->deep_frontier.count) return;
    if (n < 0) n = 0;
    if (n > 255) n = 255;
    ctx->deep_frontier.node[idx].branching = (uint8_t)n;
}

static void deep_frontier_note_pair(Context *ctx, unsigned depth, uint32_t pn, uint32_t dn)
{
    if (!ctx->deep_frontier_enabled || depth > ctx->deep_frontier_depth ||
        depth > DEEP_FRONTIER_HARD_MAX_DEPTH) return;
    int32_t idx = ctx->deep_path_node[depth];
    if (idx < 0 || (uint32_t)idx >= ctx->deep_frontier.count) return;
    DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    WorkStamp ew = deep_frontier_effective_stamp(ctx, idx);
    if (!n->pair_valid || n->last_pn != pn || n->last_dn != dn) {
        n->pair_valid = 1;
        n->last_pn = pn;
        n->last_dn = dn;
        n->work_at_pair_change = ew.expansions;
    }
}

static uint64_t deep_frontier_stall(const Context *ctx, int32_t idx)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || idx < 0 ||
        (uint32_t)idx >= ctx->deep_frontier.count) return 0;
    const DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    WorkStamp ew = deep_frontier_effective_stamp(ctx, idx);
    return n->pair_valid && ew.expansions >= n->work_at_pair_change ?
           ew.expansions - n->work_at_pair_change : 0;
}

static void deep_frontier_path_text(const Context *ctx, int32_t idx, char *out, size_t cap)
{
    if (!out || cap == 0) return;
    out[0] = '\0';
    if (!ctx->deep_frontier.node || idx <= 0 || (uint32_t)idx >= ctx->deep_frontier.count) {
        snprintf(out, cap, "ROOT");
        return;
    }
    uint8_t moves[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    unsigned nm = 0;
    int32_t cur = idx;
    while (cur > 0 && nm <= DEEP_FRONTIER_HARD_MAX_DEPTH) {
        moves[nm++] = ctx->deep_frontier.node[cur].move;
        cur = ctx->deep_frontier.node[cur].parent;
    }
    size_t used = 0;
    for (unsigned k = 0; k < nm; ++k) {
        uint8_t mv = moves[nm - 1 - k];
        char m[8]; move_text(mv, m);
        int wr = snprintf(out + used, cap > used ? cap - used : 0, "%s%s", k ? ">" : "", m);
        if (wr < 0 || (size_t)wr >= (cap > used ? cap - used : 0)) break;
        used += (size_t)wr;
    }
}

static int deep_frontier_ranked_children(const Context *ctx, int32_t parent,
                                         int32_t out[MAX_MOVES], unsigned max_out,
                                         uint64_t *sum_work)
{
    if (sum_work) *sum_work = 0;
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || parent < 0 ||
        (uint32_t)parent >= ctx->deep_frontier.count || max_out == 0) return 0;
    unsigned nout = 0;
    for (int32_t c = ctx->deep_frontier.node[parent].first_child; c >= 0;
         c = ctx->deep_frontier.node[c].next_sibling) {
        WorkStamp cw = deep_frontier_effective_stamp(ctx, c);
        if (sum_work) *sum_work += cw.expansions;
        unsigned pos;
        if (nout < max_out) {
            pos = nout++;
        } else {
            WorkStamp tail = deep_frontier_effective_stamp(ctx, out[max_out - 1]);
            if (tail.expansions >= cw.expansions) continue;
            pos = max_out - 1;
        }
        while (pos > 0) {
            WorkStamp pw = deep_frontier_effective_stamp(ctx, out[pos - 1]);
            if (pw.expansions >= cw.expansions) break;
            out[pos] = out[pos - 1];
            pos--;
        }
        out[pos] = c;
    }
    return (int)nout;
}

static void deep_frontier_csv_header(FILE *fp)
{
    fputs("event,phase,elapsed_s,root_pn,root_dn,root_stall_exp,depth,rank,path,parent_path,move,"
          "pn,dn,tt_resident,descents,branching,work_calls,work_expansions,sibling_work_expansions,work_share,"
          "stall_work_expansions,tss_calls,tss_nodes,tss_proofs,tss_proof_rate,collapses,tt_hits,child_cache_hits,"
          "profile_nodes,profile_capacity,dropped_nodes\n", fp);
    fflush(fp);
}

static void deep_frontier_snapshot(Context *ctx, const char *event, double elapsed)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || ctx->deep_frontier.count == 0) return;
    DeepFrontierProfile *dp = &ctx->deep_frontier;
    int32_t parent = 0;
    char console[768];
    size_t cu = 0;
    console[0] = '\0';

    for (unsigned depth = 1; depth <= dp->max_depth && parent >= 0; ++depth) {
        int32_t ranked[MAX_MOVES];
        uint64_t sibling_sum = 0;
        int nr = deep_frontier_ranked_children(ctx, parent, ranked,
                                               dp->top < MAX_MOVES ? dp->top : MAX_MOVES,
                                               &sibling_sum);
        if (nr <= 0) break;

        if (ctx->deep_frontier_fp) {
            char parent_path[128]; deep_frontier_path_text(ctx, parent, parent_path, sizeof(parent_path));
            for (int r = 0; r < nr; ++r) {
                int32_t idx = ranked[r];
                DeepFrontierNode *n = &dp->node[idx];
                WorkStamp ew = deep_frontier_effective_stamp(ctx, idx);
                uint32_t pn = 1, dn = 1;
                bool resident = tt_pair_peek(ctx, &n->board, &pn, &dn);
                char path[128], mv[8];
                deep_frontier_path_text(ctx, idx, path, sizeof(path));
                move_text(n->move, mv);
                double share = sibling_sum ? (double)ew.expansions / (double)sibling_sum : 0.0;
                double tss_rate = ew.tss_calls ? (double)ew.tss_proofs / (double)ew.tss_calls : 0.0;
                fprintf(ctx->deep_frontier_fp,
                        "%s,%s,%.6f,%u,%u,%" PRIu64 ",%u,%d,%s,%s,%s,%u,%u,%u,%" PRIu64 ",%u,"
                        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.8f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.10f,"
                        "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%u,%" PRIu64 "\n",
                        event, ctx->phase_name, elapsed, ctx->root_pn, ctx->root_dn,
                        frontier_stall_exp(ctx), depth, r + 1, path, parent_path, mv,
                        pn, dn, resident ? 1u : 0u, n->descents + (ctx->deep_active_node[depth] == idx ? 1u : 0u),
                        (unsigned)n->branching, ew.calls, ew.expansions, sibling_sum, share,
                        deep_frontier_stall(ctx, idx), ew.tss_calls, ew.tss_nodes, ew.tss_proofs, tss_rate,
                        ew.collapses, ew.tt_hits, ew.child_cache_hits, dp->count, dp->capacity, dp->dropped_nodes);
            }
        }

        int32_t hot = ranked[0];
        DeepFrontierNode *hn = &dp->node[hot];
        WorkStamp hw = deep_frontier_effective_stamp(ctx, hot);
        uint32_t hpn = 1, hdn = 1; tt_pair_peek(ctx, &hn->board, &hpn, &hdn);
        char hm[8]; move_text(hn->move, hm);
        double hshare = sibling_sum ? 100.0 * (double)hw.expansions / (double)sibling_sum : 0.0;
        int wr = snprintf(console + cu, sizeof(console) - cu,
                          "%s%s %.1f%% (%u,%u) br=%u stall=%" PRIu64,
                          depth == 1 ? "" : " > ", hm, hshare, hpn, hdn,
                          (unsigned)hn->branching, deep_frontier_stall(ctx, hot));
        if (wr > 0 && (size_t)wr < sizeof(console) - cu) cu += (size_t)wr;
        parent = hot;
    }
    if (ctx->deep_frontier_fp) fflush(ctx->deep_frontier_fp);
    if (console[0]) {
        printf("[deep-frontier] %s nodes=%u/%u dropped=%" PRIu64 "\n",
               console, dp->count, dp->capacity, dp->dropped_nodes);
        fflush(stdout);
    }
}

static void frontier_reset(Context *ctx)
{
    memset(&ctx->frontier, 0, sizeof(ctx->frontier));
    ctx->frontier.last_root_pn = 1;
    ctx->frontier.last_root_dn = 1;
    ctx->frontier.expansions_at_root_change = 0;
}

static void frontier_init_root(Context *ctx, const Child children[MAX_MOVES], int n, bool root_is_or)
{
    if (!ctx->frontier_enabled || ctx->frontier.initialized || n <= 0) return;
    FrontierProfile *fp = &ctx->frontier;
    fp->initialized = true;
    fp->root_is_or = root_is_or;
    fp->child_count = (uint8_t)(n > MAX_MOVES ? MAX_MOVES : n);
    fp->last_root_pn = ctx->root_pn;
    fp->last_root_dn = ctx->root_dn;
    fp->expansions_at_root_change = ctx->stats.expansions;
    for (int i = 0; i < n && i < MAX_MOVES; ++i) {
        fp->child[i].valid = true;
        fp->child[i].move = children[i].move;
        fp->child[i].board = children[i].board;
    }
}

static int frontier_child_by_move(Context *ctx, uint8_t move)
{
    FrontierProfile *fp = &ctx->frontier;
    if (!ctx->frontier_enabled || !fp->initialized) return -1;
    for (unsigned i = 0; i < fp->child_count; ++i)
        if (fp->child[i].valid && fp->child[i].move == move) return (int)i;
    return -1;
}

static int frontier_child_by_board(Context *ctx, const Board *b)
{
    FrontierProfile *fp = &ctx->frontier;
    if (!ctx->frontier_enabled || !fp->initialized) return -1;
    for (unsigned i = 0; i < fp->child_count; ++i)
        if (fp->child[i].valid && same_canonical_state(&fp->child[i].board, b)) return (int)i;
    return -1;
}

static int frontier_reply_get(Context *ctx, int child_index, const Child *reply)
{
    if (child_index < 0 || child_index >= MAX_MOVES || !reply) return -1;
    FrontierChild *fc = &ctx->frontier.child[child_index];
    for (unsigned i = 0; i < fc->reply_count; ++i) {
        if (fc->reply[i].valid && fc->reply[i].move == reply->move &&
            same_canonical_state(&fc->reply[i].board, &reply->board)) return (int)i;
    }
    if (fc->reply_count >= MAX_MOVES) return -1;
    unsigned i = fc->reply_count++;
    FrontierReply *fr = &fc->reply[i];
    memset(fr, 0, sizeof(*fr));
    fr->valid = true;
    fr->move = reply->move;
    fr->board = reply->board;
    return (int)i;
}

static void frontier_add_child_work(Context *ctx, int ci, WorkStamp before, WorkStamp after)
{
    if (ci < 0 || ci >= MAX_MOVES) return;
    FrontierChild *fc = &ctx->frontier.child[ci];
    fc->descents++;
    add_work_delta(&fc->work_calls, &fc->work_expansions,
                   &fc->work_tss_calls, &fc->work_tss_nodes, &fc->work_tss_proofs,
                   &fc->work_collapses, &fc->work_tt_hits, &fc->work_child_cache_hits,
                   before, after);
}

static void frontier_add_reply_work(Context *ctx, int ci, int ri, WorkStamp before, WorkStamp after)
{
    if (ci < 0 || ci >= MAX_MOVES || ri < 0 || ri >= MAX_MOVES) return;
    FrontierReply *fr = &ctx->frontier.child[ci].reply[ri];
    fr->descents++;
    add_work_delta(&fr->work_calls, &fr->work_expansions,
                   &fr->work_tss_calls, &fr->work_tss_nodes, &fr->work_tss_proofs,
                   &fr->work_collapses, &fr->work_tt_hits, &fr->work_child_cache_hits,
                   before, after);
}

static void frontier_note_root_pair(Context *ctx, uint32_t pn, uint32_t dn)
{
    if (!ctx->frontier_enabled || !ctx->frontier.initialized) return;
    FrontierProfile *fp = &ctx->frontier;
    if (pn != fp->last_root_pn || dn != fp->last_root_dn) {
        fp->last_root_pn = pn;
        fp->last_root_dn = dn;
        fp->expansions_at_root_change = ctx->stats.expansions;
    }
}

static uint64_t frontier_stall_exp(const Context *ctx)
{
    const FrontierProfile *fp = &ctx->frontier;
    return ctx->stats.expansions >= fp->expansions_at_root_change ?
        ctx->stats.expansions - fp->expansions_at_root_change : 0;
}

static void frontier_csv_header(FILE *fp)
{
    fputs("event,phase,elapsed_s,root_pn,root_dn,root_type,stall_exp,snap_delta_exp,root_delta_pn,root_delta_dn,root_motion_per_mexp,root_move,"
          "child_pn,child_dn,child_tt_resident,child_descents,work_calls,work_expansions,work_share,"
          "work_tss_calls,work_tss_nodes,work_tss_proofs,work_collapses,work_tt_hits,work_child_cache_hits,"
          "hot_reply,reply_pn,reply_dn,reply_tt_resident,reply_descents,reply_work_expansions,reply_work_share_within_child,"
          "hot_attempts,hot_nodes,hot_level,hot_result,tss_gate\n", fp);
    fflush(fp);
}

static void frontier_snapshot(Context *ctx, const char *event, double elapsed)
{
    if (!ctx->frontier_enabled || !ctx->frontier.initialized) return;
    FrontierProfile *fp = &ctx->frontier;
    if (ctx->root_pn != fp->last_root_pn || ctx->root_dn != fp->last_root_dn) {
        fp->last_root_pn = ctx->root_pn;
        fp->last_root_dn = ctx->root_dn;
        fp->expansions_at_root_change = ctx->stats.expansions;
    }
    uint64_t stall = ctx->stats.expansions >= fp->expansions_at_root_change ?
        ctx->stats.expansions - fp->expansions_at_root_change : 0;
    uint64_t snap_delta_exp = 0;
    uint32_t root_delta_pn = 0, root_delta_dn = 0;
    double root_motion_per_mexp = 0.0;
    if (fp->snapshot_valid) {
        snap_delta_exp = ctx->stats.expansions - fp->prev_snapshot_expansions;
        root_delta_pn = ctx->root_pn >= fp->prev_snapshot_pn ? ctx->root_pn - fp->prev_snapshot_pn : fp->prev_snapshot_pn - ctx->root_pn;
        root_delta_dn = ctx->root_dn >= fp->prev_snapshot_dn ? ctx->root_dn - fp->prev_snapshot_dn : fp->prev_snapshot_dn - ctx->root_dn;
        if (snap_delta_exp) root_motion_per_mexp = 1000000.0 * (double)((uint64_t)root_delta_pn + root_delta_dn) / (double)snap_delta_exp;
    }

    int hot = -1, mpc = -1;
    uint64_t hot_work = 0;
    uint32_t best_metric = PN_INF;
    for (unsigned i = 0; i < fp->child_count; ++i) {
        FrontierChild *fc = &fp->child[i];
        uint32_t pn = 1, dn = 1;
        tt_pair_peek(ctx, &fc->board, &pn, &dn);
        if (hot < 0 || fc->work_expansions > hot_work) { hot = (int)i; hot_work = fc->work_expansions; }
        uint32_t metric = fp->root_is_or ? pn : dn;
        if (mpc < 0 || metric < best_metric) { mpc = (int)i; best_metric = metric; }
    }

    if (ctx->frontier_fp) {
        for (unsigned i = 0; i < fp->child_count; ++i) {
            FrontierChild *fc = &fp->child[i];
            uint32_t pn = 1, dn = 1;
            bool child_resident = tt_pair_peek(ctx, &fc->board, &pn, &dn);
            int hot_reply = -1;
            uint64_t reply_work = 0;
            for (unsigned j = 0; j < fc->reply_count; ++j) {
                FrontierReply *fr = &fc->reply[j];
                if (hot_reply < 0 || fr->work_expansions > reply_work) {
                    hot_reply = (int)j; reply_work = fr->work_expansions;
                }
            }
            char mv[8], rmv[8]; move_text(fc->move, mv); strcpy(rmv, "--");
            uint32_t rpn = 1, rdn = 1; uint64_t rdesc = 0; bool reply_resident = false;
            if (hot_reply >= 0) {
                FrontierReply *fr = &fc->reply[hot_reply];
                move_text(fr->move, rmv);
                reply_resident = tt_pair_peek(ctx, &fr->board, &rpn, &rdn);
                rdesc = fr->descents;
            }
            double share = ctx->stats.expansions ? (double)fc->work_expansions / (double)ctx->stats.expansions : 0.0;
            double rshare = fc->work_expansions ? (double)reply_work / (double)fc->work_expansions : 0.0;
            fprintf(ctx->frontier_fp,
                    "%s,%s,%.6f,%u,%u,%s,%" PRIu64 ",%" PRIu64 ",%u,%u,%.8f,%s,%u,%u,%u,%" PRIu64
                    ",%" PRIu64 ",%" PRIu64 ",%.8f,%" PRIu64 ",%" PRIu64 ",%" PRIu64
                    ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%s,%u,%u,%u,%" PRIu64 ",%" PRIu64 ",%.8f,%" PRIu64 ",%" PRIu64 ",%u,%d,%u\n",
                    event, ctx->phase_name, elapsed, ctx->root_pn, ctx->root_dn,
                    fp->root_is_or ? "OR" : "AND", stall, snap_delta_exp, root_delta_pn, root_delta_dn, root_motion_per_mexp, mv, pn, dn, child_resident ? 1u : 0u, fc->descents,
                    fc->work_calls, fc->work_expansions, share, fc->work_tss_calls,
                    fc->work_tss_nodes, fc->work_tss_proofs, fc->work_collapses,
                    fc->work_tt_hits, fc->work_child_cache_hits, rmv, rpn, rdn, reply_resident ? 1u : 0u,
                    rdesc, reply_work, rshare,
                    hot_reply >= 0 ? fc->reply[hot_reply].hot_exact_attempts : 0,
                    hot_reply >= 0 ? fc->reply[hot_reply].hot_exact_nodes : 0,
                    hot_reply >= 0 ? (unsigned)fc->reply[hot_reply].hot_attempt_level : 0u,
                    (hot_reply >= 0 && fc->reply[hot_reply].hot_result_valid) ? (int)fc->reply[hot_reply].hot_result : -1,
                    (hot_reply >= 0 && fc->reply[hot_reply].tss_gate_active) ? 1u : 0u);
        }
        fflush(ctx->frontier_fp);
    }

    if (hot >= 0) {
        FrontierChild *hc = &fp->child[hot];
        uint32_t hpn = 1, hdn = 1; tt_pair_peek(ctx, &hc->board, &hpn, &hdn);
        char hmv[8], mmv[8]; move_text(hc->move, hmv); strcpy(mmv, "--");
        if (mpc >= 0) move_text(fp->child[mpc].move, mmv);
        double share = ctx->stats.expansions ? 100.0 * (double)hc->work_expansions / (double)ctx->stats.expansions : 0.0;
        int hr = -1; uint64_t hrw = 0;
        for (unsigned j = 0; j < hc->reply_count; ++j)
            if (hr < 0 || hc->reply[j].work_expansions > hrw) { hr=(int)j; hrw=hc->reply[j].work_expansions; }
        char rmv[8]; strcpy(rmv, "--");
        if (hr >= 0) move_text(hc->reply[hr].move, rmv);
        printf("[frontier] hot=%s %.1f%% child=(%u,%u) reply=%s mpc=%s stall=%" PRIu64 "exp motion=%.2f/M\n",
               hmv, share, hpn, hdn, rmv, mmv, stall, root_motion_per_mexp);
        fflush(stdout);
    }
    fp->snapshot_valid = true;
    fp->prev_snapshot_pn = ctx->root_pn;
    fp->prev_snapshot_dn = ctx->root_dn;
    fp->prev_snapshot_expansions = ctx->stats.expansions;
}

static void deep_reroot_node_status(const Context *ctx, unsigned *depth_out,
                                    uint64_t *node_stall_out, int32_t *idx_out);

static void csv_header(FILE *fp)
{
    fputs("event,phase,elapsed_s,dfpn_calls,expansions,expanded_children,avg_branch,"
          "child_descents,unique_children_examined,avg_children_examined,avg_examined_rank,top3_examined_rate,first_pick_nodes,avg_first_pick_rank,first_pick_top3_rate,"
          "child_generations,moves_generated,child_cache_probes,child_cache_hits,child_cache_hit_rate,child_cache_stores,child_cache_restored,tt_probes,tt_hits,tt_hit_rate,"
          "tss_calls,tss_nodes,tss_proofs,tss_retries,tss_deep_proofs,tss_cache_hits,"
          "tss_forcing_candidates,tss_candidates_kept,tss_signature_pruned,sibling_dedup_pruned,"
          "dep_light_nodes,dep_light_segments_tested,dep_maps,dep_segments_tested,"
          "dep_viable_green,dep_viable_yellow,dep_hint_uses,dep_ordered_moves,dep_dead_checks,dep_dead_prunes,"
          "dep_oracle_calls,dep_oracle_nodes,dep_oracle_proofs,dep_oracle_candidates,dep_oracle_defender_replies,dep_oracle_budget_exhausted,"
          "tactical_calls,tactical_nodes,tactical_children_solved,tactical_wins,"
          "tactical_losses,tactical_draws,tactical_budget_exhausted,"
          "hot_exact_triggers,hot_exact_attempts,hot_exact_nodes,hot_exact_tt_hits,hot_exact_proofs,hot_exact_wins,hot_exact_losses,hot_exact_draws,hot_exact_budget_exhausted,hot_exact_depth_cutoffs,hot_tss_gate_activations,hot_tss_gated_nodes,"
          "work_quantum,work_quantum_yields,work_quantum_yield_expansions,work_quantum_max_call_exp,root_mpc_switches,"
          "deep_reroot_triggers,deep_reroot_attempts,deep_reroot_proofs,deep_reroot_proved,deep_reroot_disproved,deep_reroot_unknown,deep_reroot_sub_calls,deep_reroot_sub_expansions,deep_reroot_sub_ms,deep_reroot_overlay_hits,deep_reroot_fresh_sessions,deep_reroot_resumes,deep_reroot_session_evictions,deep_reroot_sub_tt_probes,deep_reroot_sub_tt_hits,deep_reroot_slices,deep_reroot_slice_unknown,deep_reroot_focus_extra_slices,deep_reroot_focus_exhausted,deep_reroot_node_depth,deep_reroot_node_stall_exp,"
          "bridge_calls,bridge_nodes,bridge_proofs,bridge_candidates,bridge_defender_replies,"
          "bridge_budget_exhausted,oracle_collapses,forced_defenses,double_threat_wins,"
          "double_threat_losses,unsafe_pruned,symmetry_pruned,immediate_terminals,"
          "max_ply,root_pn,root_dn,calls_per_s\n", fp);
    fflush(fp);
}

static void log_stats(Context *ctx, const char *event)
{
    double now = wall_time();
    double elapsed = now - ctx->phase_start;
    double rate = elapsed > 0.0 ? (double)ctx->stats.dfpn_calls / elapsed : 0.0;
    double hit_rate = ctx->stats.tt_probes ? (double)ctx->stats.tt_hits / (double)ctx->stats.tt_probes : 0.0;
    double child_cache_hit_rate = ctx->stats.child_cache_probes ?
        (double)ctx->stats.child_cache_hits / (double)ctx->stats.child_cache_probes : 0.0;
    double avg_branch = ctx->stats.expansions ?
        (double)ctx->stats.expanded_children / (double)ctx->stats.expansions : 0.0;
    double avg_examined = ctx->stats.expansions ?
        (double)ctx->stats.unique_children_examined / (double)ctx->stats.expansions : 0.0;
    double avg_rank = ctx->stats.unique_children_examined ?
        (double)ctx->stats.unique_rank_sum / (double)ctx->stats.unique_children_examined : 0.0;
    double top3_rate = ctx->stats.unique_children_examined ?
        (double)ctx->stats.unique_top3_examined / (double)ctx->stats.unique_children_examined : 0.0;
    double avg_first_rank = ctx->stats.first_pick_nodes ?
        (double)ctx->stats.first_pick_rank_sum / (double)ctx->stats.first_pick_nodes : 0.0;
    double first_top3_rate = ctx->stats.first_pick_nodes ?
        (double)ctx->stats.first_pick_top3 / (double)ctx->stats.first_pick_nodes : 0.0;
    char dur[32];
    format_duration(elapsed, dur);
    unsigned rr_node_depth = 0;
    uint64_t rr_node_stall = 0;
    deep_reroot_node_status(ctx, &rr_node_depth, &rr_node_stall, NULL);

    printf("[%s %s] calls=%" PRIu64 " exp=%" PRIu64 " branch=%.2f exam=%.2f rank=%.2f top3=%.1f%%"
           " first=%.2f/%.1f%% gC=%.1f%%/%" PRIu64 " TT=%.1f%% TSS=%" PRIu64 "/%" PRIu64 " sigpr=%" PRIu64
           " depL=%" PRIu64 " depF=%" PRIu64 " hint=%" PRIu64
           " dead=%" PRIu64 "/%" PRIu64 " hotE=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " tGate=%" PRIu64
           " q=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " mpcSw=%" PRIu64
           " rr=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " rrExp=%" PRIu64
           " rrFR=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " rrTT=%.1f%%"
           " rrSl=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
           " rrNode=%u/%" PRIu64 " collapse=%" PRIu64 " root=(%u,%u) maxply=%u %.0f calls/s\n",
           ctx->phase_name, dur, ctx->stats.dfpn_calls, ctx->stats.expansions,
           avg_branch, avg_examined, avg_rank, 100.0 * top3_rate,
           avg_first_rank, 100.0 * first_top3_rate,
           100.0 * child_cache_hit_rate, ctx->stats.child_cache_restored,
           100.0 * hit_rate, ctx->stats.tss_proofs, ctx->stats.tss_calls,
           ctx->stats.tss_signature_pruned,
           ctx->stats.dep_light_nodes, ctx->stats.dep_maps, ctx->stats.dep_hint_uses,
           ctx->stats.dep_dead_prunes, ctx->stats.dep_dead_checks,
           ctx->stats.hot_exact_proofs, ctx->stats.hot_exact_attempts, ctx->stats.hot_exact_nodes,
           ctx->stats.hot_tss_gated_nodes, ctx->work_quantum, ctx->stats.work_quantum_yields,
           ctx->stats.work_quantum_max_call_exp, ctx->stats.root_mpc_switches,
           ctx->stats.deep_reroot_proofs, ctx->stats.deep_reroot_attempts,
           ctx->stats.deep_reroot_unknown, ctx->stats.deep_reroot_sub_expansions,
           ctx->stats.deep_reroot_fresh_sessions, ctx->stats.deep_reroot_resumes,
           ctx->stats.deep_reroot_session_evictions,
           ctx->stats.deep_reroot_sub_tt_probes ?
               100.0 * (double)ctx->stats.deep_reroot_sub_tt_hits /
               (double)ctx->stats.deep_reroot_sub_tt_probes : 0.0,
           ctx->stats.deep_reroot_slices, ctx->stats.deep_reroot_slice_unknown,
           ctx->stats.deep_reroot_focus_extra_slices, ctx->stats.deep_reroot_focus_exhausted,
           rr_node_depth, rr_node_stall,
           ctx->stats.oracle_collapses, ctx->root_pn, ctx->root_dn,
           ctx->stats.max_ply, rate);
    fflush(stdout);

    if (ctx->log_fp) {
        FILE *fp = ctx->log_fp;
        fprintf(fp, "%s,%s,%.6f", event, ctx->phase_name, elapsed);
#define U64F(x) fprintf(fp, ",%" PRIu64, (uint64_t)(x))
#define DBLF(x) fprintf(fp, ",%.8f", (double)(x))
        U64F(ctx->stats.dfpn_calls);
        U64F(ctx->stats.expansions);
        U64F(ctx->stats.expanded_children);
        DBLF(avg_branch);
        U64F(ctx->stats.child_descents);
        U64F(ctx->stats.unique_children_examined);
        DBLF(avg_examined);
        DBLF(avg_rank);
        DBLF(top3_rate);
        U64F(ctx->stats.first_pick_nodes);
        DBLF(avg_first_rank);
        DBLF(first_top3_rate);
        U64F(ctx->stats.child_generations);
        U64F(ctx->stats.moves_generated);
        U64F(ctx->stats.child_cache_probes);
        U64F(ctx->stats.child_cache_hits);
        DBLF(child_cache_hit_rate);
        U64F(ctx->stats.child_cache_stores);
        U64F(ctx->stats.child_cache_restored);
        U64F(ctx->stats.tt_probes);
        U64F(ctx->stats.tt_hits);
        DBLF(hit_rate);
        U64F(ctx->stats.tss_calls);
        U64F(ctx->stats.tss_nodes);
        U64F(ctx->stats.tss_proofs);
        U64F(ctx->stats.tss_retries);
        U64F(ctx->stats.tss_deep_proofs);
        U64F(ctx->stats.tss_cache_hits);
        U64F(ctx->stats.tss_forcing_candidates);
        U64F(ctx->stats.tss_candidates_kept);
        U64F(ctx->stats.tss_signature_pruned);
        U64F(ctx->stats.sibling_dedup_pruned);
        U64F(ctx->stats.dep_light_nodes);
        U64F(ctx->stats.dep_light_segments_tested);
        U64F(ctx->stats.dep_maps);
        U64F(ctx->stats.dep_segments_tested);
        U64F(ctx->stats.dep_viable_green);
        U64F(ctx->stats.dep_viable_yellow);
        U64F(ctx->stats.dep_hint_uses);
        U64F(ctx->stats.dep_ordered_moves);
        U64F(ctx->stats.dep_dead_checks);
        U64F(ctx->stats.dep_dead_prunes);
        U64F(ctx->stats.dep_oracle_calls);
        U64F(ctx->stats.dep_oracle_nodes);
        U64F(ctx->stats.dep_oracle_proofs);
        U64F(ctx->stats.dep_oracle_candidates);
        U64F(ctx->stats.dep_oracle_defender_replies);
        U64F(ctx->stats.dep_oracle_budget_exhausted);
        U64F(ctx->stats.tactical_calls);
        U64F(ctx->stats.tactical_nodes);
        U64F(ctx->stats.tactical_children_solved);
        U64F(ctx->stats.tactical_wins);
        U64F(ctx->stats.tactical_losses);
        U64F(ctx->stats.tactical_draws);
        U64F(ctx->stats.tactical_budget_exhausted);
        U64F(ctx->stats.hot_exact_triggers);
        U64F(ctx->stats.hot_exact_attempts);
        U64F(ctx->stats.hot_exact_nodes);
        U64F(ctx->stats.hot_exact_tt_hits);
        U64F(ctx->stats.hot_exact_proofs);
        U64F(ctx->stats.hot_exact_wins);
        U64F(ctx->stats.hot_exact_losses);
        U64F(ctx->stats.hot_exact_draws);
        U64F(ctx->stats.hot_exact_budget_exhausted);
        U64F(ctx->stats.hot_exact_depth_cutoffs);
        U64F(ctx->stats.hot_tss_gate_activations);
        U64F(ctx->stats.hot_tss_gated_nodes);
        U64F(ctx->work_quantum);
        U64F(ctx->stats.work_quantum_yields);
        U64F(ctx->stats.work_quantum_yield_expansions);
        U64F(ctx->stats.work_quantum_max_call_exp);
        U64F(ctx->stats.root_mpc_switches);
        U64F(ctx->stats.deep_reroot_triggers);
        U64F(ctx->stats.deep_reroot_attempts);
        U64F(ctx->stats.deep_reroot_proofs);
        U64F(ctx->stats.deep_reroot_proved);
        U64F(ctx->stats.deep_reroot_disproved);
        U64F(ctx->stats.deep_reroot_unknown);
        U64F(ctx->stats.deep_reroot_sub_calls);
        U64F(ctx->stats.deep_reroot_sub_expansions);
        U64F(ctx->stats.deep_reroot_sub_ms);
        U64F(ctx->stats.deep_reroot_overlay_hits);
        U64F(ctx->stats.deep_reroot_fresh_sessions);
        U64F(ctx->stats.deep_reroot_resumes);
        U64F(ctx->stats.deep_reroot_session_evictions);
        U64F(ctx->stats.deep_reroot_sub_tt_probes);
        U64F(ctx->stats.deep_reroot_sub_tt_hits);
        U64F(ctx->stats.deep_reroot_slices);
        U64F(ctx->stats.deep_reroot_slice_unknown);
        U64F(ctx->stats.deep_reroot_focus_extra_slices);
        U64F(ctx->stats.deep_reroot_focus_exhausted);
        U64F(rr_node_depth);
        U64F(rr_node_stall);
        U64F(ctx->stats.bridge_calls);
        U64F(ctx->stats.bridge_nodes);
        U64F(ctx->stats.bridge_proofs);
        U64F(ctx->stats.bridge_candidates);
        U64F(ctx->stats.bridge_defender_replies);
        U64F(ctx->stats.bridge_budget_exhausted);
        U64F(ctx->stats.oracle_collapses);
        U64F(ctx->stats.forced_defenses);
        U64F(ctx->stats.double_threat_wins);
        U64F(ctx->stats.double_threat_losses);
        U64F(ctx->stats.unsafe_pruned);
        U64F(ctx->stats.symmetry_pruned);
        U64F(ctx->stats.immediate_terminals);
        fprintf(fp, ",%u,%u,%u,%.3f\n", ctx->stats.max_ply,
                ctx->root_pn, ctx->root_dn, rate);
#undef U64F
#undef DBLF
        fflush(fp);
    }
    frontier_snapshot(ctx, event, elapsed);
    deep_frontier_snapshot(ctx, event, elapsed);
}

static void dfpn(Context *ctx, const Board *b, uint32_t thpn, uint32_t thdn, unsigned depth);
static void reset_phase(Context *ctx, uint8_t prover, const char *name);
static int maybe_schedule_hot_exact(Context *ctx);
static int maybe_schedule_deep_reroot(Context *ctx);
static void sp_promote_exact_descendant(Context *ctx, const Board *b,
                                        uint32_t pn, uint32_t dn,
                                        const Board *path, unsigned path_len);

/* v9.8.2: deep-node PN/DN-stall reroot trigger.
 *
 * v9.5.1 used the age of one CURRENT recursive frame.  That still misses a
 * hard prefix when DF-PN repeatedly returns from depth D and immediately
 * re-enters the same profiled node: deep_active_start[D] is reset on every
 * re-entry even though that node's PN/DN pair has not changed for millions of
 * expansions.
 *
 * v9.8.2 therefore measures the currently active deep-frontier NODE by
 * deep_frontier_stall(): cumulative effective work since that node's PN/DN
 * last changed.  This survives return/re-entry.  Root PN/DN must also be
 * stalled.  Historical work shares remain diagnostics only.
 */
static uint64_t deep_active_node_stall_at(const Context *ctx, unsigned depth)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node ||
        depth == 0 || depth > ctx->deep_frontier.max_depth ||
        depth > DEEP_FRONTIER_HARD_MAX_DEPTH)
        return 0;
    int32_t idx = ctx->deep_active_node[depth];
    if (idx < 0 || (uint32_t)idx >= ctx->deep_frontier.count) return 0;
    return deep_frontier_stall(ctx, idx);
}

static double deep_frontier_node_share(Context *ctx, int32_t idx)
{
    if (!ctx->deep_frontier_enabled || !ctx->deep_frontier.node || idx < 0 ||
        (uint32_t)idx >= ctx->deep_frontier.count) return 0.0;
    DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    int32_t parent = n->parent;
    if (parent < 0 || (uint32_t)parent >= ctx->deep_frontier.count) return 0.0;
    uint64_t sum = 0;
    for (int32_t c = ctx->deep_frontier.node[parent].first_child; c >= 0;
         c = ctx->deep_frontier.node[c].next_sibling) {
        if (!ctx->deep_frontier.node[c].valid) continue;
        WorkStamp w = deep_frontier_effective_stamp(ctx, c);
        sum += w.expansions;
    }
    if (!sum) return 0.0;
    WorkStamp w = deep_frontier_effective_stamp(ctx, idx);
    return (double)w.expansions / (double)sum;
}

static void deep_reroot_node_status(const Context *ctx, unsigned *depth_out,
                                    uint64_t *node_stall_out, int32_t *idx_out)
{
    unsigned best_depth = 0;
    uint64_t best_stall = 0;
    int32_t best_idx = -1;
    if (ctx->deep_frontier_enabled && ctx->deep_frontier.node) {
        unsigned maxd = ctx->deep_frontier.max_depth;
        if (maxd > DEEP_FRONTIER_HARD_MAX_DEPTH) maxd = DEEP_FRONTIER_HARD_MAX_DEPTH;
        for (unsigned d = maxd; d >= 1; --d) {
            int32_t idx = ctx->deep_active_node[d];
            if (idx >= 0 && (uint32_t)idx < ctx->deep_frontier.count) {
                best_depth = d;
                best_idx = idx;
                best_stall = deep_active_node_stall_at(ctx, d);
                break;
            }
            if (d == 1) break;
        }
    }
    if (depth_out) *depth_out = best_depth;
    if (node_stall_out) *node_stall_out = best_stall;
    if (idx_out) *idx_out = best_idx;
}

static int32_t deep_reroot_candidate(Context *ctx, double *root_share_out,
                                     double *leaf_share_out, uint64_t *leaf_work_out,
                                     uint64_t *node_stall_out)
{
    if (!ctx->deep_reroot_enabled || !ctx->deep_frontier_enabled ||
        !ctx->deep_frontier.node || !ctx->frontier_enabled ||
        !ctx->frontier.initialized) return -1;
    if (ctx->stats.expansions < ctx->deep_reroot_min_total_exp) return -1;

    uint64_t root_stall = frontier_stall_exp(ctx);
    if (root_stall < ctx->deep_reroot_stall_exp) return -1;

    uint64_t node_need = ctx->deep_reroot_stall_exp;
    if (ctx->deep_reroot_min_work > node_need) node_need = ctx->deep_reroot_min_work;

    /* Choose the deepest CURRENTLY ACTIVE profiled node whose own PN/DN pair
       has remained unchanged for long enough.  deep_frontier_stall() uses
       cumulative effective node work, so short return/re-entry cycles do not
       reset this timer. */
    int32_t candidate = -1;
    uint64_t node_stall = 0;
    unsigned maxd = ctx->deep_frontier.max_depth;
    if (maxd > DEEP_FRONTIER_HARD_MAX_DEPTH) maxd = DEEP_FRONTIER_HARD_MAX_DEPTH;
    for (unsigned d = maxd; d >= ctx->deep_reroot_min_depth; --d) {
        int32_t idx = ctx->deep_active_node[d];
        if (idx >= 0 && (uint32_t)idx < ctx->deep_frontier.count &&
            ctx->deep_frontier.node[idx].valid) {
            uint64_t ns = deep_active_node_stall_at(ctx, d);
            if (ns >= node_need) {
                candidate = idx;
                node_stall = ns;
                break;
            }
        }
        if (d == ctx->deep_reroot_min_depth) break;
    }
    if (candidate < 0) return -1;

    DeepFrontierNode *n = &ctx->deep_frontier.node[candidate];
    WorkStamp ew = deep_frontier_effective_stamp(ctx, candidate);
    if (n->reroot_result_valid) return -1;
    if (n->reroot_attempts >= ctx->deep_reroot_max_attempts) return -1;
    if (n->reroot_attempts > 0) {
        if (ew.expansions < n->reroot_last_attempt_work) return -1;
        if (ew.expansions - n->reroot_last_attempt_work < ctx->deep_reroot_retry_work) return -1;
    }

    uint32_t pn = 1, dn = 1;
    if (tt_pair_peek(ctx, &n->board, &pn, &dn) && (pn == 0 || dn == 0)) return -1;

    double root_share = 0.0;
    int32_t root_idx = ctx->deep_active_node[1];
    if (root_idx >= 0) root_share = deep_frontier_node_share(ctx, root_idx);
    double leaf_share = deep_frontier_node_share(ctx, candidate);
    if (root_share_out) *root_share_out = root_share;
    if (leaf_share_out) *leaf_share_out = leaf_share;
    if (leaf_work_out) *leaf_work_out = ew.expansions;
    if (node_stall_out) *node_stall_out = node_stall;
    return candidate;
}

static void deep_reroot_subcontext_destroy(Context *sub);

static void deep_reroot_sessions_destroy(Context *ctx)
{
    if (!ctx) return;
    for (unsigned i = 0; i < REROOT_SESSION_MAX_SLOTS; ++i) {
        RerootSession *rs = &ctx->reroot_session[i];
        if (rs->sub_context) {
            deep_reroot_subcontext_destroy((Context *)rs->sub_context);
        }
        memset(rs, 0, sizeof(*rs));
    }
}

static void deep_reroot_subcontext_destroy(Context *sub)
{
    if (!sub) return;
    deep_reroot_sessions_destroy(sub);
    deep_frontier_destroy(sub);
    child_cache_destroy(&sub->child_cache);
    hot_tt_destroy(&sub->hot_tt);
    tt_destroy(&sub->tt);
    free(sub);
}

static void deep_reroot_set_promote_path(Context *ctx, const Board *path, unsigned path_len)
{
    if (!ctx) return;
    if (!path || path_len == 0) {
        ctx->parallel_promote_path_len = 0;
        return;
    }
    if (path_len > SP_PATH_MAX) path_len = SP_PATH_MAX;
    memcpy(ctx->parallel_promote_path, path, sizeof(Board) * path_len);
    ctx->parallel_promote_path_len = path_len;
}

/* Build the concrete path from the main SP root to a deep-frontier node.
   A reroot subcontext remembers the main-root -> subroot prefix; the local
   deep-frontier chain is appended without duplicating the subroot board. */
static unsigned deep_reroot_global_path(const Context *ctx, int32_t idx,
                                        Board out[SP_PATH_MAX])
{
    if (!ctx || !out || !ctx->deep_frontier.node || idx < 0 ||
        (uint32_t)idx >= ctx->deep_frontier.count) return 0;
    unsigned len = 0;
    if (ctx->parallel_promote_path_len > 0) {
        len = ctx->parallel_promote_path_len;
        if (len > SP_PATH_MAX) len = SP_PATH_MAX;
        memcpy(out, ctx->parallel_promote_path, sizeof(Board) * len);
    } else {
        out[len++] = ctx->deep_frontier.node[0].board;
    }

    int32_t chain[DEEP_FRONTIER_HARD_MAX_DEPTH + 2];
    unsigned n = 0;
    int32_t cur = idx;
    while (cur > 0 && n < DEEP_FRONTIER_HARD_MAX_DEPTH + 2) {
        chain[n++] = cur;
        cur = ctx->deep_frontier.node[cur].parent;
    }
    for (unsigned k = 0; k < n && len < SP_PATH_MAX; ++k) {
        int32_t ni = chain[n - 1 - k];
        const Board *b = &ctx->deep_frontier.node[ni].board;
        if (len > 0 && same_canonical_state(&out[len - 1], b)) continue;
        out[len++] = *b;
    }
    return len;
}

static Context *deep_reroot_subcontext_create(Context *parent, const Board *target,
                                               const Board *promote_path, unsigned promote_path_len)
{
    Context *sub = (Context *)calloc(1, sizeof(*sub));
    if (!sub) return NULL;
    sub->floor_row = parent->floor_row;
    sub->tss_enabled = parent->deep_reroot_tss;
    sub->tss_compress = parent->tss_compress;
    sub->tss_depth = parent->tss_depth;
    sub->tss_budget = parent->tss_budget;
    sub->tss_start_ply = parent->tss_start_ply;
    sub->tss_levels = parent->tss_levels;
    sub->tss_depth_step = parent->tss_depth_step;
    sub->tss_budget_mult = parent->tss_budget_mult;
    sub->tactical_enabled = false;
    sub->bridge_enabled = false;
    sub->dependency_enabled = parent->dependency_enabled;
    sub->dep_full_enabled = parent->dep_full_enabled;
    sub->dep_oracle_enabled = false;
    sub->legacy_quiet_order = parent->legacy_quiet_order;
    sub->or_dependency_only = parent->or_dependency_only;
    sub->child_cache_enabled = parent->deep_reroot_child_cache_mb > 0.0;
    unsigned next_level = parent->deep_reroot_level + 1u;
    bool recursive_reroot = next_level < parent->deep_reroot_max_nesting;
    sub->frontier_enabled = recursive_reroot;
    sub->frontier_top = 1;
    sub->deep_frontier_enabled = recursive_reroot;
    sub->deep_frontier_depth = parent->deep_frontier_depth;
    sub->deep_frontier_top = 1;
    sub->deep_frontier_max_nodes = parent->deep_reroot_profile_nodes;
    sub->hot_exact_enabled = false;
    sub->hot_tss_gate_enabled = false;
    sub->work_quantum = 0;
    sub->deep_reroot_enabled = recursive_reroot;
    sub->deep_reroot_memory_gb = parent->deep_reroot_memory_gb;
    sub->deep_reroot_child_cache_mb = parent->deep_reroot_child_cache_mb;
    sub->deep_reroot_time_sec = parent->deep_reroot_time_sec;
    sub->deep_reroot_min_total_exp = parent->deep_reroot_min_total_exp;
    sub->deep_reroot_stall_exp = parent->deep_reroot_stall_exp;
    sub->deep_reroot_min_work = parent->deep_reroot_min_work;
    sub->deep_reroot_retry_work = parent->deep_reroot_retry_work;
    sub->deep_reroot_hot_age_sec = parent->deep_reroot_hot_age_sec;
    sub->deep_reroot_hot_work = parent->deep_reroot_hot_work;
    sub->deep_reroot_hot_max_change = parent->deep_reroot_hot_max_change;
    sub->deep_reroot_hot_probe_sec = parent->deep_reroot_hot_probe_sec;
    sub->deep_reroot_hot_gap_sec = parent->deep_reroot_hot_gap_sec;
    sub->deep_reroot_hot_min_busy = parent->deep_reroot_hot_min_busy;
    sub->deep_reroot_hot_region_depth = parent->deep_reroot_hot_region_depth;
    sub->deep_reroot_root_share = parent->deep_reroot_root_share;
    sub->deep_reroot_inner_share = parent->deep_reroot_inner_share;
    sub->deep_reroot_min_depth = parent->deep_reroot_min_depth;
    sub->deep_reroot_max_attempts = parent->deep_reroot_max_attempts;
    sub->deep_reroot_max_nesting = parent->deep_reroot_max_nesting;
    sub->deep_reroot_session_slots = parent->deep_reroot_session_slots;
    sub->deep_reroot_profile_nodes = parent->deep_reroot_profile_nodes;
    sub->deep_reroot_focus_slices = parent->deep_reroot_focus_slices;
    sub->deep_reroot_harvest_enabled = parent->deep_reroot_harvest_enabled;
    sub->deep_reroot_harvest_after_slices = parent->deep_reroot_harvest_after_slices;
    sub->deep_reroot_harvest_min_depth = parent->deep_reroot_harvest_min_depth;
    sub->deep_reroot_harvest_max_depth = parent->deep_reroot_harvest_max_depth;
    sub->deep_reroot_harvest_attempts = parent->deep_reroot_harvest_attempts;
    sub->deep_reroot_harvest_focus_slices = parent->deep_reroot_harvest_focus_slices;
    sub->deep_reroot_harvest_min_work = parent->deep_reroot_harvest_min_work;
    sub->deep_reroot_harvest_deep_min_work = parent->deep_reroot_harvest_deep_min_work;
    sub->deep_reroot_harvest_deep_min_stability = parent->deep_reroot_harvest_deep_min_stability;
    sub->deep_reroot_harvest_resume_pressure = parent->deep_reroot_harvest_resume_pressure;
    sub->deep_reroot_harvest_time_sec = parent->deep_reroot_harvest_time_sec;
    sub->deep_reroot_level = next_level;
    sub->deep_reroot_tss = parent->deep_reroot_tss;
    sub->dep_start_ply = parent->dep_start_ply;
    sub->dep_full_start_ply = parent->dep_full_start_ply;
    sub->dep_full_tss_level = parent->dep_full_tss_level;
    sub->dep_dead_empty = parent->dep_dead_empty;
    sub->dep_deadcheck_ply = parent->dep_deadcheck_ply;
    sub->report_interval = 1.0e9;
    sub->parallel_stop_shared = parent->parallel_stop_shared;
    sub->parallel_exact_shared = parent->parallel_exact_shared;
    deep_reroot_set_promote_path(sub, promote_path, promote_path_len);

    if (!tt_init(&sub->tt, parent->deep_reroot_memory_gb)) {
        free(sub);
        return NULL;
    }
    if (sub->child_cache_enabled &&
        !child_cache_init(&sub->child_cache, parent->deep_reroot_child_cache_mb)) {
        tt_destroy(&sub->tt);
        free(sub);
        return NULL;
    }
    if (sub->deep_frontier_enabled && !deep_frontier_init(sub, target)) {
        child_cache_destroy(&sub->child_cache);
        tt_destroy(&sub->tt);
        free(sub);
        return NULL;
    }

    /* Exactly once for this persistent session.  In particular, do NOT call
       reset_phase() again on resume: generation must remain unchanged so the
       retained TT entries stay visible. */
    reset_phase(sub, parent->prover, "RR");
    if (sub->deep_frontier_enabled) deep_frontier_reset_phase(sub, target);
    return sub;
}

static RerootSession *deep_reroot_session_acquire(Context *parent,
                                                   const Board *target,
                                                   const Board *promote_path,
                                                   unsigned promote_path_len,
                                                   bool *resumed_out)
{
    uint64_t k[4]; bool mirrored;
    canonical_key(target, k, &mirrored); (void)mirrored;
    parent->reroot_session_tick++;
    if (parent->reroot_session_tick == 0) parent->reroot_session_tick = 1;

    unsigned slots = parent->deep_reroot_session_slots;
    if (slots < 1 || slots > REROOT_SESSION_MAX_SLOTS) slots = REROOT_SESSION_DEFAULT_SLOTS;
    for (unsigned i = 0; i < slots; ++i) {
        RerootSession *rs = &parent->reroot_session[i];
        if (!rs->valid || rs->prover != parent->prover) continue;
        if (rs->k[0] == k[0] && rs->k[1] == k[1] &&
            rs->k[2] == k[2] && rs->k[3] == k[3] &&
            rs->root.p[0] == target->p[0] && rs->root.p[1] == target->p[1] &&
            rs->root.twos == target->twos && rs->root.threes == target->threes &&
            rs->root.turn == target->turn) {
            rs->uses++;
            rs->last_used_tick = parent->reroot_session_tick;
            parent->stats.deep_reroot_resumes++;
            if (rs->sub_context) {
                Context *sub = (Context *)rs->sub_context;
                sub->parallel_exact_shared = parent->parallel_exact_shared;
                sub->parallel_stop_shared = parent->parallel_stop_shared;
                deep_reroot_set_promote_path(sub, promote_path, promote_path_len);
            }
            if (resumed_out) *resumed_out = true;
            return rs;
        }
    }

    unsigned slot = slots;
    for (unsigned i = 0; i < slots; ++i) {
        if (!parent->reroot_session[i].valid) { slot = i; break; }
    }
    if (slot == slots) {
        slot = 0;
        for (unsigned i = 1; i < slots; ++i) {
            if (parent->reroot_session[i].last_used_tick <
                parent->reroot_session[slot].last_used_tick) slot = i;
        }
        RerootSession *old = &parent->reroot_session[slot];
        if (old->sub_context) deep_reroot_subcontext_destroy((Context *)old->sub_context);
        memset(old, 0, sizeof(*old));
        parent->stats.deep_reroot_session_evictions++;
    }

    Context *sub = deep_reroot_subcontext_create(parent, target, promote_path, promote_path_len);
    if (!sub) return NULL;
    RerootSession *rs = &parent->reroot_session[slot];
    memset(rs, 0, sizeof(*rs));
    rs->valid = true;
    rs->prover = parent->prover;
    memcpy(rs->k, k, sizeof(k));
    rs->root = *target;
    rs->sub_context = sub;
    rs->uses = 1;
    rs->last_used_tick = parent->reroot_session_tick;
    parent->stats.deep_reroot_fresh_sessions++;
    if (resumed_out) *resumed_out = false;
    return rs;
}

/* Return an already-created persistent subcontext without changing LRU/use
   counters.  The SP outer reroot has just called persistent_solve(), so this
   is used only to inspect its retained private MPN/deep-frontier state. */
static Context *deep_reroot_session_context(Context *parent, const Board *target)
{
    if (!parent || !target) return NULL;
    uint64_t k[4]; bool mirrored;
    canonical_key(target, k, &mirrored); (void)mirrored;
    unsigned slots = parent->deep_reroot_session_slots;
    if (slots < 1 || slots > REROOT_SESSION_MAX_SLOTS) slots = REROOT_SESSION_DEFAULT_SLOTS;
    for (unsigned i = 0; i < slots; ++i) {
        RerootSession *rs = &parent->reroot_session[i];
        if (!rs->valid || !rs->sub_context || rs->prover != parent->prover) continue;
        if (rs->k[0] == k[0] && rs->k[1] == k[1] &&
            rs->k[2] == k[2] && rs->k[3] == k[3] &&
            rs->root.p[0] == target->p[0] && rs->root.p[1] == target->p[1] &&
            rs->root.twos == target->twos && rs->root.threes == target->threes &&
            rs->root.turn == target->turn)
            return (Context *)rs->sub_context;
    }
    return NULL;
}

static int deep_reroot_persistent_solve(Context *parent, const Board *target,
                                        const Board *promote_path, unsigned promote_path_len,
                                        uint32_t *out_pn, uint32_t *out_dn,
                                        uint64_t *out_calls, uint64_t *out_exp,
                                        uint64_t *out_tt_probes, uint64_t *out_tt_hits,
                                        double *out_sec, bool *out_resumed,
                                        uint64_t *out_session_uses)
{
    bool resumed = false;
    RerootSession *rs = deep_reroot_session_acquire(parent, target, promote_path, promote_path_len, &resumed);
    if (!rs || !rs->sub_context) return -2;
    Context *sub = (Context *)rs->sub_context;

    uint64_t before_calls = sub->stats.dfpn_calls + sub->stats.deep_reroot_sub_calls;
    uint64_t before_exp = sub->stats.expansions + sub->stats.deep_reroot_sub_expansions;
    uint64_t before_tt_probes = sub->stats.tt_probes;
    uint64_t before_tt_hits = sub->stats.tt_hits;

    double start = wall_time();
    sub->stop = false;
    sub->deep_reroot_unwind = false;
    sub->deadline = parent->deep_reroot_time_sec > 0.0 ?
        start + parent->deep_reroot_time_sec : 0.0;
    if (parent->deadline > 0.0 &&
        (sub->deadline <= 0.0 || parent->deadline < sub->deadline))
        sub->deadline = parent->deadline;
    sub->next_report = start + 1.0e9;

    dfpn(sub, target, PN_INF, PN_INF, 0);
    uint32_t pn = 1, dn = 1;
    tt_pair(sub, target, &pn, &dn);

    double sec = wall_time() - start;
    uint64_t after_calls = sub->stats.dfpn_calls + sub->stats.deep_reroot_sub_calls;
    uint64_t after_exp = sub->stats.expansions + sub->stats.deep_reroot_sub_expansions;
    uint64_t calls = after_calls >= before_calls ? after_calls - before_calls : 0;
    uint64_t exp = after_exp >= before_exp ? after_exp - before_exp : 0;
    uint64_t tt_probes = sub->stats.tt_probes >= before_tt_probes ?
        sub->stats.tt_probes - before_tt_probes : 0;
    uint64_t tt_hits = sub->stats.tt_hits >= before_tt_hits ?
        sub->stats.tt_hits - before_tt_hits : 0;

    if (out_pn) *out_pn = pn;
    if (out_dn) *out_dn = dn;
    if (out_calls) *out_calls = calls;
    if (out_exp) *out_exp = exp;
    if (out_tt_probes) *out_tt_probes = tt_probes;
    if (out_tt_hits) *out_tt_hits = tt_hits;
    if (out_sec) *out_sec = sec;
    if (out_resumed) *out_resumed = resumed;
    if (out_session_uses) *out_session_uses = rs->uses;

    if (pn == 0) return 1;
    if (dn == 0) return 0;
    return -1;
}

static int maybe_schedule_deep_reroot(Context *ctx)
{
    double root_share = 0.0, leaf_share = 0.0;
    uint64_t leaf_work = 0, node_stall = 0;
    int32_t idx = deep_reroot_candidate(ctx, &root_share, &leaf_share, &leaf_work,
                                        &node_stall);
    if (idx < 0) return -1;

    DeepFrontierNode *n = &ctx->deep_frontier.node[idx];
    n->reroot_attempts++;  /* focus episodes for this deep-prefix node */
    /* Retry spacing remains cumulative node work.  v9.8.2 deliberately does
       NOT require new main work between the individual slices inside one
       focus episode: UNKNOWN slices immediately resume the same session. */
    n->reroot_last_attempt_work = leaf_work;
    ctx->stats.deep_reroot_triggers++;
    ctx->stats.deep_reroot_attempts++;

    unsigned focus_slices = ctx->deep_reroot_focus_slices;
    if (focus_slices < 1) focus_slices = 1;

    char path[160];
    deep_frontier_path_text(ctx, idx, path, sizeof(path));
    printf("[deep-reroot] trigger level=%u path=%s depth=%u nodestall=%" PRIu64
           " rootstall=%" PRIu64 " work=%" PRIu64
           " diagShare=%.1f%%/%.1f%% episode=%u/%u focus=%u x %.1fs"
           " TT=%.3fGiB TSS=%s\n",
           ctx->deep_reroot_level, path, (unsigned)n->depth, node_stall,
           frontier_stall_exp(ctx), leaf_work,
           100.0 * root_share, 100.0 * leaf_share,
           (unsigned)n->reroot_attempts, ctx->deep_reroot_max_attempts,
           focus_slices, ctx->deep_reroot_time_sec,
           ctx->deep_reroot_memory_gb,
           ctx->deep_reroot_tss ? "on" : "off");
    fflush(stdout);

    Board promote_path[SP_PATH_MAX];
    unsigned promote_path_len = 0;
    if (ctx->parallel_exact_shared)
        promote_path_len = deep_reroot_global_path(ctx, idx, promote_path);

    int episode_result = -1;
    unsigned slices_done = 0;
    for (unsigned slice = 1; slice <= focus_slices; ++slice) {
        if (g_interrupted) {
            ctx->stop = true;
            break;
        }
        if (ctx->deadline > 0.0 && wall_time() >= ctx->deadline) {
            ctx->stop = true;
            break;
        }

        uint32_t pn = 1, dn = 1;
        uint64_t calls = 0, exp = 0, tt_probes = 0, tt_hits = 0, session_uses = 0;
        double sec = 0.0;
        bool resumed = false;
        int result = deep_reroot_persistent_solve(ctx, &n->board,
                                                   promote_path_len ? promote_path : NULL,
                                                   promote_path_len,
                                                   &pn, &dn, &calls, &exp, &tt_probes, &tt_hits,
                                                   &sec, &resumed, &session_uses);
        slices_done++;
        ctx->stats.deep_reroot_slices++;
        if (slice > 1) ctx->stats.deep_reroot_focus_extra_slices++;
        ctx->stats.deep_reroot_sub_calls += calls;
        ctx->stats.deep_reroot_sub_expansions += exp;
        ctx->stats.deep_reroot_sub_ms += (uint64_t)(sec * 1000.0 + 0.5);
        ctx->stats.deep_reroot_sub_tt_probes += tt_probes;
        ctx->stats.deep_reroot_sub_tt_hits += tt_hits;

        double slice_tt = tt_probes ? 100.0 * (double)tt_hits / (double)tt_probes : 0.0;
        if (result >= 0) {
            n->reroot_result_valid = true;
            n->reroot_result = (uint8_t)result;
            reroot_overlay_store(ctx, &n->board, pn, dn);
            ctx->stats.deep_reroot_proofs++;
            if (result == 1) ctx->stats.deep_reroot_proved++;
            else ctx->stats.deep_reroot_disproved++;
            ctx->stats.oracle_collapses++;
            ctx->deep_reroot_unwind = true;
            if (ctx->parallel_exact_shared && promote_path_len > 0)
                sp_promote_exact_descendant(ctx, &n->board, pn, dn,
                                            promote_path, promote_path_len);
            printf("[deep-reroot] EXACT path=%s focus=%u/%u mode=%s use=%" PRIu64
                   " proposition=%s pn/dn=(%u,%u) calls=%" PRIu64
                   " exp=%" PRIu64 " tt=%.1f%% sec=%.3f -> unwind\n",
                   path, slice, focus_slices, resumed ? "resume" : "fresh", session_uses,
                   result == 1 ? "PROVED" : "DISPROVED", pn, dn, calls, exp,
                   slice_tt, sec);
            fflush(stdout);
            return result;
        }

        if (result == -2) {
            printf("[deep-reroot] ERROR path=%s focus=%u/%u allocation failure\n",
                   path, slice, focus_slices);
            fprintf(stderr, "[deep-reroot] allocation failure; disabling deep reroot.\n");
            ctx->deep_reroot_enabled = false;
            episode_result = -2;
            break;
        }

        ctx->stats.deep_reroot_slice_unknown++;
        printf("[deep-reroot] UNKNOWN path=%s focus=%u/%u mode=%s use=%" PRIu64
               " pn/dn=(%u,%u) calls=%" PRIu64 " exp=%" PRIu64
               " tt=%.1f%% sec=%.3f (session retained%s)\n",
               path, slice, focus_slices, resumed ? "resume" : "fresh", session_uses,
               pn, dn, calls, exp, slice_tt, sec,
               slice < focus_slices ? "; immediate resume" : "");
        fflush(stdout);

        /* The parent/global deadline always dominates the focus policy. */
        if (ctx->deadline > 0.0 && wall_time() >= ctx->deadline) {
            ctx->stop = true;
            break;
        }
    }

    /* An episode is UNKNOWN only once, regardless of how many persistent
       slices were consumed inside it.  This keeps rr(p/a/u) backward-compatible
       while rrSl reports slice-level behaviour. */
    ctx->stats.deep_reroot_unknown++;
    if (episode_result != -2 && slices_done >= focus_slices)
        ctx->stats.deep_reroot_focus_exhausted++;
    printf("[deep-reroot] FOCUS-END path=%s result=UNKNOWN slices=%u/%u"
           " session retained; return to parent search\n",
           path, slices_done, focus_slices);
    fflush(stdout);
    return episode_result;
}

static void maybe_stop_or_report(Context *ctx)
{
    if (g_interrupted) ctx->stop = true;
    if (ctx->parallel_shared && atomic_load_explicit(&ctx->parallel_shared->solved, memory_order_acquire) != 0)
        ctx->stop = true;
    if (ctx->parallel_stop_shared && atomic_load_explicit(&ctx->parallel_stop_shared->solved, memory_order_acquire) != 0)
        ctx->stop = true;
    if ((ctx->stats.dfpn_calls & UINT64_C(0x7ff)) != 0) return;
    double now = wall_time();
    if (ctx->deadline > 0.0 && now >= ctx->deadline) ctx->stop = true;

    /* v9.8.2: check deep-node PN/DN stall at a regular safe checkpoint, not
       only when the root recursion happens to return.  This is essential for
       2B/2C, where a single root-child descent can consume tens of millions of
       expansions.  maybe_hot_exact() remains exact-only; UNKNOWN is ignored. */
    if (!ctx->stop) (void)maybe_schedule_deep_reroot(ctx);
    if (!ctx->stop && !ctx->deep_reroot_unwind) (void)maybe_schedule_hot_exact(ctx);

    if (now >= ctx->next_report) {
        log_stats(ctx, "periodic");
        ctx->next_report = now + ctx->report_interval;
    }
}

/* Tactical status: -1 unresolved, 0/1 forced winner, 2 draw. */
static int tactical_status(Context *ctx, const Board *b, int *forced_col)
{
    ThreatInfo own = immediate_info(b, b->turn);
    if (own.colmask) {
        ctx->stats.immediate_terminals++;
        return b->turn;
    }
    ThreatInfo opp = immediate_info(b, b->turn ^ 1u);
    unsigned nopp = popcount7(opp.colmask);
    if (nopp >= 2) {
        ctx->stats.double_threat_losses++;
        return b->turn ^ 1u;
    }
    if (board_full(b)) return 2;
    if (nopp == 1) {
        ctx->stats.forced_defenses++;
        for (int c = 0; c < COLS; ++c) {
            if (opp.colmask & (1u << c)) { *forced_col = c; break; }
        }
    } else {
        *forced_col = -1;
    }
    return -1;
}

static uint8_t move_priority(const Board *child, int mover, int tile, int col, ThreatInfo threats)
{
    (void)mover; (void)tile; (void)col; (void)child;
    unsigned nt = popcount7(threats.colmask);
    if (nt >= 2) return 0;
    if (threats.best_class == WIN_322) return 1;
    if (threats.best_class == WIN_2221) return 2;
    if (threats.best_class == WIN_OTHER) return 3;
    /*
     * v6: all quiet moves share one tactical class.  In v5 every quiet move
     * received an almost-unique 10..30 priority before dep_score was compared,
     * which accidentally made dependency ordering/hints nearly inert.  Quiet
     * moves are now ordered primarily by dep_score; center/tile preference is
     * already included as the final small term in the dependency scores.
     */
    return 10;
}

static uint8_t legacy_quiet_priority(int tile, int col)
{
    int tile_rank = (tile == 2) ? 0 : (tile == 3 ? 1 : 2);
    return (uint8_t)(10 + CENTER_RANK[col] * 3 + tile_rank);
}

static int fallback_quiet_score(int tile, int col)
{
    /* Used only when dependency ordering is disabled/not yet active. */
    int tile_score = (tile == 2) ? 3 : (tile == 3 ? 2 : 1);
    return (6 - CENTER_RANK[col]) * 8 + tile_score;
}

/*
 * Heuristic only: score how strongly the just-played stone participates in
 * contiguous sub-runs that are one legal tile (1..3) short of seven.
 *
 * This score is NEVER used for an exact pruning decision.  v3 uses it only
 * to choose which attacker's quiet setup moves are attempted by the
 * selective bridge oracle.  Therefore a low score cannot make DF-PN miss a
 * legal move; failure of the oracle simply falls back to the exact search.
 */
static unsigned setup_relevance(const Board *b, int player, int col, int row)
{
    static const int dx[4] = {1, 0, 1, 1};
    static const int dy[4] = {0, 1, 1, -1};
    unsigned score = 0;

    for (int d = 0; d < 4; ++d) {
        int vals[13], cs[13], rs[13];
        int len = 0, center = 0;
        int lvals[6], lcs[6], lrs[6], nl = 0;

        for (int step = 1; step <= 6; ++step) {
            int c = col - dx[d] * step;
            int r = row - dy[d] * step;
            if (c < 0 || c >= COLS || r < 0 || r >= MAX_ROWS) break;
            if (owner_at(b, c, r) != player) break;
            lvals[nl] = tile_value_at(b, c, r);
            lcs[nl] = c; lrs[nl] = r; nl++;
        }
        for (int i = nl - 1; i >= 0; --i) {
            vals[len] = lvals[i]; cs[len] = lcs[i]; rs[len] = lrs[i]; len++;
        }
        center = len;
        vals[len] = tile_value_at(b, col, row);
        cs[len] = col; rs[len] = row; len++;

        for (int step = 1; step <= 6; ++step) {
            int c = col + dx[d] * step;
            int r = row + dy[d] * step;
            if (c < 0 || c >= COLS || r < 0 || r >= MAX_ROWS) break;
            if (owner_at(b, c, r) != player) break;
            vals[len] = tile_value_at(b, c, r);
            cs[len] = c; rs[len] = r; len++;
        }

        for (int l = 0; l <= center; ++l) {
            int sum = 0;
            for (int rix = l; rix < len; ++rix) {
                sum += vals[rix];
                if (sum >= 7) break;
                if (rix < center) continue;
                int deficit = 7 - sum;
                if (deficit < 1 || deficit > 3) continue;

                unsigned base = sum >= 6 ? 10u : sum == 5 ? 7u : sum == 4 ? 4u : 1u;
                bool open = false;

                if (l == 0) {
                    int ec = cs[l] - dx[d], er = rs[l] - dy[d];
                    if (ec >= 0 && ec < COLS && er >= 0 && er < MAX_ROWS &&
                        owner_at(b, ec, er) < 0 &&
                        (deficit != 3 || (bit_at(ec, er) & THREE_MASK))) {
                        open = true;
                        score += base;
                        if (b->height[ec] == er) score += 3;
                    }
                }
                if (rix == len - 1) {
                    int ec = cs[rix] + dx[d], er = rs[rix] + dy[d];
                    if (ec >= 0 && ec < COLS && er >= 0 && er < MAX_ROWS &&
                        owner_at(b, ec, er) < 0 &&
                        (deficit != 3 || (bit_at(ec, er) & THREE_MASK))) {
                        open = true;
                        score += base;
                        if (b->height[ec] == er) score += 3;
                    }
                }
                if (open && deficit == 2) score += 1; /* 2 is unrestricted and tactically dense. */
            }
        }
    }
    return score > 255u ? 255u : score;
}

static int child_cmp(const void *a_, const void *b_)
{
    const Child *a = (const Child *)a_;
    const Child *b = (const Child *)b_;
    if (a->priority != b->priority) return (int)a->priority - (int)b->priority;
    if (a->dep_score != b->dep_score) return (int)b->dep_score - (int)a->dep_score;
    return (int)a->move - (int)b->move;
}

/*
 * Exact move pruning:
 * - if opponent has one immediate winning column, only that column can defend;
 * - if there is any move that leaves no opponent immediate win, discard moves that lose in one;
 * - symmetric root/subpositions use one representative of mirrored quiet moves.
 */
static int generate_children(Context *ctx, const Board *b, Child out[MAX_MOVES],
                             uint8_t tt_move, int *status, bool first_expansion,
                             TTEntry *entry, bool mirrored)
{
    /*
     * v7 ordering policy:
     *   default: v6 center/tile-fast ordering (already ~98-99% first-pick top3);
     *   experimental: dependency-first on all quiet nodes or OR nodes only.
     * The dependency heuristic never removes a legal move.
     */
    bool dep_order_here = ctx->dependency_enabled && !ctx->legacy_quiet_order &&
                          (!ctx->or_dependency_only || b->turn == ctx->prover);
    bool use_light = dep_order_here && first_expansion && b->ply >= ctx->dep_start_ply;
    bool hint_ready = dep_order_here && (entry->flags & TT_FLAG_DEP_HINT_READY) != 0;
    bool full_hint_ready = dep_order_here && (entry->flags & TT_FLAG_DEP_FULL_HINT_READY) != 0;
    bool build_full = dep_order_here && ctx->dep_full_enabled &&
                      b->ply >= ctx->dep_full_start_ply &&
                      (!ctx->tss_enabled || entry->tss_level >= ctx->dep_full_tss_level) &&
                      !full_hint_ready;

    uint8_t dep_hints[4] = {0,0,0,0};
    if (hint_ready) {
        for (unsigned r = 0; r < 4; ++r) dep_hints[r] = tt_dep_hint_at(entry, mirrored, r);
        if (dep_hints[0]) ctx->stats.dep_hint_uses++;
    }

    /*
     * Hot-node generation cache.  If a full dependency hint has just become
     * due, deliberately bypass the cache once so that the new ranking can be
     * built and then replace the cached base order.
     */
    if (!build_full) {
        bool cmir = false;
        ChildCacheEntry *ce = child_cache_lookup(ctx, b, &cmir);
        if (ce) {
            *status = ce->status;
            if (*status != -1) return 0;
            int n = (int)ce->count;
            for (int i = 0; i < n; ++i) {
                uint8_t mv = cmir ? mirror_move(ce->move[i]) : ce->move[i];
                Board child = *b;
                if (!board_drop(&child, move_col(mv), move_tile(mv))) {
                    /* Full-key equality should make this impossible. Treat as
                       a cache miss rather than risking correctness. */
                    n = -1;
                    break;
                }
                out[i].board = child;
                out[i].move = mv;
                out[i].priority = 10;
                out[i].dep_score = 0;
                out[i].terminal_winner = ce->terminal[i];
            }
            if (n >= 0) {
                ctx->stats.child_cache_restored += (uint64_t)n;
                if (dep_order_here && hint_ready)
                    promote_dep_hints_in_quiet(out, n, (int)ce->quiet_start, dep_hints);
                promote_move_stable(out, n, tt_move);
                return n;
            }
        }
    }

    int forced_col = -1;
    *status = tactical_status(ctx, b, &forced_col);
    if (*status != -1) {
        child_cache_store(ctx, b, *status, out, 0, 0);
        return 0;
    }

    DependencyMap dm;
    if (use_light) ctx->stats.dep_light_nodes++;
    if (build_full) dependency_map_build(ctx, b, &dm);

    Child tmp[MAX_MOVES];
    int n = 0;
    bool symmetric = (forced_col < 0) && board_symmetric(b);

    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        if (forced_col >= 0 && col != forced_col) continue;
        if (symmetric && col > 3) {
            ctx->stats.symmetry_pruned += 3;
            continue;
        }
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            Board child = *b;
            int mover = b->turn;
            board_drop(&child, col, tile);

            ThreatInfo opp = immediate_info(&child, child.turn);
            bool unsafe = opp.colmask != 0;

            ThreatInfo mine = immediate_info(&child, mover);
            Child ch;
            ch.board = child;
            ch.move = move_encode(tile, col);
            ch.priority = move_priority(&child, mover, tile, col, mine);
            if (!dep_order_here && ch.priority == 10)
                ch.priority = legacy_quiet_priority(tile, col);
            if (build_full)
                ch.dep_score = (int16_t)dependency_move_score(b, &child, mover, col, tile, &dm);
            else if (use_light)
                ch.dep_score = (int16_t)dependency_light_score(ctx, b, &child, mover, col, tile);
            else
                ch.dep_score = (int16_t)fallback_quiet_score(tile, col);

            if (hint_ready && ch.priority >= 10 && dep_order_here) {
                static const int bonus[4] = {12000, 8000, 5000, 2500};
                for (unsigned r = 0; r < 4; ++r) {
                    if (dep_hints[r] && ch.move == dep_hints[r]) {
                        int boosted = (int)ch.dep_score + bonus[r];
                        ch.dep_score = (int16_t)(boosted > 32767 ? 32767 : boosted);
                        break;
                    }
                }
            }
            if (build_full || use_light || hint_ready) ctx->stats.dep_ordered_moves++;

            ch.terminal_winner = -1;
            if (popcount7(mine.colmask) >= 2 && !unsafe) {
                ch.terminal_winner = (int8_t)mover;
                ctx->stats.double_threat_wins++;
            } else if (board_full(&child) && !unsafe) {
                ch.terminal_winner = 2;
            }
            if (unsafe) ch.priority = UINT8_C(250);
            tmp[n++] = ch;
        }
    }

    ctx->stats.child_generations++;
    ctx->stats.moves_generated += (uint64_t)n;

    int safe_count = 0;
    for (int i = 0; i < n; ++i) if (tmp[i].priority != UINT8_C(250)) safe_count++;
    if (safe_count == 0) {
        *status = b->turn ^ 1u;
        child_cache_store(ctx, b, *status, out, 0, 0);
        return 0;
    }

    int m = 0;
    uint64_t seen_keys[MAX_MOVES][4];
    for (int i = 0; i < n; ++i) {
        if (tmp[i].priority == UINT8_C(250)) {
            ctx->stats.unsafe_pruned++;
            continue;
        }

        uint64_t key[4]; bool kmir;
        canonical_key(&tmp[i].board, key, &kmir);
        (void)kmir;
        bool dup = false;
        for (int j = 0; j < m; ++j) {
            if (memcmp(seen_keys[j], key, sizeof(key)) == 0) {
                dup = true;
                break;
            }
        }
        if (dup) {
            ctx->stats.sibling_dedup_pruned++;
            continue;
        }
        memcpy(seen_keys[m], key, sizeof(key));
        out[m++] = tmp[i];
    }

    /* Cache top dependency hints for experimental dependency modes. */
    if ((build_full || use_light) && m > 0 && dep_order_here) {
        int idxs[MAX_MOVES];
        for (int i = 0; i < m; ++i) idxs[i] = i;
        for (int i = 1; i < m; ++i) {
            int x = idxs[i], j = i;
            while (j > 0 && out[idxs[j-1]].dep_score < out[x].dep_score) {
                idxs[j] = idxs[j-1]; --j;
            }
            idxs[j] = x;
        }
        uint8_t top[4] = {0,0,0,0};
        int lim = m < 4 ? m : 4;
        for (int r = 0; r < lim; ++r) top[r] = out[idxs[r]].move;
        tt_set_dep_hints(entry, mirrored, top, build_full);
    }

    /* Stable base order first; TT principal move is promoted after caching. */
    qsort(out, (size_t)m, sizeof(out[0]), child_cmp);
    int quiet_start = m;
    for (int i = 0; i < m; ++i) {
        if (out[i].priority >= 10) { quiet_start = i; break; }
    }
    child_cache_store(ctx, b, -1, out, m, quiet_start);
    promote_move_stable(out, m, tt_move);
    return m;
}


static void proposition_pair(int winner, uint8_t prover, uint32_t *pn, uint32_t *dn);

/* v9.8.3 Exact Lemma Closure Oracle.
 * Return 1 when the current prover is exactly PROVED, 0 when exactly DISPROVED,
 * and -1 when the available exact lemmas are insufficient within depth/budget.
 * No finite PN/DN value is ever consumed as evidence. */
static int exact_lemma_closure(Context *ctx, const Board *b, unsigned depth_left,
                               uint64_t *budget, uint32_t *out_pn, uint32_t *out_dn)
{
    if (!ctx->exact_db || !budget || *budget == 0 || ctx->stop) return -1;
    (*budget)--;

    uint32_t pn=1,dn=1;
    if (exact_db_lookup_pair(ctx->exact_db,b,ctx->prover,&pn,&dn)) {
        if (out_pn) *out_pn=pn;
        if (out_dn) *out_dn=dn;
        return pn==0 ? 1 : dn==0 ? 0 : -1;
    }

    bool hit=false, mirrored=false;
    TTEntry *entry=tt_access(ctx,b,true,&hit,&mirrored);
    if (entry->pn==0 || entry->dn==0) {
        if (out_pn) *out_pn=entry->pn;
        if (out_dn) *out_dn=entry->dn;
        return entry->pn==0 ? 1 : 0;
    }
    if (depth_left==0) return -1;

    uint8_t tt_move=tt_best_move(ctx,entry,mirrored);
    Child children[MAX_MOVES]; int status=-1;
    int n=generate_children(ctx,b,children,tt_move,&status,false,entry,mirrored);
    if (status!=-1) {
        proposition_pair(status,ctx->prover,&pn,&dn);
        entry->pn=pn; entry->dn=dn; sp_context_publish(ctx,b,pn,dn);
        if(out_pn)*out_pn=pn;
        if(out_dn)*out_dn=dn;
        return pn==0?1:0;
    }
    if (n<=0) {
        proposition_pair(2,ctx->prover,&pn,&dn);
        entry->pn=pn; entry->dn=dn; sp_context_publish(ctx,b,pn,dn);
        if(out_pn)*out_pn=pn;
        if(out_dn)*out_dn=dn;
        return pn==0?1:0;
    }

    const bool or_node=(b->turn==ctx->prover);
    bool saw_unknown=false;
    for(int i=0;i<n;++i){
        uint32_t cpn=1,cdn=1; int cr=-1;
        if(children[i].terminal_winner>=0){
            proposition_pair(children[i].terminal_winner,ctx->prover,&cpn,&cdn);
            cr=cpn==0?1:0;
        } else {
            cr=exact_lemma_closure(ctx,&children[i].board,depth_left-1,budget,&cpn,&cdn);
        }
        if(ctx->stop || *budget==0) saw_unknown=true;
        if(or_node && cr==1){ pn=0; dn=PN_INF; goto exact_done; }
        if(!or_node && cr==0){ pn=PN_INF; dn=0; goto exact_done; }
        if(cr<0) saw_unknown=true;
    }
    if(saw_unknown) return -1;
    if(or_node){ pn=PN_INF; dn=0; }
    else { pn=0; dn=PN_INF; }

exact_done:
    entry=tt_access(ctx,b,true,&hit,&mirrored);
    entry->pn=pn; entry->dn=dn; sp_context_publish(ctx,b,pn,dn);
    if(out_pn)*out_pn=pn;
        if(out_dn)*out_dn=dn;
    return pn==0?1:0;
}


/* Sound but incomplete threat-space proof search. False means "not proven within this TSS". */
static bool tss_prove(Context *ctx, const Board *b, uint8_t attacker,
                      unsigned depth_left, uint64_t *budget);

static bool tss_prove_impl(Context *ctx, const Board *b, uint8_t attacker,
                           unsigned depth_left, uint64_t *budget)
{
    if (g_interrupted) ctx->stop = true;
    if (*budget == 0 || depth_left == 0 || ctx->stop) return false;
    (*budget)--;
    ctx->stats.tss_nodes++;
    if ((ctx->stats.tss_nodes & UINT64_C(0x3ff)) == 0) {
        double now = wall_time();
        if (ctx->deadline > 0.0 && now >= ctx->deadline) {
            ctx->stop = true;
            return false;
        }
    }

    ThreatInfo own = immediate_info(b, b->turn);
    if (own.colmask) return b->turn == attacker;

    ThreatInfo opp = immediate_info(b, b->turn ^ 1u);
    unsigned nopp = popcount7(opp.colmask);
    if (nopp >= 2) return (b->turn ^ 1u) == attacker;
    if (board_full(b)) return false;

    if (nopp == 1) {
        int col = 0;
        while (!(opp.colmask & (1u << col))) ++col;
        if (b->turn == attacker) {
            /* Attacker is forced to defend; finding one continuing proof is enough. */
            for (int tile = 1; tile <= 3; ++tile) {
                if (!legal_tile_at(b, col, tile)) continue;
                Board child = *b;
                board_drop(&child, col, tile);
                if (tss_prove(ctx, &child, attacker, depth_left - 1, budget)) return true;
                if (ctx->stop || *budget == 0) return false;
            }
            return false;
        } else {
            /* Defender is forced into this column; proof must survive every legal tile choice. */
            bool had = false;
            for (int tile = 1; tile <= 3; ++tile) {
                if (!legal_tile_at(b, col, tile)) continue;
                had = true;
                Board child = *b;
                board_drop(&child, col, tile);
                if (!tss_prove(ctx, &child, attacker, depth_left - 1, budget)) return false;
                if (ctx->stop || *budget == 0) return false;
            }
            return had;
        }
    }

    /* Without an existing forcing threat, only attacker moves can remain in TSS. */
    if (b->turn != attacker) return false;

    Child candidates[MAX_MOVES];
    uint16_t signatures[MAX_MOVES];
    int n = 0;
    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            Board child = *b;
            board_drop(&child, col, tile);
            /* A forcing move must not give defender a win in one. */
            if (immediate_info(&child, child.turn).colmask) continue;
            ThreatInfo mine = immediate_info(&child, attacker);
            if (!mine.colmask) continue;

            ctx->stats.tss_forcing_candidates++;
            Child ch = {0};
            ch.board = child;
            ch.move = move_encode(tile, col);
            ch.priority = move_priority(&child, attacker, tile, col, mine);
            ch.terminal_winner = -1;

            /*
             * Selective proof-search compression: attacker branches that create
             * the same immediate threat-column set and best pattern class are
             * represented by the best-ordered move.  Defender replies are
             * never compressed.  Because omitted ATTACKER moves can only make
             * this incomplete oracle miss a proof (never invent one), the
             * exact DF-PN fallback remains sound.
             */
            uint16_t sig = (uint16_t)mine.colmask |
                           (uint16_t)((uint16_t)mine.best_class << 8) |
                           (uint16_t)((uint16_t)mine.best_tile << 10);
            if (ctx->tss_compress) {
                int same = -1;
                for (int j = 0; j < n; ++j) {
                    if (signatures[j] == sig) { same = j; break; }
                }
                if (same >= 0) {
                    ctx->stats.tss_signature_pruned++;
                    if (child_cmp(&ch, &candidates[same]) < 0) {
                        candidates[same] = ch;
                        signatures[same] = sig;
                    }
                    continue;
                }
            }
            signatures[n] = sig;
            candidates[n++] = ch;
            ctx->stats.tss_candidates_kept++;
        }
    }
    qsort(candidates, (size_t)n, sizeof(candidates[0]), child_cmp);
    for (int i = 0; i < n; ++i) {
        ThreatInfo mine = immediate_info(&candidates[i].board, attacker);
        if (popcount7(mine.colmask) >= 2) return true;
        if (tss_prove(ctx, &candidates[i].board, attacker, depth_left - 1, budget)) return true;
        if (ctx->stop || *budget == 0) return false;
    }
    return false;
}


static bool tss_prove(Context *ctx, const Board *b, uint8_t attacker,
                      unsigned depth_left, uint64_t *budget)
{
    if (tss_cached_proof(ctx, b, attacker)) return true;
    bool ok = tss_prove_impl(ctx, b, attacker, depth_left, budget);
    if (ok) tss_mark_proven(ctx, b, attacker);
    return ok;
}

static uint64_t scaled_budget(uint64_t base, unsigned mult, unsigned level)
{
    uint64_t b = base;
    for (unsigned i = 0; i < level; ++i) {
        if (mult <= 1) break;
        if (b > UINT64_MAX / mult) return UINT64_MAX;
        b *= mult;
    }
    return b;
}

/*
 * Selective "bridge" proof oracle.
 *
 * Standard TSS needs an attacker move to create an immediate forcing threat.
 * That misses quiet setup moves.  v3 tries a small relevance-ranked subset of
 * attacker's quiet setup moves.  For each attempted setup it then checks EVERY
 * legal defender reply.  Every reply must be proven winning for the attacker
 * by TSS (or, at deeper bridge levels, another bridge).
 *
 * Soundness:
 *   - attacker moves may be selectively omitted: failure is only "not proven";
 *   - defender replies are never selectively omitted;
 *   - a TRUE result is therefore a valid forced-win proof.
 *
 * The exact DF-PN search remains the fallback, so this oracle cannot change an
 * exact W/L/D result; it can only collapse a node before full branching.
 */
static bool bridge_prove(Context *ctx, const Board *b, uint8_t attacker,
                         unsigned bridge_layers, unsigned max_candidates,
                         unsigned tss_depth, uint64_t *budget)
{
    if (ctx->stop || g_interrupted || *budget == 0 || bridge_layers == 0) return false;
    (*budget)--;
    ctx->stats.bridge_nodes++;

    if ((ctx->stats.bridge_nodes & UINT64_C(0x3ff)) == 0) {
        double now = wall_time();
        if (ctx->deadline > 0.0 && now >= ctx->deadline) {
            ctx->stop = true;
            return false;
        }
    }

    if (b->turn != attacker) return false;
    ThreatInfo own = immediate_info(b, attacker);
    if (own.colmask) return true;
    ThreatInfo opp = immediate_info(b, attacker ^ 1u);
    if (opp.colmask) {
        /* Existing tactical obligation is exactly the domain of TSS. */
        return tss_prove(ctx, b, attacker, tss_depth, budget);
    }
    if (board_full(b)) return false;

    Child cand[MAX_MOVES];
    int relscore[MAX_MOVES];
    int n = 0;

    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            if (winning_move_class(b, attacker, col, tile) != WIN_NONE) return true;

            Board child = *b;
            int row = child.height[col];
            board_drop(&child, col, tile);

            /* Quiet setup must not hand the defender a win in one. */
            if (immediate_info(&child, child.turn).colmask) continue;

            /* Immediate forcing moves were already covered by ordinary TSS. */
            ThreatInfo mine = immediate_info(&child, attacker);
            if (mine.colmask) continue;

            unsigned rel = setup_relevance(&child, attacker, col, row);
            if (rel == 0) continue;

            Child ch = {0};
            ch.board = child;
            ch.move = move_encode(tile, col);
            /* Higher relevance first; keep deterministic center/tile tie-breaks. */
            unsigned capped = rel > 80u ? 80u : rel;
            ch.priority = (uint8_t)(90u - capped);
            relscore[n] = (int)rel;
            cand[n++] = ch;
        }
    }

    /* Small insertion sort so relscore stays aligned with candidates. */
    for (int i = 1; i < n; ++i) {
        Child c = cand[i];
        int rs = relscore[i];
        int j = i - 1;
        while (j >= 0 &&
               (relscore[j] < rs ||
                (relscore[j] == rs && child_cmp(&cand[j], &c) > 0))) {
            cand[j + 1] = cand[j];
            relscore[j + 1] = relscore[j];
            --j;
        }
        cand[j + 1] = c;
        relscore[j + 1] = rs;
    }

    int limit = n < (int)max_candidates ? n : (int)max_candidates;
    for (int i = 0; i < limit; ++i) {
        if (*budget == 0 || ctx->stop) break;
        ctx->stats.bridge_candidates++;

        const Board *after_setup = &cand[i].board;
        int defender = after_setup->turn;
        bool all_replies_win = true;
        bool had_reply = false;

        for (int oi = 0; oi < COLS && all_replies_win; ++oi) {
            int col = CENTER_ORDER[oi];
            for (int tile = 1; tile <= 3; ++tile) {
                if (!legal_tile_at(after_setup, col, tile)) continue;
                had_reply = true;
                ctx->stats.bridge_defender_replies++;

                if (winning_move_class(after_setup, defender, col, tile) != WIN_NONE) {
                    all_replies_win = false;
                    break;
                }

                Board reply = *after_setup;
                board_drop(&reply, col, tile); /* attacker to move again */

                bool proved = tss_prove(ctx, &reply, attacker, tss_depth, budget);
                if (!proved && bridge_layers > 1 && !ctx->stop && *budget) {
                    proved = bridge_prove(ctx, &reply, attacker,
                                          bridge_layers - 1, max_candidates,
                                          tss_depth, budget);
                }
                if (!proved) {
                    all_replies_win = false;
                    break;
                }
                if (ctx->stop || *budget == 0) {
                    all_replies_win = false;
                    break;
                }
            }
        }

        if (had_reply && all_replies_win) {
            tss_mark_proven(ctx, b, attacker);
            return true;
        }
    }

    if (*budget == 0) ctx->stats.bridge_budget_exhausted++;
    return false;
}



/*
 * v4 dependency proof oracle.
 *
 * This extends the quiet-setup idea using the global winning-structure
 * dependency map rather than a local run score.  At attacker OR nodes we try
 * only the top dependency-ranked moves (safe because one successful move is
 * sufficient for a proof).  At defender AND nodes EVERY legal defender reply
 * must remain winning for the attacker.  Therefore TRUE is a sound forced-win
 * proof; FALSE is only "not proven" and falls back to full DF-PN.
 */
static bool dependency_prove(Context *ctx, const Board *b, uint8_t attacker,
                             unsigned layers, unsigned max_candidates,
                             unsigned tss_depth, uint64_t *budget)
{
    if (ctx->stop || g_interrupted || *budget == 0 || layers == 0) return false;
    (*budget)--;
    ctx->stats.dep_oracle_nodes++;
    if ((ctx->stats.dep_oracle_nodes & UINT64_C(0x3ff)) == 0) {
        double now = wall_time();
        if (ctx->deadline > 0.0 && now >= ctx->deadline) {
            ctx->stop = true;
            return false;
        }
    }

    ThreatInfo own = immediate_info(b, b->turn);
    if (own.colmask) return b->turn == attacker;
    ThreatInfo opp = immediate_info(b, b->turn ^ 1u);
    unsigned nopp = popcount7(opp.colmask);
    if (nopp >= 2) return (b->turn ^ 1u) == attacker;
    if (board_full(b)) return false;
    if (!player_has_viable_segment(ctx, b, attacker)) return false;

    /* Existing immediate tactical obligations are handled more efficiently by TSS. */
    if (nopp == 1) return tss_prove(ctx, b, attacker, tss_depth, budget);

    if (b->turn == attacker) {
        DependencyMap dm;
        dependency_map_build(ctx, b, &dm);
        Child cand[MAX_MOVES];
        int n = 0;
        for (int oi = 0; oi < COLS; ++oi) {
            int col = CENTER_ORDER[oi];
            for (int tile = 1; tile <= 3; ++tile) {
                if (!legal_tile_at(b, col, tile)) continue;
                if (winning_move_class(b, attacker, col, tile) != WIN_NONE) return true;
                Board child = *b;
                board_drop(&child, col, tile);
                if (immediate_info(&child, child.turn).colmask) continue; /* loses in one */
                ThreatInfo mine = immediate_info(&child, attacker);
                if (popcount7(mine.colmask) >= 2) return true;
                Child ch = {0};
                ch.board = child;
                ch.move = move_encode(tile, col);
                ch.priority = mine.colmask ? 1 : 10;
                ch.dep_score = (int16_t)dependency_move_score(b, &child, attacker, col, tile, &dm);
                ch.terminal_winner = -1;
                cand[n++] = ch;
            }
        }
        qsort(cand, (size_t)n, sizeof(cand[0]), child_cmp);
        int limit = n < (int)max_candidates ? n : (int)max_candidates;
        for (int i = 0; i < limit; ++i) {
            if (ctx->stop || *budget == 0) break;
            ctx->stats.dep_oracle_candidates++;
            bool proved = tss_prove(ctx, &cand[i].board, attacker, tss_depth, budget);
            if (!proved && layers > 1 && !ctx->stop && *budget) {
                proved = dependency_prove(ctx, &cand[i].board, attacker,
                                          layers - 1, max_candidates,
                                          tss_depth, budget);
            }
            if (proved) {
                tss_mark_proven(ctx, b, attacker);
                return true;
            }
        }
        if (*budget == 0) ctx->stats.dep_oracle_budget_exhausted++;
        return false;
    }

    /* Defender to move: proof must survive EVERY legal defender move. */
    int defender = b->turn;
    bool had = false;
    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            had = true;
            ctx->stats.dep_oracle_defender_replies++;
            if (winning_move_class(b, defender, col, tile) != WIN_NONE) return false;
            Board child = *b;
            board_drop(&child, col, tile);
            bool proved = tss_prove(ctx, &child, attacker, tss_depth, budget);
            if (!proved && layers > 1 && !ctx->stop && *budget) {
                proved = dependency_prove(ctx, &child, attacker,
                                          layers - 1, max_candidates,
                                          tss_depth, budget);
            }
            if (!proved) return false;
            if (ctx->stop || *budget == 0) return false;
        }
    }
    return had;
}

/*
 * Bounded exact tactical solver.
 *
 * Returns 0/1 for a PROVEN winner, 2 for a PROVEN draw, and -1 for UNKNOWN.
 * Reaching the depth/budget limit never produces a loss/draw claim.  This is
 * therefore safe to use as a child pre-solver: it can eliminate branches that
 * are already decided tactically without changing the exact fallback search.
 */
static int tactical_exact(Context *ctx, const Board *b, unsigned depth_left,
                          uint64_t *budget)
{
    if (ctx->stop || g_interrupted || *budget == 0) return -1;
    (*budget)--;
    ctx->stats.tactical_nodes++;

    if ((ctx->stats.tactical_nodes & UINT64_C(0x3ff)) == 0) {
        double now = wall_time();
        if (ctx->deadline > 0.0 && now >= ctx->deadline) {
            ctx->stop = true;
            return -1;
        }
    }

    int mover = b->turn;
    ThreatInfo own = immediate_info(b, mover);
    if (own.colmask) return mover;

    ThreatInfo opp = immediate_info(b, mover ^ 1u);
    unsigned nopp = popcount7(opp.colmask);
    if (nopp >= 2) return mover ^ 1u;
    if (board_full(b)) return 2;
    if (depth_left == 0) return -1;

    int forced_col = -1;
    if (nopp == 1) {
        for (int c = 0; c < COLS; ++c) {
            if (opp.colmask & (1u << c)) { forced_col = c; break; }
        }
    }

    bool symmetric = forced_col < 0 && board_symmetric(b);
    bool any = false, saw_unknown = false, saw_draw = false;

    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        if (forced_col >= 0 && col != forced_col) continue;
        if (symmetric && col > 3) continue;

        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            any = true;

            Board child = *b;
            board_drop(&child, col, tile);
            int result = tactical_exact(ctx, &child, depth_left - 1, budget);

            if (result == mover) return mover; /* one exact winning choice is enough */
            if (result < 0) saw_unknown = true;
            else if (result == 2) saw_draw = true;

            if (ctx->stop || *budget == 0) {
                if (*budget == 0) ctx->stats.tactical_budget_exhausted++;
                return -1;
            }
        }
    }

    if (!any) return 2;
    if (saw_unknown) return -1;
    if (saw_draw) return 2;
    return mover ^ 1u; /* every legal move was proven losing */
}

static bool board_descendant_of(const Board *b, const Board *ancestor)
{
    if (b->ply < ancestor->ply) return false;
    uint64_t aocc = ancestor->p[0] | ancestor->p[1];
    if ((b->p[0] & aocc) != ancestor->p[0]) return false;
    if ((b->p[1] & aocc) != ancestor->p[1]) return false;
    if ((b->twos & aocc) != ancestor->twos) return false;
    if ((b->threes & aocc) != ancestor->threes) return false;
    return true;
}

/* Bounded, exact-only local DFS for a detected hot subtree. UNKNOWN at a
 * cutoff is never cached or injected into the main DF-PN TT. */
static int hot_exact_dfs(Context *ctx, const Board *b, unsigned depth_left,
                         uint64_t *budget)
{
    if (ctx->stop || g_interrupted || *budget == 0) return -1;
    int cached = -1;
    if (hot_tt_lookup(ctx, b, &cached)) return cached;

    (*budget)--;
    ctx->stats.hot_exact_nodes++;
    if ((ctx->stats.hot_exact_nodes & UINT64_C(0x7ff)) == 0) {
        double now = wall_time();
        if (ctx->deadline > 0.0 && now >= ctx->deadline) {
            ctx->stop = true;
            return -1;
        }
    }

    int mover = b->turn;
    int oppp = mover ^ 1;
    ThreatInfo own = immediate_info(b, mover);
    if (own.colmask) {
        hot_tt_store(ctx, b, mover);
        return mover;
    }
    ThreatInfo opp = immediate_info(b, oppp);
    unsigned nopp = popcount7(opp.colmask);
    if (nopp >= 2) {
        hot_tt_store(ctx, b, oppp);
        return oppp;
    }
    if (board_full(b)) {
        hot_tt_store(ctx, b, 2);
        return 2;
    }
    if (depth_left == 0) {
        ctx->stats.hot_exact_depth_cutoffs++;
        return -1;
    }

    int forced_col = -1;
    if (nopp == 1) {
        for (int c = 0; c < COLS; ++c) {
            if (opp.colmask & (1u << c)) { forced_col = c; break; }
        }
    }

    bool symmetric = forced_col < 0 && board_symmetric(b);
    static const int tile_order[3] = {2, 3, 1};
    bool any = false;
    bool saw_unknown = false;
    bool saw_draw = false;

    for (int oi = 0; oi < COLS; ++oi) {
        int col = CENTER_ORDER[oi];
        if (forced_col >= 0 && col != forced_col) continue;
        if (symmetric && col > 3) continue;
        for (int ti = 0; ti < 3; ++ti) {
            int tile = tile_order[ti];
            if (!legal_tile_at(b, col, tile)) continue;
            any = true;
            Board child = *b;
            board_drop(&child, col, tile);
            int r = hot_exact_dfs(ctx, &child, depth_left - 1, budget);
            if (r == mover) {
                hot_tt_store(ctx, b, mover);
                return mover;
            }
            if (r < 0) saw_unknown = true;
            else if (r == 2) saw_draw = true;
            if (ctx->stop || *budget == 0) return -1;
        }
    }

    if (!any) {
        hot_tt_store(ctx, b, 2);
        return 2;
    }
    if (saw_unknown) return -1;
    int r = saw_draw ? 2 : oppp;
    hot_tt_store(ctx, b, r);
    return r;
}

static uint64_t hot_scaled_budget(uint64_t base, unsigned mult, unsigned level)
{
    uint64_t x = base;
    for (unsigned i = 0; i < level; ++i) {
        if (mult <= 1) break;
        if (x > UINT64_MAX / mult) return UINT64_MAX;
        x *= mult;
    }
    return x;
}

static uint64_t frontier_effective_child_work(const Context *ctx, int ci)
{
    if (ci < 0 || ci >= MAX_MOVES) return 0;
    uint64_t w = ctx->frontier.child[ci].work_expansions;
    if (ctx->active_frontier_ci == ci &&
        ctx->stats.expansions >= ctx->active_child_start_exp) {
        uint64_t live = ctx->stats.expansions - ctx->active_child_start_exp;
        w = UINT64_MAX - w < live ? UINT64_MAX : w + live;
    }
    return w;
}

static uint64_t frontier_effective_reply_work(const Context *ctx, int ci, int ri)
{
    if (ci < 0 || ci >= MAX_MOVES || ri < 0 || ri >= MAX_MOVES) return 0;
    uint64_t w = ctx->frontier.child[ci].reply[ri].work_expansions;
    if (ctx->active_frontier_ci == ci && ctx->active_frontier_ri == ri &&
        ctx->stats.expansions >= ctx->active_reply_start_exp) {
        uint64_t live = ctx->stats.expansions - ctx->active_reply_start_exp;
        w = UINT64_MAX - w < live ? UINT64_MAX : w + live;
    }
    return w;
}

static int maybe_hot_exact(Context *ctx, int ci, int ri, const Board *target)
{
    if (!ctx->hot_exact_enabled || !ctx->frontier_enabled || !ctx->frontier.initialized ||
        ci < 0 || ri < 0 || ci >= MAX_MOVES || ri >= MAX_MOVES) return -1;

    FrontierChild *fc = &ctx->frontier.child[ci];
    FrontierReply *fr = &fc->reply[ri];
    if (!fr->valid) return -1;
    if (fr->hot_result_valid) return fr->hot_result;
    if (ctx->stats.expansions < ctx->hot_exact_min_total_exp) return -1;

    uint64_t stall = frontier_stall_exp(ctx);
    uint64_t child_work = frontier_effective_child_work(ctx, ci);
    uint64_t reply_work = frontier_effective_reply_work(ctx, ci, ri);
    if (stall < ctx->hot_exact_stall_exp ||
        reply_work < ctx->hot_exact_min_reply_exp) return -1;

    double cshare = ctx->stats.expansions ?
        (double)child_work / (double)ctx->stats.expansions : 0.0;
    double rshare = child_work ?
        (double)reply_work / (double)child_work : 0.0;
    if (cshare < ctx->hot_exact_child_share ||
        rshare < ctx->hot_exact_reply_share) return -1;

    double tss_rate = fr->work_tss_calls ?
        (double)fr->work_tss_proofs / (double)fr->work_tss_calls : 0.0;

    if (ctx->hot_tss_gate_enabled && !fr->tss_gate_active &&
        fr->work_tss_calls >= ctx->hot_tss_min_calls &&
        tss_rate <= ctx->hot_tss_max_proof_rate) {
        fr->tss_gate_active = true;
        ctx->hot_tss_gate_valid = true;
        ctx->hot_tss_gate_board = *target;
        ctx->stats.hot_tss_gate_activations++;
        char cm[8], rm[8];
        move_text(fc->move, cm); move_text(fr->move, rm);
        printf("[hot-tss] gate %s>%s tss=%" PRIu64 "/%" PRIu64
               " rate=%.4f%% cap=%u\n",
               cm, rm, fr->work_tss_proofs, fr->work_tss_calls,
               100.0 * tss_rate, ctx->hot_tss_level_cap);
        fflush(stdout);
    }

    if (fr->hot_attempt_level >= ctx->hot_exact_levels) return -1;
    if (fr->hot_exact_attempts &&
        reply_work >= fr->hot_last_attempt_work &&
        reply_work - fr->hot_last_attempt_work < ctx->hot_exact_retry_work)
        return -1;

    unsigned level = fr->hot_attempt_level++;
    unsigned depth = ctx->hot_exact_depth + level * ctx->hot_exact_depth_step;
    uint64_t budget = hot_scaled_budget(ctx->hot_exact_budget,
                                        ctx->hot_exact_budget_mult, level);
    fr->hot_last_attempt_work = reply_work;
    fr->hot_exact_attempts++;
    ctx->stats.hot_exact_triggers++;
    ctx->stats.hot_exact_attempts++;

    uint64_t nodes0 = ctx->stats.hot_exact_nodes;
    char cm[8], rm[8];
    move_text(fc->move, cm); move_text(fr->move, rm);
    printf("[hot-exact] trigger %s>%s child=%.1f%% reply=%.1f%% stall=%" PRIu64
           " level=%u depth=%u budget=%" PRIu64 " tss=%.4f%%\n",
           cm, rm, 100.0 * cshare, 100.0 * rshare, stall,
           level, depth, budget, 100.0 * tss_rate);
    fflush(stdout);

    uint64_t left = budget;
    int result = hot_exact_dfs(ctx, target, depth, &left);
    uint64_t used = ctx->stats.hot_exact_nodes - nodes0;
    fr->hot_exact_nodes += used;
    if (left == 0 && result < 0) ctx->stats.hot_exact_budget_exhausted++;

    if (result >= 0) {
        fr->hot_result_valid = true;
        fr->hot_result = (int8_t)result;
        ctx->stats.hot_exact_proofs++;
        if (result == 2) ctx->stats.hot_exact_draws++;
        else if (result == ctx->prover) ctx->stats.hot_exact_wins++;
        else ctx->stats.hot_exact_losses++;

        bool hit, mirrored;
        TTEntry *e = tt_access(ctx, target, true, &hit, &mirrored);
        (void)hit; (void)mirrored;
        if (result == (int)ctx->prover) {
            e->pn = 0; e->dn = PN_INF;
        } else {
            e->pn = PN_INF; e->dn = 0;
        }
        ctx->stats.oracle_collapses++;
        printf("[hot-exact] PROVED %s>%s result=%s nodes=%" PRIu64
               " left=%" PRIu64 "\n",
               cm, rm, result == 2 ? "DRAW" : player_name(result), used, left);
    } else {
        printf("[hot-exact] UNKNOWN %s>%s nodes=%" PRIu64
               " left=%" PRIu64 "\n", cm, rm, used, left);
    }
    fflush(stdout);
    return result;
}


/*
 * v9.8.2 scheduler fix: v9 only attempted hot exact search when the normal DF-PN
 * happened to descend through that exact depth-2 reply at that moment.  A
 * historically dominant stalled subtree can cease to be the instantaneous MPC,
 * so the trigger could remain at 0/0/0 despite overwhelming accumulated work.
 *
 * This root-level scheduler selects the globally hottest root child and, within
 * it, the hottest recorded reply, then asks maybe_hot_exact() whether all exact
 * trigger conditions are satisfied.  The stored FrontierReply contains the
 * exact board, so the local solver no longer depends on the current DF-PN path.
 */
static int maybe_schedule_hot_exact(Context *ctx)
{
    if (!ctx->hot_exact_enabled || !ctx->frontier_enabled || !ctx->frontier.initialized)
        return -1;

    FrontierProfile *fp = &ctx->frontier;
    int hot_ci = -1;
    uint64_t hot_child_work = 0;
    for (unsigned i = 0; i < fp->child_count; ++i) {
        FrontierChild *fc = &fp->child[i];
        if (!fc->valid || fc->reply_count == 0) continue;
        uint64_t ew = frontier_effective_child_work(ctx, (int)i);
        if (hot_ci < 0 || ew > hot_child_work) {
            hot_ci = (int)i;
            hot_child_work = ew;
        }
    }
    if (hot_ci < 0) return -1;

    FrontierChild *fc = &fp->child[hot_ci];
    int hot_ri = -1;
    uint64_t hot_reply_work = 0;
    for (unsigned j = 0; j < fc->reply_count; ++j) {
        FrontierReply *fr = &fc->reply[j];
        if (!fr->valid) continue;
        uint64_t ew = frontier_effective_reply_work(ctx, hot_ci, (int)j);
        if (hot_ri < 0 || ew > hot_reply_work) {
            hot_ri = (int)j;
            hot_reply_work = ew;
        }
    }
    if (hot_ri < 0) return -1;

    FrontierReply *fr = &fc->reply[hot_ri];
    return maybe_hot_exact(ctx, hot_ci, hot_ri, &fr->board);
}

static void proposition_pair(int winner, uint8_t prover, uint32_t *pn, uint32_t *dn)
{
    if (winner == (int)prover) { *pn = 0; *dn = PN_INF; }
    else { *pn = PN_INF; *dn = 0; }
}

static void child_pair(Context *ctx, const Child *ch, uint32_t *pn, uint32_t *dn)
{
    if (ch->terminal_winner >= 0) {
        proposition_pair(ch->terminal_winner, ctx->prover, pn, dn);
        return;
    }
    tt_pair(ctx, &ch->board, pn, dn);
}


/* ---------------- v9.8.2 SPDFPN-style job scheduler ---------------- */

static void sp_true_pair(ParallelShared *sp, const Child *ch, uint32_t *pn, uint32_t *dn)
{
    if (ch->terminal_winner >= 0) {
        proposition_pair(ch->terminal_winner, sp->prover, pn, dn);
        return;
    }
    uint64_t work=0;
    if (!sp_shared_read(sp,&ch->board,pn,dn,&work)) { *pn=1; *dn=1; }
}

static int sp_generate_children(Context *sched, ParallelShared *sp, const Board *b,
                                Child out[MAX_MOVES], int *status)
{
    bool hit, mirrored;
    TTEntry *entry=tt_access(sched,b,true,&hit,&mirrored);
    uint8_t tt_move=tt_best_move(sched,entry,mirrored);
    bool first=!(entry->flags & TT_FLAG_EXPANDED);
    int n=generate_children(sched,b,out,tt_move,status,first,entry,mirrored);
    if (*status != -1) {
        uint32_t pn,dn; proposition_pair(*status,sp->prover,&pn,&dn);
        sp_shared_publish(sp,b,pn,dn);
    } else if (n<=0) {
        uint32_t pn,dn; proposition_pair(2,sp->prover,&pn,&dn);
        sp_shared_publish(sp,b,pn,dn);
    }
    return n;
}

static void sp_compute_pair(Context *sched, ParallelShared *sp, const Board *b,
                            uint32_t *pn, uint32_t *dn,
                            uint8_t *best_move, uint32_t *second_metric,
                            Child children[MAX_MOVES], int *n_out)
{
    int status=-1;
    int n=sp_generate_children(sched,sp,b,children,&status);
    if (n_out) *n_out=n;
    if (status!=-1) {
        proposition_pair(status,sp->prover,pn,dn);
        if (best_move) *best_move=0;
        if (second_metric) *second_metric=PN_INF;
        return;
    }
    if(n<=0) {
        proposition_pair(2,sp->prover,pn,dn);
        if (best_move) *best_move=0;
        if (second_metric) *second_metric=PN_INF;
        return;
    }
    bool is_or=b->turn==sp->prover;
    uint32_t np=is_or?PN_INF:0, nd=is_or?0:PN_INF;
    uint32_t best=PN_INF, second=PN_INF; int bi=-1;
    for(int i=0;i<n;++i){
        uint32_t cp,cd;
        sp_true_pair(sp,&children[i],&cp,&cd);
        uint32_t metric=is_or?cp:cd;
        if(is_or){ if(cp<np)np=cp; nd=sat_add(nd,cd); }
        else { np=sat_add(np,cp); if(cd<nd)nd=cd; }
        if(metric<best){ second=best; best=metric; bi=i; }
        else if(metric<second) second=metric;
    }
    *pn=np; *dn=nd;
    if(best_move)*best_move=bi>=0?children[bi].move:0;
    if(second_metric)*second_metric=second;
}

static void sp_recompute_true_path(Context *sched, ParallelShared *sp, const SPActiveJob *job)
{
    if(!job || !job->path_len) return;
    for(int d=(int)job->path_len-1; d>=0; --d){
        Child ch[MAX_MOVES]; int n=0; uint32_t pn,dn,second; uint8_t bm;
        sp_compute_pair(sched,sp,&job->path[d],&pn,&dn,&bm,&second,ch,&n);
        sp_shared_publish(sp,&job->path[d],pn,dn);
    }
}

/* v9.8.2: immediately promote an exact descendant discovered inside a
   recursive reroot session into the main canonical shared TT, then
   reaggregate the concrete main-root -> descendant ancestor chain.  Only
   pn==0/dn==0 is accepted here; UNKNOWN/finite reroot values never cross this
   channel. */
static void sp_promote_exact_descendant(Context *ctx, const Board *b,
                                        uint32_t pn, uint32_t dn,
                                        const Board *path, unsigned path_len)
{
    if (!ctx || !ctx->parallel_exact_shared || !b ||
        (pn != 0 && dn != 0) || (pn == 0 && dn == 0)) return;
    ParallelShared *sp = ctx->parallel_exact_shared;
    atomic_fetch_add_explicit(&sp->reroot_promote_found, 1, memory_order_relaxed);

    Board full[SP_PATH_MAX];
    unsigned len = 0;
    if (path && path_len) {
        len = path_len > SP_PATH_MAX ? SP_PATH_MAX : path_len;
        memcpy(full, path, sizeof(Board) * len);
    }
    if (len == 0) full[len++] = *b;
    else if (!same_canonical_state(&full[len - 1], b) && len < SP_PATH_MAX)
        full[len++] = *b;

    bool inserted = false, conflict = false;
    uint64_t ancestors_changed = 0;
    uint32_t root_pn = 1, root_dn = 1;
    int solved_now = 0;

    sp_job_lock(sp);
    uint32_t old_pn = 1, old_dn = 1; uint64_t old_work = 0;
    bool had = sp_shared_read(sp, b, &old_pn, &old_dn, &old_work);
    if (had && (old_pn == 0 || old_dn == 0) &&
        (old_pn != pn || old_dn != dn)) {
        conflict = true;
    } else if (!had || old_pn != pn || old_dn != dn) {
        sp_shared_publish(sp, b, pn, dn);
        uint32_t chk_pn = 1, chk_dn = 1; uint64_t chk_work = 0;
        if (sp_shared_read(sp, b, &chk_pn, &chk_dn, &chk_work) &&
            chk_pn == pn && chk_dn == dn) inserted = true;
        else if (chk_pn == 0 || chk_dn == 0) conflict = true;
    }

    if (!conflict && len > 1) {
        /* Descendant itself is already exact.  Recompute only its ancestors,
           deepest first, using true child PN/DN from the shared table. */
        for (int d = (int)len - 2; d >= 0; --d) {
            uint32_t before_pn = 1, before_dn = 1; uint64_t w = 0;
            bool before = sp_shared_read(sp, &full[d], &before_pn, &before_dn, &w);
            Child ch[MAX_MOVES]; int n = 0; uint32_t np = 1, nd = 1, second = PN_INF; uint8_t bm = 0;
            sp_compute_pair(ctx, sp, &full[d], &np, &nd, &bm, &second, ch, &n);
            sp_shared_publish(sp, &full[d], np, nd);
            if (!before || np != before_pn || nd != before_dn) ancestors_changed++;
        }
    }

    uint64_t rw = 0;
    if (sp_shared_read(sp, &sp->root, &root_pn, &root_dn, &rw)) {
        if (root_pn == 0) { atomic_store_explicit(&sp->solved, 1, memory_order_release); solved_now = 1; }
        else if (root_dn == 0) { atomic_store_explicit(&sp->solved, 2, memory_order_release); solved_now = 2; }
    }
    sp_job_unlock(sp);

    if (inserted)
        atomic_fetch_add_explicit(&sp->reroot_promote_inserted, 1, memory_order_relaxed);
    if (ancestors_changed)
        atomic_fetch_add_explicit(&sp->reroot_promote_ancestor_changed, ancestors_changed, memory_order_relaxed);
    if (conflict)
        atomic_fetch_add_explicit(&sp->reroot_promote_conflicts, 1, memory_order_relaxed);

    printf("[rr-promote] exact depth=%u proposition=%s inserted=%u ancestorChanged=%" PRIu64
           " conflict=%u root=(%u,%u) solved=%d\n",
           len ? len - 1u : 0u, pn == 0 ? "PROVED" : "DISPROVED",
           inserted ? 1u : 0u, ancestors_changed, conflict ? 1u : 0u,
           root_pn, root_dn, solved_now);
    fflush(stdout);
}

static unsigned sp_child_busy(const ParallelShared *sp, unsigned depth,
                              const Board *parent, const Board *child)
{
    unsigned busy=0;
    for(unsigned j=0;j<sp->threads;++j){
        const SPActiveJob *a=&sp->active[j];
        if(!a->active || a->path_len<=depth+1) continue;
        if(!same_canonical_state(&a->path[depth],parent)) continue;
        if(same_canonical_state(&a->path[depth+1],child)) busy++;
    }
    if (sp->reroot_active && sp->reroot_request.valid &&
        sp->reroot_request.path_len > depth + 1 &&
        same_canonical_state(&sp->reroot_request.path[depth], parent) &&
        same_canonical_state(&sp->reroot_request.path[depth+1], child)) busy++;
    return busy;
}

/* Number of active jobs whose reserved path currently passes through this
   exact scheduler node.  A busy node is not assigned again as a whole job;
   instead the scheduler descends and lets the next worker split the hard
   subtree at a deeper MPN-compatible point. */
static unsigned sp_node_busy(const ParallelShared *sp, unsigned depth, const Board *node)
{
    unsigned busy=0;
    for(unsigned j=0;j<sp->threads;++j){
        const SPActiveJob *a=&sp->active[j];
        if(!a->active || a->path_len<=depth) continue;
        if(same_canonical_state(&a->path[depth],node)) busy++;
    }
    if (sp->reroot_active && sp->reroot_request.valid &&
        sp->reroot_request.path_len > depth &&
        same_canonical_state(&sp->reroot_request.path[depth], node)) busy++;
    return busy;
}

static bool sp_assign_job(Context *sched, ParallelShared *sp, unsigned worker_id,
                          SPActiveJob *out)
{
    memset(out,0,sizeof(*out));
    if(atomic_load_explicit(&sp->solved,memory_order_acquire)!=0) return false;

    Board cur=sp->root;
    out->path[0]=cur; out->path_len=1;
    for(unsigned depth=0; depth<SP_PATH_MAX-1; ++depth){
        uint32_t tpn=1,tdn=1; uint64_t work=0;
        if(!sp_shared_read(sp,&cur,&tpn,&tdn,&work)){ tpn=1;tdn=1;work=0; }
        if(tpn==0 || tdn==0){
            atomic_fetch_add_explicit(&sp->assign_fail_exact,1,memory_order_relaxed);
            if(depth==0) atomic_store(&sp->solved,tpn==0?1:2);
            else sp_recompute_true_path(sched,sp,out);
            return false;
        }
        unsigned node_busy = sp_node_busy(sp,depth,&cur);
        if(depth>=sp->min_job_depth && work < sp->max_work_per_job && node_busy==0){
            out->active=true; out->worker_id=worker_id;
            out->thpn=PN_INF; out->thdn=PN_INF;
            out->initial_pn=tpn; out->initial_dn=tdn;
            return true;
        }

        Child children[MAX_MOVES]; int status=-1;
        int n=sp_generate_children(sched,sp,&cur,children,&status);
        if(status!=-1 || n<=0){
            atomic_fetch_add_explicit(&sp->assign_fail_terminal,1,memory_order_relaxed);
            sp_recompute_true_path(sched,sp,out);
            return false;
        }

        bool is_or=cur.turn==sp->prover;
        uint32_t node_pn=is_or?PN_INF:0, node_dn=is_or?0:PN_INF;
        uint32_t cp[MAX_MOVES],cd[MAX_MOVES];
        unsigned busy[MAX_MOVES];
        for(int i=0;i<n;++i){
            sp_true_pair(sp,&children[i],&cp[i],&cd[i]);
            if(is_or){ if(cp[i]<node_pn)node_pn=cp[i]; node_dn=sat_add(node_dn,cd[i]); }
            else { node_pn=sat_add(node_pn,cp[i]); if(cd[i]<node_dn)node_dn=cd[i]; }
            busy[i]=sp_child_busy(sp,depth,&cur,&children[i].board);
        }
        /* Shared TT always receives TRUE finite estimates.  Busy reservations
           are scheduler-only.  v9.8.2 deliberately uses a SOFT busy penalty
           rather than metric*(busy+1): several workers may stay inside the
           globally most-proving region, but they are progressively split at
           deeper nodes instead of duplicating the exact same assigned node. */
        sp_shared_publish(sp,&cur,node_pn,node_dn);

        int best=-1; uint64_t best_virtual=UINT64_MAX; uint32_t best_true=PN_INF;
        uint64_t penalty_milli=(uint64_t)(sp->busy_penalty*1000.0+0.5);
        for(int i=0;i<n;++i){
            uint32_t metric=is_or?cp[i]:cd[i];
            uint64_t scale=1000u + penalty_milli*(uint64_t)busy[i];
            uint64_t virtual_metric=(scale && (uint64_t)metric>UINT64_MAX/scale)?UINT64_MAX:(uint64_t)metric*scale;
            if(best<0 || virtual_metric<best_virtual ||
               (virtual_metric==best_virtual && metric<best_true)){
                best_virtual=virtual_metric; best_true=metric; best=i;
            }
        }
        if(best>=0 && busy[best]>0)
            atomic_fetch_add_explicit(&sp->concentrated_assignments,1,memory_order_relaxed);
        if(best<0){
            atomic_fetch_add_explicit(&sp->assign_fail_terminal,1,memory_order_relaxed);
            return false;
        }
        cur=children[best].board;
        out->path[out->path_len++]=cur;
        atomic_fetch_add_explicit(&sp->scheduler_descents,1,memory_order_relaxed);
    }

    /* At the deepest representable path, keep making bounded progress rather
       than idling merely because historical work exceeded MaxWorkPerJob. */
    if(out->path_len>sp->min_job_depth){
        const Board *cand=&out->path[out->path_len-1];
        uint32_t pn=1,dn=1; uint64_t work=0; (void)sp_shared_read(sp,cand,&pn,&dn,&work);
        if(pn!=0 && dn!=0){
            out->active=true; out->worker_id=worker_id; out->thpn=PN_INF; out->thdn=PN_INF;
            out->initial_pn=pn; out->initial_dn=dn;
            atomic_fetch_add_explicit(&sp->assign_fail_pathmax,1,memory_order_relaxed);
            return true;
        }
    }
    atomic_fetch_add_explicit(&sp->assign_fail_pathmax,1,memory_order_relaxed);
    return false;
}

static void sp_finish_job(Context *sched, ParallelShared *sp, SPActiveJob *job, uint64_t work)
{
    if(!job || !job->active) return;
    for(unsigned d=0; d<job->path_len; ++d) sp_shared_add_work(sp,&job->path[d],work);
    sp_recompute_true_path(sched,sp,job);
    job->active=false;
    sp->active[job->worker_id].active=false;
    uint32_t rpn=1,rdn=1; uint64_t rw=0;
    if(sp_shared_read(sp,&sp->root,&rpn,&rdn,&rw)){
        uint64_t total_exp = atomic_load_explicit(&sp->total_worker_expansions, memory_order_relaxed);
        if (!sp->reroot_root_valid || rpn != sp->reroot_root_pn || rdn != sp->reroot_root_dn) {
            sp->reroot_root_valid = true;
            sp->reroot_root_pn = rpn; sp->reroot_root_dn = rdn;
            sp->reroot_root_change_exp = total_exp;
        }
        if(rpn==0) atomic_store_explicit(&sp->solved,1,memory_order_release);
        else if(rdn==0) atomic_store_explicit(&sp->solved,2,memory_order_release);
    }
}


static SPRerootTrack *sp_reroot_track_get(ParallelShared *sp, const Board *b, uint8_t prover)
{
    uint64_t k[4]; bool mirrored;
    canonical_key(b,k,&mirrored); (void)mirrored;
    sp->reroot_track_tick++;
    if (sp->reroot_track_tick == 0) sp->reroot_track_tick = 1;
    for (unsigned i=0;i<SP_REROOT_TRACK_SLOTS;++i) {
        SPRerootTrack *t=&sp->reroot_track[i];
        if (!t->valid || t->prover!=prover) continue;
        if (t->k[0]==k[0] && t->k[1]==k[1] && t->k[2]==k[2] && t->k[3]==k[3] &&
            t->board.p[0]==b->p[0] && t->board.p[1]==b->p[1] &&
            t->board.twos==b->twos && t->board.threes==b->threes && t->board.turn==b->turn) {
            t->last_used_tick=sp->reroot_track_tick;
            return t;
        }
    }
    unsigned slot=SP_REROOT_TRACK_SLOTS;
    for (unsigned i=0;i<SP_REROOT_TRACK_SLOTS;++i) if (!sp->reroot_track[i].valid) { slot=i; break; }
    if (slot==SP_REROOT_TRACK_SLOTS) {
        slot=0;
        for (unsigned i=1;i<SP_REROOT_TRACK_SLOTS;++i)
            if (sp->reroot_track[i].last_used_tick < sp->reroot_track[slot].last_used_tick) slot=i;
    }
    SPRerootTrack *t=&sp->reroot_track[slot];
    memset(t,0,sizeof(*t));
    t->valid=true; t->prover=prover; memcpy(t->k,k,sizeof(k)); t->board=*b;
    t->last_used_tick=sp->reroot_track_tick;
    return t;
}

static void sp_reroot_path_text(const SPRerootRequest *rr, char *buf, size_t cap)
{
    if (!buf || cap==0) return;
    buf[0]='\0';
    if (!rr) return;
    size_t used=0;
    for (unsigned d=1; d<rr->path_len; ++d) {
        uint8_t m=rr->moves[d];
        if (!m) continue;
        int n=snprintf(buf+used,cap-used,"%s%d%c",used?">":"",move_tile(m),'A'+move_col(m));
        if (n<0 || (size_t)n>=cap-used) { buf[cap-1]='\0'; return; }
        used+=(size_t)n;
    }
}

/* v9.8.2 parallel-aware hot-region detector.

   v9.7.2 required the ROOT PN/DN pair to remain exactly unchanged.  Under
   SPDFPN that is too strict: other workers keep nudging the root while the
   same deep proof obligation can consume millions of calls.  v9.8.2 therefore
   watches the ACTIVE JOB RESERVATIONS themselves.

   A shallow canonical prefix (depth 1..--reroot-hot-region-depth) is a hot
   region when >= --reroot-hot-min-busy active workers currently pass through
   it.  A region matures when:
     - accumulated hot exposure reaches --reroot-hot-age seconds; short
       absences <= --reroot-hot-gap are tolerated,
     - aggregate shared-path work added during the hot window reaches
       --reroot-hot-work DFPN calls,
     - both PN and DN stay within --reroot-hot-max-change of the current
       baseline; a larger change is meaningful progress and restarts the
       stability window.

   Once a region matures, the scheduler chooses a concrete deep descendant
   (depth >= --reroot-min-depth) currently reserved inside that region, with at
   least --reroot-min-work historical shared work, and sends that board to the
   persistent reroot worker.  The detector is scheduler-only: virtual/busy
   values never enter the true shared TT and only exact reroot results may be
   injected into it. */
static double sp_rel_u32(uint32_t a, uint32_t b)
{
    double den=(double)(a ? a : 1u);
    double d=(double)(a>b ? a-b : b-a);
    return d/den;
}

static SPHotTrack *sp_hot_track_get(ParallelShared *sp, const Board *b,
                                    uint8_t prover, unsigned depth,
                                    double now, uint64_t probe_id, uint64_t work,
                                    uint32_t pn, uint32_t dn,
                                    double max_change, double gap_sec)
{
    uint64_t k[4]; bool mirrored;
    canonical_key(b,k,&mirrored); (void)mirrored;
    sp->hot_track_tick++;
    if (sp->hot_track_tick==0) sp->hot_track_tick=1;
    SPHotTrack *t=NULL;
    for (unsigned i=0;i<SP_HOT_TRACK_SLOTS;++i) {
        SPHotTrack *q=&sp->hot_track[i];
        if (!q->valid || q->prover!=prover) continue;
        if (q->k[0]==k[0] && q->k[1]==k[1] && q->k[2]==k[2] && q->k[3]==k[3] &&
            q->board.p[0]==b->p[0] && q->board.p[1]==b->p[1] &&
            q->board.twos==b->twos && q->board.threes==b->threes && q->board.turn==b->turn) {
            t=q; break;
        }
    }
    if (!t) {
        unsigned slot=SP_HOT_TRACK_SLOTS;
        for (unsigned i=0;i<SP_HOT_TRACK_SLOTS;++i)
            if (!sp->hot_track[i].valid) { slot=i; break; }
        if (slot==SP_HOT_TRACK_SLOTS) {
            slot=0;
            for (unsigned i=1;i<SP_HOT_TRACK_SLOTS;++i)
                if (sp->hot_track[i].last_used_tick < sp->hot_track[slot].last_used_tick) slot=i;
        }
        t=&sp->hot_track[slot]; memset(t,0,sizeof(*t));
        t->valid=true; t->prover=prover; t->depth=(uint8_t)depth;
        memcpy(t->k,k,sizeof(k)); t->board=*b;
        t->first_seen_time=now; t->last_seen_time=now; t->exposure_sec=0.0; t->last_probe_id=probe_id;
        t->start_work=work; t->last_work=work;
        t->start_pn=pn; t->start_dn=dn; t->last_pn=pn; t->last_dn=dn;
        t->last_used_tick=sp->hot_track_tick;
        return t;
    }

    bool gap_reset = gap_sec>0.0 && now-t->last_seen_time > gap_sec;
    double dp=sp_rel_u32(t->start_pn,pn), dd=sp_rel_u32(t->start_dn,dn);
    bool progress_reset = max_change>=0.0 && (dp>max_change || dd>max_change);
    if (gap_reset || progress_reset || work < t->start_work) {
        t->first_seen_time=now; t->start_work=work;
        t->start_pn=pn; t->start_dn=dn; t->exposure_sec=0.0;
    } else if (t->last_probe_id+1u==probe_id) {
        double dt=now-t->last_seen_time;
        if (dt>0.0 && dt<60.0) t->exposure_sec += dt;
    }
    t->depth=(uint8_t)depth; t->last_seen_time=now; t->last_probe_id=probe_id;
    t->last_work=work; t->last_pn=pn; t->last_dn=dn; t->last_used_tick=sp->hot_track_tick;
    return t;
}

static bool sp_build_reroot_request(Context *sched, ParallelShared *sp,
                                    SPRerootRequest *out)
{
    memset(out,0,sizeof(*out));
    sp->hot_diag_valid=false; sp->hot_diag_candidate=false;
    if (!sched->deep_reroot_enabled || sp->reroot_pending || sp->reroot_active) return false;

    double now=wall_time();
    uint64_t total_exp=atomic_load_explicit(&sp->total_worker_expansions,memory_order_relaxed);
    if (total_exp < sched->deep_reroot_min_total_exp) return false;
    sp->hot_probe_id++; if (sp->hot_probe_id==0) sp->hot_probe_id=1;
    uint64_t probe_id=sp->hot_probe_id;

    uint64_t root_work=0; uint32_t rpn=1,rdn=1;
    (void)sp_shared_read(sp,&sp->root,&rpn,&rdn,&root_work);

    bool have=false; SPRerootRequest best; memset(&best,0,sizeof(best));
    double best_cand_score=-1.0, diag_score=-1.0; unsigned diag_depth=0;
    SPHotTrack *best_anchor_track=NULL;

    /* Parallel hot REGION detection has two levels:

       (1) Define a region by a shallow reserved prefix (depth 1..regionDepth)
           shared by >= minBusy active jobs.  This remains meaningful even when
           root MPC and deeper exact nodes oscillate.
       (2) Once that region has accumulated enough stable exposure/work, reroot
           the most-worked CURRENT deep descendant (depth >= minDepth) carried
           by one of those jobs.

       Thus the trigger measures where parallel CPU is persistently going,
       while the independent solver still starts at a concrete deep board. */
    unsigned max_region=sched->deep_reroot_hot_region_depth;
    if (max_region>=sched->deep_reroot_min_depth && sched->deep_reroot_min_depth>1)
        max_region=sched->deep_reroot_min_depth-1;
    if (max_region<1) max_region=1;

    for (unsigned j=0;j<sp->threads;++j) {
        const SPActiveJob *a=&sp->active[j];
        if (!a->active) continue;
        unsigned lim=a->path_len<SP_PATH_MAX?a->path_len:SP_PATH_MAX;
        if (lim<=1) continue;
        unsigned rlim=lim-1<max_region?lim-1:max_region;
        for (unsigned rdepth=1; rdepth<=rlim; ++rdepth) {
            const Board *anchor=&a->path[rdepth];
            bool duplicate=false;
            for (unsigned k=0;k<j && !duplicate;++k) {
                const SPActiveJob *b=&sp->active[k];
                if (!b->active || b->path_len<=rdepth) continue;
                if (same_canonical_state(&b->path[rdepth],anchor)) duplicate=true;
            }
            if (duplicate) continue;
            unsigned busy=sp_node_busy(sp,rdepth,anchor);
            if (busy<sched->deep_reroot_hot_min_busy) continue;

            uint32_t pn=1,dn=1; uint64_t work=0,change_work=0;
            if (!sp_shared_read_meta(sp,anchor,&pn,&dn,&work,&change_work) || pn==0 || dn==0 || work==0) continue;
            SPHotTrack *ht=sp_hot_track_get(sp,anchor,sp->prover,rdepth,now,probe_id,work,pn,dn,
                                            sched->deep_reroot_hot_max_change,
                                            sched->deep_reroot_hot_gap_sec);
            double age=ht->exposure_sec;
            uint64_t hot_work=work>=ht->start_work?work-ht->start_work:0;
            double dp=sp_rel_u32(ht->start_pn,pn), dd=sp_rel_u32(ht->start_dn,dn);
            double ar=sched->deep_reroot_hot_age_sec>0?age/sched->deep_reroot_hot_age_sec:1.0;
            double wr=sched->deep_reroot_hot_work>0?(double)hot_work/(double)sched->deep_reroot_hot_work:1.0;
            double maturity=ar<wr?ar:wr;
            double dscore=maturity+0.03*(double)busy+0.002*(double)rdepth;
            if (!sp->hot_diag_valid || dscore>diag_score || (dscore==diag_score && rdepth>diag_depth)) {
                sp->hot_diag_valid=true; sp->hot_diag_depth=0; sp->hot_diag_region_depth=rdepth; sp->hot_diag_busy=busy;
                sp->hot_diag_age_sec=age; sp->hot_diag_work=hot_work; sp->hot_diag_delta_pn=dp; sp->hot_diag_delta_dn=dd;
                sp->hot_diag_pn=pn; sp->hot_diag_dn=dn; sp->hot_diag_board=*anchor;
                diag_score=dscore; diag_depth=rdepth;
            }

            bool stable=dp<=sched->deep_reroot_hot_max_change && dd<=sched->deep_reroot_hot_max_change;
            if (!(age>=sched->deep_reroot_hot_age_sec && hot_work>=sched->deep_reroot_hot_work && stable)) continue;

            /* Select a concrete deep descendant inside the stable hot region.
               Prefer descendants whose own shared PN/DN has stalled, but do
               not require complete local immobility: parallel recomputation
               can keep a deep node moving slightly even when the REGION is the
               persistent proof bottleneck.  --reroot-min-work is the minimum
               historical target work floor in parallel v9.8.2. */
            const SPActiveJob *ta=NULL; unsigned tdepth=0; uint64_t twork=0, tstall=0;
            uint32_t tpn=1,tdn=1;
            for (unsigned k=0;k<sp->threads;++k) {
                const SPActiveJob *b=&sp->active[k];
                if (!b->active || b->path_len<=rdepth || !same_canonical_state(&b->path[rdepth],anchor)) continue;
                for (unsigned d=sched->deep_reroot_min_depth; d<b->path_len; ++d) {
                    uint32_t xp=1,xd=1; uint64_t xw=0,xc=0;
                    if (!sp_shared_read_meta(sp,&b->path[d],&xp,&xd,&xw,&xc) || xp==0 || xd==0 || xw==0) continue;
                    uint64_t xs=xw>=xc?xw-xc:0;
                    if (xw < sched->deep_reroot_min_work) continue;
                    if (!ta || xs>tstall || (xs==tstall && (xw>twork || (xw==twork && d>tdepth)))) {
                        ta=b; tdepth=d; twork=xw; tstall=xs; tpn=xp; tdn=xd;
                    }
                }
            }
            if (!ta) continue;
            const Board *target=&ta->path[tdepth];
            SPRerootTrack *tr=sp_reroot_track_get(sp,target,sp->prover);
            bool retry_ok=tr && tr->attempts<sched->deep_reroot_max_attempts;
            if (retry_ok && tr->attempts>0) {
                if (twork<tr->last_attempt_work || twork-tr->last_attempt_work<sched->deep_reroot_retry_work) retry_ok=false;
            }
            if (!retry_ok) continue;

            double cscore=maturity+0.08*(double)busy+0.002*(double)tdepth;
            if (!have || cscore>best_cand_score) {
                memset(&best,0,sizeof(best)); best.valid=true; best.depth=tdepth; best.path_len=tdepth+1;
                best.target=*target; memcpy(best.path,ta->path,sizeof(Board)*(tdepth+1));
                for (unsigned d=1; d<=tdepth; ++d) {
                    Child ch[MAX_MOVES]; int status=-1; int n=sp_generate_children(sched,sp,&ta->path[d-1],ch,&status);
                    if (status==-1) for (int ci=0;ci<n;++ci) if (same_canonical_state(&ch[ci].board,&ta->path[d])) { best.moves[d]=ch[ci].move; break; }
                }
                best.target_work=twork; best.target_stall_work=tstall; best.hot_region_work=hot_work; best.root_stall_exp=0;
                best.hot_age_sec=age; best.hot_delta_pn=dp; best.hot_delta_dn=dd; best.hot_busy=busy; best.hot_region_depth=rdepth;
                best.root_share=root_work?(double)twork/(double)root_work:0.0;
                best.inner_share=work?(double)twork/(double)work:0.0;
                best.attempt=tr->attempts+1u; have=true; best_cand_score=cscore; best_anchor_track=ht;
                sp->hot_diag_depth=tdepth;
                (void)tpn; (void)tdn;
            }
        }
    }

    if (!have) return false;
    SPRerootTrack *tr=sp_reroot_track_get(sp,&best.target,sp->prover);
    if (!tr || tr->attempts>=sched->deep_reroot_max_attempts) return false;
    tr->attempts++; tr->last_attempt_work=best.target_work; best.attempt=tr->attempts;
    /* Require fresh parallel pressure before launching another episode from
       the same hot anchor; this prevents one region monopolizing the reroot worker. */
    if (best_anchor_track) {
        best_anchor_track->exposure_sec=0.0; best_anchor_track->start_work=best_anchor_track->last_work;
        best_anchor_track->start_pn=best_anchor_track->last_pn; best_anchor_track->start_dn=best_anchor_track->last_dn;
        best_anchor_track->first_seen_time=now;
    }
    sp->hot_diag_candidate=true; *out=best; return true;
}

static bool sp_maybe_queue_reroot(Context *sched, ParallelShared *sp)
{
    double now=wall_time();
    if (now < sp->hot_next_probe) return false;
    sp->hot_next_probe=now+sched->deep_reroot_hot_probe_sec;
    SPRerootRequest rr;
    if (!sp_build_reroot_request(sched,sp,&rr)) return false;
    sp->reroot_request=rr; sp->reroot_pending=true;
    atomic_fetch_add_explicit(&sp->reroot_triggers,1,memory_order_relaxed);
    char path[256]; sp_reroot_path_text(&rr,path,sizeof(path));
    printf("[SP-reroot] QUEUE path=%s depth=%u regionDepth=%u hotAge=%.1fs hotWork=%" PRIu64
           " nodeStall=%" PRIu64 " hotDelta=%.1f%%/%.1f%% busy=%u work=%" PRIu64
           " share(root/inner)=%.1f%%/%.1f%% attempt=%u/%u\n",
           path,rr.depth,rr.hot_region_depth,rr.hot_age_sec,rr.hot_region_work,rr.target_stall_work,
           100.0*rr.hot_delta_pn,100.0*rr.hot_delta_dn,rr.hot_busy,rr.target_work,
           100.0*rr.root_share,100.0*rr.inner_share,rr.attempt,sched->deep_reroot_max_attempts);
    fflush(stdout);
    return true;
}

/* v9.8.2: follow the retained reroot subsolver's CURRENT MPN chain.
   The private TT best move is authoritative when its profiled child exists;
   if that exact edge has already fallen out of the TT, fall back to the
   highest-work profiled child.  The fallback keeps harvesting robust to TT
   replacement without exporting any heuristic value to the exact proof. */
static int32_t deep_harvest_mpn_child(Context *ctx, int32_t parent, bool *tt_edge_out)
{
    if (tt_edge_out) *tt_edge_out = false;
    if (!ctx || !ctx->deep_frontier.node || parent < 0 ||
        (uint32_t)parent >= ctx->deep_frontier.count) return -1;
    uint8_t mv = 0;
    if (tt_best_move_peek(ctx, &ctx->deep_frontier.node[parent].board, &mv) && mv) {
        for (int32_t c = ctx->deep_frontier.node[parent].first_child; c >= 0;
             c = ctx->deep_frontier.node[c].next_sibling) {
            if (!ctx->deep_frontier.node[c].valid) continue;
            if (ctx->deep_frontier.node[c].move == mv) {
                if (tt_edge_out) *tt_edge_out = true;
                return c;
            }
        }
    }

    int32_t best = -1;
    uint64_t best_work = 0, best_stall = 0;
    for (int32_t c = ctx->deep_frontier.node[parent].first_child; c >= 0;
         c = ctx->deep_frontier.node[c].next_sibling) {
        if (!ctx->deep_frontier.node[c].valid) continue;
        WorkStamp ew = deep_frontier_effective_stamp(ctx, c);
        uint64_t st = deep_frontier_stall(ctx, c);
        if (best < 0 || ew.expansions > best_work ||
            (ew.expansions == best_work && st > best_stall)) {
            best = c; best_work = ew.expansions; best_stall = st;
        }
    }
    return best;
}

/* v9.8.2 shallow-first ladder budget.  The shallowest configured rung gets
   the sticky budget (default two 5 s slices); every deeper rung gets one
   exploratory slice.  This encodes the empirical v9.7.6 result that +4
   descendants produced all natural exact harvests, while +5/+6 consumed
   substantial work without exact closure. */
static unsigned deep_harvest_attempt_budget(const Context *outer, unsigned depth)
{
    if (!outer) return 1;
    if (depth <= outer->deep_reroot_harvest_min_depth) {
        unsigned b = outer->deep_reroot_harvest_attempts;
        return b ? b : 1;
    }
    return 1;
}

/* Select the SHALLOWEST eligible unresolved node on the current MPN chain.
   Work/stall/score are diagnostics only; they no longer rank +4/+5/+6 against
   each other.  Once the shallow rung exhausts its budget, the next call in the
   same harvest ladder advances to +5 and then +6. */
static int32_t deep_harvest_candidate(Context *outer, unsigned *depth_out,
                                      uint64_t *work_out, uint64_t *stall_out,
                                      double *score_out, unsigned *tt_edges_out,
                                      unsigned *gate_skip_depth_out, uint64_t *gate_work_out,
                                      double *gate_stability_out)
{
    if (gate_skip_depth_out) *gate_skip_depth_out = 0;
    if (gate_work_out) *gate_work_out = 0;
    if (gate_stability_out) *gate_stability_out = 0.0;
    if (!outer || !outer->deep_reroot_harvest_enabled ||
        !outer->deep_frontier_enabled || !outer->deep_frontier.node ||
        outer->deep_reroot_level >= outer->deep_reroot_max_nesting)
        return -1;

    unsigned mind = outer->deep_reroot_harvest_min_depth;
    unsigned maxd = outer->deep_reroot_harvest_max_depth;
    if (mind < 1) mind = 1;
    if (maxd > outer->deep_frontier.max_depth) maxd = outer->deep_frontier.max_depth;
    if (maxd > DEEP_FRONTIER_HARD_MAX_DEPTH) maxd = DEEP_FRONTIER_HARD_MAX_DEPTH;
    if (mind > maxd) return -1;

    int32_t cand[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    uint64_t work[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    uint64_t stall[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    unsigned tt_prefix[DEEP_FRONTIER_HARD_MAX_DEPTH + 1];
    memset(cand, 0xff, sizeof(cand));
    memset(work, 0, sizeof(work));
    memset(stall, 0, sizeof(stall));
    memset(tt_prefix, 0, sizeof(tt_prefix));

    int32_t cur = 0;
    uint64_t max_work = 0;
    unsigned chain_tt_edges = 0;
    for (unsigned d = 1; d <= maxd; ++d) {
        bool used_tt = false;
        cur = deep_harvest_mpn_child(outer, cur, &used_tt);
        if (cur < 0) break;
        if (used_tt) chain_tt_edges++;
        if (d < mind) continue;
        DeepFrontierNode *n = &outer->deep_frontier.node[cur];
        unsigned budget = deep_harvest_attempt_budget(outer, d);
        if (n->reroot_result_valid || (unsigned)n->harvest_attempts >= budget)
            continue;
        uint32_t pn = 1, dn = 1;
        if (tt_pair_peek(outer, &n->board, &pn, &dn) && (pn == 0 || dn == 0))
            continue;
        WorkStamp ew = deep_frontier_effective_stamp(outer, cur);
        if (ew.expansions < outer->deep_reroot_harvest_min_work) continue;
        cand[d] = cur;
        work[d] = ew.expansions;
        stall[d] = deep_frontier_stall(outer, cur);
        tt_prefix[d] = chain_tt_edges;
        if (work[d] > max_work) max_work = work[d];
    }
    if (!max_work) return -1;

    for (unsigned d = mind; d <= maxd; ++d) {
        if (cand[d] < 0 || work[d] == 0) continue;
        double stability = (double)stall[d] / (double)work[d];
        if (stability > 1.0) stability = 1.0;
        /* v9.8.2 evidence gate: only the deepest rung (+6 by default) needs
           strong evidence.  +4/+5 keep v9.7.7 behavior.  The gate requires
           both enough accumulated profiler work and a large unchanged-work
           fraction, avoiding expensive +6 probes that are merely busy rather
           than persistently stalled. */
        if (d == maxd && d > mind + 1u &&
            (work[d] < outer->deep_reroot_harvest_deep_min_work ||
             stability < outer->deep_reroot_harvest_deep_min_stability)) {
            if (gate_skip_depth_out) *gate_skip_depth_out = d;
            if (gate_work_out) *gate_work_out = work[d];
            if (gate_stability_out) *gate_stability_out = stability;
            continue;
        }
        double work_norm = (double)work[d] / (double)max_work;
        double depth_norm = maxd > mind ? (double)(d - mind) / (double)(maxd - mind) : 0.0;
        bool retained = deep_reroot_session_context(outer, &outer->deep_frontier.node[cand[d]].board) != NULL;
        double resume_bonus = retained ? 0.75 : 0.0;
        double diag_score = 4.0 * stability + 1.25 * work_norm + 0.75 * depth_norm + resume_bonus;
        if (depth_out) *depth_out = d;
        if (work_out) *work_out = work[d];
        if (stall_out) *stall_out = stall[d];
        if (score_out) *score_out = diag_score;
        if (tt_edges_out) *tt_edges_out = tt_prefix[d];
        return cand[d];
    }
    return -1;
}

/* v9.8.2 proof-pressure gate for the second +4 sticky slice.
   PN/DN are NOT treated as evaluation values here.  The smaller finite proof
   number is used only as a scheduler signal for whether another short
   fresh-root proof slice is worth spending.  Exact zeros are handled before
   this helper is called. */
static bool deep_harvest_pressure_allows_resume(const Context *outer, uint32_t pn, uint32_t dn)
{
    if (!outer) return false;
    uint32_t pressure = pn < dn ? pn : dn;
    return pressure <= outer->deep_reroot_harvest_resume_pressure;
}

/* v9.8.2 shallow-first exact-harvest ladder.

   At each outer harvest opportunity, freeze the current MPN chain and consume
   eligible rungs in this order:
       shallow (+minDepth, normally +4): fresh 5 s -> resume 5 s only when
                                            min(pn,dn) <= pressure threshold
       deeper  (+5, +6):                 one fresh/resume 5 s slice each

   If a rung becomes exact, stop immediately and use the v9.7.4 promotion path.
   If it stays UNKNOWN and exhausts its rung budget, advance immediately to the
   next deeper rung instead of waiting for another 30 s outer slice.  Finite
   PN/DN remain private; only exact pn==0/dn==0 reach the shared TT. */
static int sp_adaptive_exact_harvest(Context *rrctx, Context *outer,
                                     ParallelShared *sp, unsigned outer_slice,
                                     uint64_t outer_uses, const char *outer_path)
{
    if (!rrctx || !outer || !sp || !outer->deep_reroot_harvest_enabled) return -3;
    if (outer_uses < outer->deep_reroot_harvest_after_slices) return -3;

    unsigned ladder_targets = 0;
    for (;;) {
        if (g_interrupted || atomic_load_explicit(&sp->solved, memory_order_acquire) != 0) return -1;
        if (rrctx->deadline > 0.0 && wall_time() >= rrctx->deadline) return -1;

        unsigned depth = 0, tt_edges = 0, gate_skip_depth = 0;
        uint64_t work = 0, stall = 0, gate_work = 0;
        double score = 0.0, gate_stability = 0.0;
        int32_t idx = deep_harvest_candidate(outer, &depth, &work, &stall, &score, &tt_edges,
                                             &gate_skip_depth, &gate_work, &gate_stability);
        if (idx < 0) {
            if (gate_skip_depth) {
                atomic_fetch_add_explicit(&sp->reroot_harvest_deep_gate_skips, 1, memory_order_relaxed);
                printf("[rr-harvest] GATE-SKIP outer=%s slice=%u use=%" PRIu64
                       " deep=+%u work=%" PRIu64 " stable=%.1f%% requireWork>=%" PRIu64
                       " requireStable>=%.1f%%\n",
                       outer_path ? outer_path : "?", outer_slice, outer_uses, gate_skip_depth, gate_work,
                       100.0 * gate_stability, outer->deep_reroot_harvest_deep_min_work,
                       100.0 * outer->deep_reroot_harvest_deep_min_stability);
            }
            if (ladder_targets == 0) {
                printf("[rr-harvest] NO-CANDIDATE outer=%s after=%u slice=%u use=%" PRIu64
                       " ladder=+%u..+%u minWork=%" PRIu64 "\n",
                       outer_path ? outer_path : "?", outer->deep_reroot_harvest_after_slices,
                       outer_slice, outer_uses, outer->deep_reroot_harvest_min_depth,
                       outer->deep_reroot_harvest_max_depth, outer->deep_reroot_harvest_min_work);
            } else {
                printf("[rr-harvest] LADDER-END outer=%s slice=%u use=%" PRIu64
                       " targets=%u no-more-eligible-rungs\n",
                       outer_path ? outer_path : "?", outer_slice, outer_uses, ladder_targets);
            }
            fflush(stdout);
            return ladder_targets == 0 ? -3 : -1;
        }

        DeepFrontierNode *n = &outer->deep_frontier.node[idx];
        unsigned max_attempts = deep_harvest_attempt_budget(outer, depth);
        if ((unsigned)n->harvest_attempts >= max_attempts) continue;
        unsigned remaining = max_attempts - (unsigned)n->harvest_attempts;
        unsigned focus = (depth == outer->deep_reroot_harvest_min_depth)
                       ? outer->deep_reroot_harvest_focus_slices : 1u;
        if (focus < 1) focus = 1;
        if (focus > remaining) focus = remaining;

        atomic_fetch_add_explicit(&sp->reroot_harvest_triggers, 1, memory_order_relaxed);
        outer->stats.deep_reroot_triggers++;
        ladder_targets++;

        Board promote_path[SP_PATH_MAX];
        unsigned promote_path_len = deep_reroot_global_path(outer, idx, promote_path);
        char local_path[192]; deep_frontier_path_text(outer, idx, local_path, sizeof(local_path));
        double stability = work ? 100.0 * (double)stall / (double)work : 0.0;
        if (stability > 100.0) stability = 100.0;
        printf("[rr-harvest] TRIGGER outer=%s outerSlice=%u outerUse=%" PRIu64 " deep=%s +%u"
               " ladder=%u/%u work=%" PRIu64 " stall=%" PRIu64
               " stable=%.1f%% diagScore=%.3f ttEdges=%u sticky=%ux%.1fs used=%u/%u\n",
               outer_path ? outer_path : "?", outer_slice, outer_uses, local_path, depth,
               depth - outer->deep_reroot_harvest_min_depth + 1,
               outer->deep_reroot_harvest_max_depth - outer->deep_reroot_harvest_min_depth + 1,
               work, stall, stability, score, tt_edges, focus, outer->deep_reroot_harvest_time_sec,
               (unsigned)n->harvest_attempts, max_attempts);
        fflush(stdout);

        unsigned ran = 0;
        for (unsigned hs = 1; hs <= focus; ++hs) {
            if (g_interrupted || atomic_load_explicit(&sp->solved, memory_order_acquire) != 0) break;
            if (rrctx->deadline > 0.0 && wall_time() >= rrctx->deadline) break;

            n->harvest_attempts++;
            ran++;
            atomic_fetch_add_explicit(&sp->reroot_harvest_attempts, 1, memory_order_relaxed);

            uint32_t pn = 1, dn = 1;
            uint64_t calls = 0, exp = 0, probes = 0, hits = 0, uses = 0;
            double sec = 0.0; bool resumed = false;

            double saved_time = outer->deep_reroot_time_sec;
            double saved_deadline = outer->deadline;
            outer->deep_reroot_time_sec = outer->deep_reroot_harvest_time_sec;
            outer->deadline = rrctx->deadline;
            int result = deep_reroot_persistent_solve(outer, &n->board,
                                                       promote_path_len ? promote_path : NULL,
                                                       promote_path_len,
                                                       &pn, &dn, &calls, &exp, &probes, &hits,
                                                       &sec, &resumed, &uses);
            outer->deep_reroot_time_sec = saved_time;
            outer->deadline = saved_deadline;

            atomic_fetch_add_explicit(&sp->reroot_sub_calls, calls, memory_order_relaxed);
            atomic_fetch_add_explicit(&sp->reroot_sub_expansions, exp, memory_order_relaxed);
            atomic_fetch_add_explicit(&sp->reroot_sub_ms, (uint64_t)(sec * 1000.0 + 0.5), memory_order_relaxed);
            atomic_fetch_add_explicit(&sp->reroot_harvest_sub_expansions, exp, memory_order_relaxed);
            atomic_fetch_add_explicit(&sp->reroot_harvest_sub_ms, (uint64_t)(sec * 1000.0 + 0.5), memory_order_relaxed);
            if (resumed) atomic_fetch_add_explicit(&sp->reroot_harvest_resume, 1, memory_order_relaxed);
            else atomic_fetch_add_explicit(&sp->reroot_harvest_fresh, 1, memory_order_relaxed);

            outer->stats.deep_reroot_attempts++;
            outer->stats.deep_reroot_slices++;
            outer->stats.deep_reroot_sub_calls += calls;
            outer->stats.deep_reroot_sub_expansions += exp;
            outer->stats.deep_reroot_sub_ms += (uint64_t)(sec * 1000.0 + 0.5);
            outer->stats.deep_reroot_sub_tt_probes += probes;
            outer->stats.deep_reroot_sub_tt_hits += hits;

            double tt = probes ? 100.0 * (double)hits / (double)probes : 0.0;
            if (result >= 0) {
                n->reroot_result_valid = true;
                n->reroot_result = (uint8_t)result;
                reroot_overlay_store(outer, &n->board, pn, dn);
                outer->stats.deep_reroot_proofs++;
                if (result) outer->stats.deep_reroot_proved++;
                else outer->stats.deep_reroot_disproved++;
                outer->stats.oracle_collapses++;
                outer->deep_reroot_unwind = true;
                atomic_fetch_add_explicit(&sp->reroot_harvest_exact, 1, memory_order_relaxed);
                if (outer->parallel_exact_shared && promote_path_len > 0)
                    sp_promote_exact_descendant(outer, &n->board, pn, dn,
                                                promote_path, promote_path_len);
                printf("[rr-harvest] EXACT outer=%s deep=%s +%u ladder=%u sticky=%u/%u mode=%s use=%" PRIu64
                       " proposition=%s pn/dn=(%u,%u) calls=%" PRIu64 " exp=%" PRIu64
                       " tt=%.1f%% sec=%.3f -> overlay+promotion\n",
                       outer_path ? outer_path : "?", local_path, depth,
                       depth - outer->deep_reroot_harvest_min_depth + 1,
                       hs, focus, resumed ? "resume" : "fresh", uses,
                       result ? "PROVED" : "DISPROVED", pn, dn, calls, exp, tt, sec);
                fflush(stdout);
                return result;
            }
            if (result == -2) {
                printf("[rr-harvest] ERROR outer=%s deep=%s allocation failure; harvesting disabled for this session\n",
                       outer_path ? outer_path : "?", local_path);
                outer->deep_reroot_harvest_enabled = false;
                fflush(stdout);
                return -2;
            }

            outer->stats.deep_reroot_unknown++;
            outer->stats.deep_reroot_slice_unknown++;
            atomic_fetch_add_explicit(&sp->reroot_harvest_unknown, 1, memory_order_relaxed);

            bool pressure_skip = false;
            uint32_t pressure = pn < dn ? pn : dn;
            if (depth == outer->deep_reroot_harvest_min_depth && hs < focus &&
                !deep_harvest_pressure_allows_resume(outer, pn, dn)) {
                pressure_skip = true;
                /* Consume the unused sticky slot so this same descendant is not
                   reconsidered on a later outer slice.  Global attempt telemetry
                   still counts only the slice that actually ran. */
                n->harvest_attempts = (uint8_t)max_attempts;
                atomic_fetch_add_explicit(&sp->reroot_harvest_pressure_skips, 1, memory_order_relaxed);
            }

            printf("[rr-harvest] UNKNOWN outer=%s deep=%s +%u ladder=%u sticky=%u/%u mode=%s use=%" PRIu64
                   " pn/dn=(%u,%u) calls=%" PRIu64 " exp=%" PRIu64
                   " tt=%.1f%% sec=%.3f retained%s\n",
                   outer_path ? outer_path : "?", local_path, depth,
                   depth - outer->deep_reroot_harvest_min_depth + 1,
                   hs, focus, resumed ? "resume" : "fresh", uses, pn, dn, calls, exp, tt, sec,
                   pressure_skip ? "; pressure gate skips resume" : (hs < focus ? "; immediate sticky resume" : ""));
            if (pressure_skip) {
                printf("[rr-harvest] PRESSURE-SKIP outer=%s deep=%s +%u pressure=%u threshold=%u "
                       "pn/dn=(%u,%u); consume shallow resume budget and advance ladder\n",
                       outer_path ? outer_path : "?", local_path, depth, pressure,
                       outer->deep_reroot_harvest_resume_pressure, pn, dn);
            }
            fflush(stdout);
            if (pressure_skip) break;
        }

        if (ran == 0) {
            printf("[rr-harvest] ABORTED outer=%s deep=%s +%u no nested slice before global stop/deadline\n",
                   outer_path ? outer_path : "?", local_path, depth);
            fflush(stdout);
            return -1;
        }

        bool budget_exhausted = (unsigned)n->harvest_attempts >= max_attempts;
        if (budget_exhausted)
            atomic_fetch_add_explicit(&sp->reroot_harvest_budget_exhausted, 1, memory_order_relaxed);
        else
            atomic_fetch_add_explicit(&sp->reroot_harvest_focus_exhausted, 1, memory_order_relaxed);
        printf("[rr-harvest] %s outer=%s deep=%s +%u used=%u/%u retained%s\n",
               budget_exhausted ? "BUDGET-EXHAUSTED" : "FOCUS-EXHAUSTED",
               outer_path ? outer_path : "?", local_path, depth,
               (unsigned)n->harvest_attempts, max_attempts,
               budget_exhausted && depth < outer->deep_reroot_harvest_max_depth ? "; advance ladder" : "");
        fflush(stdout);

        if (!budget_exhausted) return -1;
        /* Budget exhausted: immediately select the next eligible, necessarily
           deeper rung on the frozen outer MPN chain. */
    }
}

static int sp_run_reroot_episode(Context *rrctx, ParallelShared *sp,
                                 const SPRerootRequest *rr,
                                 uint32_t *out_pn, uint32_t *out_dn)
{
    unsigned focus=rrctx->deep_reroot_focus_slices; if (focus<1) focus=1;
    char path[256]; sp_reroot_path_text(rr,path,sizeof(path));
    atomic_fetch_add_explicit(&sp->reroot_episodes,1,memory_order_relaxed);
    int episode=-1; uint32_t final_pn=1,final_dn=1;
    for (unsigned slice=1; slice<=focus; ++slice) {
        if (g_interrupted || atomic_load_explicit(&sp->solved,memory_order_acquire)!=0) break;
        if (sp->deadline>0.0 && wall_time()>=sp->deadline) break;
        uint32_t pn=1,dn=1; uint64_t calls=0,exp=0,probes=0,hits=0,uses=0; double sec=0.0; bool resumed=false;
        int result=deep_reroot_persistent_solve(rrctx,&rr->target,rr->path,rr->path_len,
                                                &pn,&dn,&calls,&exp,&probes,&hits,&sec,&resumed,&uses);
        final_pn=pn; final_dn=dn;
        atomic_fetch_add_explicit(&sp->reroot_slices,1,memory_order_relaxed);
        atomic_fetch_add_explicit(&sp->reroot_sub_calls,calls,memory_order_relaxed);
        atomic_fetch_add_explicit(&sp->reroot_sub_expansions,exp,memory_order_relaxed);
        atomic_fetch_add_explicit(&sp->reroot_sub_ms,(uint64_t)(sec*1000.0+0.5),memory_order_relaxed);
        if (resumed) atomic_fetch_add_explicit(&sp->reroot_resume_slices,1,memory_order_relaxed);
        else atomic_fetch_add_explicit(&sp->reroot_fresh_slices,1,memory_order_relaxed);
        double tt=probes?100.0*(double)hits/(double)probes:0.0;
        if (result>=0) {
            atomic_fetch_add_explicit(&sp->reroot_exact,1,memory_order_relaxed);
            printf("[SP-reroot] EXACT path=%s slice=%u/%u mode=%s use=%" PRIu64
                   " proposition=%s pn/dn=(%u,%u) calls=%" PRIu64 " exp=%" PRIu64 " tt=%.1f%% sec=%.3f\n",
                   path,slice,focus,resumed?"resume":"fresh",uses,result?"PROVED":"DISPROVED",
                   pn,dn,calls,exp,tt,sec); fflush(stdout); episode=result; break;
        }
        if (result==-2) {
            fprintf(stderr,"[SP-reroot] allocation failure; disabling further hybrid reroot requests.\n");
            rrctx->deep_reroot_enabled=false; episode=-2; break;
        }
        printf("[SP-reroot] UNKNOWN path=%s slice=%u/%u mode=%s use=%" PRIu64
               " pn/dn=(%u,%u) calls=%" PRIu64 " exp=%" PRIu64 " tt=%.1f%% sec=%.3f retained%s\n",
               path,slice,focus,resumed?"resume":"fresh",uses,pn,dn,calls,exp,tt,sec,
               slice<focus?"; adaptive-harvest check then resume":""); fflush(stdout);

        /* v9.8.2 adaptive recursion: do not wait for the serial node-stall
           trigger to happen inside a 30-second outer slice.  Once the outer
           session has already failed to close after N slices, inspect that
           session's retained private MPN chain, choose one descendant 4..6
           plies deeper, and keep that same persistent target for a focused
           fresh -> resume -> resume harvest batch. */
        Context *outer = deep_reroot_session_context(rrctx, &rr->target);
        if (outer && outer->deep_reroot_harvest_enabled &&
            uses >= outer->deep_reroot_harvest_after_slices &&
            atomic_load_explicit(&sp->solved, memory_order_acquire) == 0) {
            (void)sp_adaptive_exact_harvest(rrctx, outer, sp, slice, uses, path);
            if (atomic_load_explicit(&sp->solved, memory_order_acquire) != 0) break;
        }
    }
    if (episode<0 && episode!=-2) atomic_fetch_add_explicit(&sp->reroot_unknown,1,memory_order_relaxed);
    if (out_pn) *out_pn=final_pn;
    if (out_dn) *out_dn=final_dn;
    return episode;
}

static uint64_t tss_budget_for_level(const Context *ctx, unsigned level)
{
    uint64_t b = ctx->tss_budget;
    for (unsigned i = 0; i < level; ++i) {
        if (ctx->tss_budget_mult <= 1) break;
        if (b > UINT64_MAX / ctx->tss_budget_mult) return UINT64_MAX;
        b *= ctx->tss_budget_mult;
    }
    return b;
}

static inline bool sp_job_budget_hit(const Context *ctx)
{
    return ctx->parallel_worker && ctx->parallel_job_budget > 0 &&
           ctx->stats.dfpn_calls - ctx->parallel_job_start_calls >= ctx->parallel_job_budget;
}

static void dfpn(Context *ctx, const Board *b, uint32_t thpn, uint32_t thdn, unsigned depth)
{
    if (ctx->stop || sp_job_budget_hit(ctx)) return;
    const uint64_t call_start_exp = ctx->stats.expansions;
    ctx->stats.dfpn_calls++;
    if (ctx->parallel_worker && depth < SP_PATH_MAX) ctx->parallel_path[depth] = *b;
    if (b->ply > ctx->stats.max_ply) ctx->stats.max_ply = b->ply;
    maybe_stop_or_report(ctx);
    if (ctx->stop) return;
    if (ctx->deep_reroot_unwind && depth > 0) return;

    uint32_t db_pn=1, db_dn=1;
    if (ctx->exact_db && exact_db_lookup_pair(ctx->exact_db, b, ctx->prover, &db_pn, &db_dn)) {
        bool dbhit=false, dbmir=false; TTEntry *dbe=tt_access(ctx,b,true,&dbhit,&dbmir); (void)dbhit; (void)dbmir;
        dbe->pn=db_pn; dbe->dn=db_dn;
        sp_context_publish(ctx,b,db_pn,db_dn);
        if (depth==0) { ctx->root_pn=db_pn; ctx->root_dn=db_dn; }
        return;
    }

    uint32_t overlay_pn = 1, overlay_dn = 1;
    if (reroot_overlay_lookup(ctx, b, &overlay_pn, &overlay_dn)) {
        ctx->stats.deep_reroot_overlay_hits++;
        sp_context_publish(ctx, b, overlay_pn, overlay_dn);
        if (depth == 0) { ctx->root_pn = overlay_pn; ctx->root_dn = overlay_dn; }
        return;
    }

    bool hit, mirrored;
    TTEntry *entry = tt_access(ctx, b, true, &hit, &mirrored);
    if (entry->pn == 0 || entry->dn == 0) {
        sp_context_publish(ctx, b, entry->pn, entry->dn);
        if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
        return;
    }

    /* Exact dead-prover test, v5: run only near the endgame.  v4 measured
       millions of full viability scans with zero early-game prunes. */
    if (ctx->dependency_enabled && ctx->dep_dead_empty > 0 &&
        b->ply >= ctx->dep_deadcheck_ply &&
        board_empty_cells(b) <= ctx->dep_dead_empty &&
        !(entry->flags & TT_FLAG_DEP_DEAD_CHECKED)) {
        entry->flags |= TT_FLAG_DEP_DEAD_CHECKED;
        ctx->stats.dep_dead_checks++;
        if (!player_has_viable_segment(ctx, b, ctx->prover)) {
            entry->pn = PN_INF;
            entry->dn = 0;
            ctx->stats.dep_dead_prunes++;
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
    }

    /*
     * Adaptive threat-chain oracle.  Level 0 is cheap; revisited hard nodes
     * receive progressively deeper/larger proof attempts.  Failure is never
     * used as a game-theoretic conclusion, so this remains sound/incomplete.
     */
    unsigned effective_tss_levels = ctx->tss_levels;
    bool hot_gated_here = ctx->hot_tss_gate_enabled && ctx->hot_tss_gate_valid &&
                          board_descendant_of(b, &ctx->hot_tss_gate_board);
    if (hot_gated_here && ctx->hot_tss_level_cap < effective_tss_levels)
        effective_tss_levels = ctx->hot_tss_level_cap;
    if (hot_gated_here && entry->tss_level >= effective_tss_levels &&
        entry->tss_level < ctx->tss_levels) {
        entry->tss_level = (uint8_t)ctx->tss_levels;
        ctx->stats.hot_tss_gated_nodes++;
    }
    if (ctx->tss_enabled && b->ply >= ctx->tss_start_ply &&
        entry->tss_level < effective_tss_levels) {
        unsigned level = entry->tss_level++;
        if (level > 0) ctx->stats.tss_retries++;
        ctx->stats.tss_calls++;
        unsigned depth_limit = ctx->tss_depth + level * ctx->tss_depth_step;
        uint64_t budget = tss_budget_for_level(ctx, level);
        if (tss_prove(ctx, b, b->turn, depth_limit, &budget)) {
            proposition_pair(b->turn, ctx->prover, &entry->pn, &entry->dn);
            ctx->stats.tss_proofs++;
            ctx->stats.oracle_collapses++;
            if (level > 0) ctx->stats.tss_deep_proofs++;
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
        if (ctx->stop) return;
    }

    /* v4 dependency-aware quiet proof oracle.  It is sound/incomplete:
       successful proofs collapse the node; failures fall through unchanged. */
    if (ctx->dependency_enabled && ctx->dep_oracle_enabled &&
        b->ply >= ctx->dep_oracle_start_ply && !ctx->stop &&
        !(entry->flags & TT_FLAG_DEP_ORACLE_TRIED)) {
        entry->flags |= TT_FLAG_DEP_ORACLE_TRIED;
        ctx->stats.dep_oracle_calls++;
        uint64_t budget = ctx->dep_oracle_budget;
        unsigned tss_depth = ctx->tss_depth + ctx->tss_depth_step;
        if (dependency_prove(ctx, b, b->turn, ctx->dep_oracle_layers,
                             ctx->dep_oracle_candidates, tss_depth, &budget)) {
            proposition_pair(b->turn, ctx->prover, &entry->pn, &entry->dn);
            ctx->stats.dep_oracle_proofs++;
            ctx->stats.oracle_collapses++;
            tss_mark_proven(ctx, b, b->turn);
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
        if (ctx->stop) return;
    }

    /*
     * v3 bridge oracle: try to prove selected quiet setup moves without
     * exposing all of them to DF-PN.  Only successful proofs are consumed;
     * failure falls through to the exact full child set.
     */
    if (ctx->bridge_enabled && b->ply >= ctx->bridge_start_ply &&
        entry->tss_level >= ctx->tss_levels &&
        entry->bridge_level < ctx->bridge_levels) {
        unsigned level = entry->bridge_level++;
        ctx->stats.bridge_calls++;
        unsigned layers = ctx->bridge_layers + level;
        unsigned candidates = ctx->bridge_candidates + 2u * level;
        unsigned tss_depth = ctx->tss_depth + (level + 1u) * ctx->tss_depth_step;
        uint64_t budget = scaled_budget(ctx->bridge_budget, ctx->bridge_budget_mult, level);
        if (bridge_prove(ctx, b, b->turn, layers, candidates, tss_depth, &budget)) {
            proposition_pair(b->turn, ctx->prover, &entry->pn, &entry->dn);
            ctx->stats.bridge_proofs++;
            ctx->stats.oracle_collapses++;
            tss_mark_proven(ctx, b, b->turn);
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
        if (ctx->stop) return;
    }

    for (;;) {
        if (ctx->stop || sp_job_budget_hit(ctx)) return;

        /* Reacquire: recursive search may have replaced this TT bucket. */
        entry = tt_access(ctx, b, true, &hit, &mirrored);
        if (entry->pn == 0 || entry->dn == 0) {
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
        uint8_t tt_move = tt_best_move(ctx, entry, mirrored);

        bool first_expansion = !(entry->flags & TT_FLAG_EXPANDED);
        Child children[MAX_MOVES];
        int status = -1;
        int n = generate_children(ctx, b, children, tt_move, &status,
                                  first_expansion, entry, mirrored);
        if (status != -1) {
            proposition_pair(status, ctx->prover, &entry->pn, &entry->dn);
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }
        if (n <= 0) {
            proposition_pair(2, ctx->prover, &entry->pn, &entry->dn);
            sp_context_publish(ctx, b, entry->pn, entry->dn);
            if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
            return;
        }

        deep_frontier_note_branching(ctx, depth, n);
        if (depth == 0) frontier_init_root(ctx, children, n, b->turn == ctx->prover);

        if (first_expansion && ctx->tactical_enabled &&
            b->ply >= ctx->tactical_start_ply && ctx->tactical_budget > 0) {
            ctx->stats.tactical_calls++;
            uint64_t budget = ctx->tactical_budget;
            int limit = n < (int)ctx->tactical_candidates ? n : (int)ctx->tactical_candidates;
            int mover = b->turn;
            bool parent_won = false;

            for (int i = 0; i < limit && !ctx->stop && budget; ++i) {
                if (children[i].terminal_winner >= 0) continue;
                int winner = tactical_exact(ctx, &children[i].board,
                                            ctx->tactical_depth, &budget);
                if (winner < 0) continue;

                children[i].terminal_winner = (int8_t)winner;
                ctx->stats.tactical_children_solved++;
                if (winner == mover) ctx->stats.tactical_wins++;
                else if (winner == 2) ctx->stats.tactical_draws++;
                else ctx->stats.tactical_losses++;

                /* Persist the exact child result when possible. */
                bool chit, cmir;
                TTEntry *ce = tt_access(ctx, &children[i].board, true, &chit, &cmir);
                (void)chit; (void)cmir;
                proposition_pair(winner, ctx->prover, &ce->pn, &ce->dn);
                sp_context_publish(ctx, &children[i].board, ce->pn, ce->dn);

                if (winner == mover) {
                    parent_won = true;
                    break;
                }
            }

            /* Child insertions can replace the parent's bucket; reacquire it. */
            entry = tt_access(ctx, b, true, &hit, &mirrored);
            if (parent_won) {
                proposition_pair(mover, ctx->prover, &entry->pn, &entry->dn);
                sp_context_publish(ctx, b, entry->pn, entry->dn);
                ctx->stats.oracle_collapses++;
                if (depth == 0) { ctx->root_pn = entry->pn; ctx->root_dn = entry->dn; }
                return;
            }
        }

        if (!(entry->flags & TT_FLAG_EXPANDED)) {
            entry->flags |= TT_FLAG_EXPANDED;
            ctx->stats.expansions++;
            ctx->stats.expanded_children += (uint64_t)n;
        }

        uint32_t cp[MAX_MOVES], cd[MAX_MOVES];
        bool is_or = b->turn == ctx->prover;
        uint32_t node_pn = is_or ? PN_INF : 0;
        uint32_t node_dn = is_or ? 0 : PN_INF;
        int best = -1;
        uint32_t best_metric = PN_INF, second_metric = PN_INF;

        for (int i = 0; i < n; ++i) {
            child_pair(ctx, &children[i], &cp[i], &cd[i]);
            if (is_or) {
                if (cp[i] < node_pn) node_pn = cp[i];
                node_dn = sat_add(node_dn, cd[i]);
                uint32_t metric = cp[i];
                if (metric < best_metric) {
                    second_metric = best_metric;
                    best_metric = metric;
                    best = i;
                } else if (metric < second_metric) {
                    second_metric = metric;
                }
            } else {
                node_pn = sat_add(node_pn, cp[i]);
                if (cd[i] < node_dn) node_dn = cd[i];
                uint32_t metric = cd[i];
                if (metric < best_metric) {
                    second_metric = best_metric;
                    best_metric = metric;
                    best = i;
                } else if (metric < second_metric) {
                    second_metric = metric;
                }
            }
        }

        entry->pn = node_pn;
        entry->dn = node_dn;
        sp_context_publish(ctx, b, node_pn, node_dn);
        sp_periodic_hot_sync(ctx, depth);
        deep_frontier_note_pair(ctx, depth, node_pn, node_dn);
        if (best >= 0) tt_set_best(entry, mirrored, children[best].move);
        if (depth == 0) {
            ctx->root_pn = node_pn;
            ctx->root_dn = node_dn;
            uint8_t new_root_move = best >= 0 ? children[best].move : 0;
            if (ctx->root_best_move && new_root_move && new_root_move != ctx->root_best_move)
                ctx->stats.root_mpc_switches++;
            ctx->root_best_move = new_root_move;
            frontier_note_root_pair(ctx, node_pn, node_dn);
            /* v9.8.2 global scheduler is driven by maybe_stop_or_report(), so it
               also runs while a long recursive descent is still active. */
            maybe_stop_or_report(ctx);
        }

        uint64_t call_work = ctx->stats.expansions - call_start_exp;
        if (depth > 0 && call_work > ctx->stats.work_quantum_max_call_exp)
            ctx->stats.work_quantum_max_call_exp = call_work;

        if (node_pn == 0 || node_dn == 0 || node_pn >= thpn || node_dn >= thdn) return;
        if (best < 0 || children[best].terminal_winner >= 0) return;

        /* v9.8.2 work-quantum yield.
         *
         * The root invocation is intentionally unlimited: it is the scheduler
         * that must reaggregate child PN/DN and choose the next MPC.  Every
         * non-root invocation may spend at most roughly --work-quantum fresh
         * DF-PN expansions before yielding.  This check is placed only after
         * the current node PN/DN has been recomputed and written to the TT, so
         * an unwind never discards the newest exact search information.
         *
         * Because a descendant may finish a quantum before control returns
         * here, the observed call_work can exceed W slightly; that is expected.
         */
        if (depth > 0 && ctx->work_quantum > 0 && call_work >= ctx->work_quantum) {
            ctx->stats.work_quantum_yields++;
            ctx->stats.work_quantum_yield_expansions += call_work;
            return;
        }

        uint32_t c_thpn, c_thdn;
        if (is_or) {
            c_thpn = thpn < sat_inc(second_metric) ? thpn : sat_inc(second_metric);
            c_thdn = threshold_sub_add(thdn, node_dn, cd[best]);
        } else {
            c_thpn = threshold_sub_add(thpn, node_pn, cp[best]);
            c_thdn = thdn < sat_inc(second_metric) ? thdn : sat_inc(second_metric);
        }

        /* Guard against pathological zero progress due to saturation. */
        if (c_thpn <= cp[best] && cp[best] < PN_INF) c_thpn = sat_inc(cp[best]);
        if (c_thdn <= cd[best] && cd[best] < PN_INF) c_thdn = sat_inc(cd[best]);

        /* v6 effective branching instrumentation: count distinct child moves
           actually selected for recursive DF-PN, in canonical orientation. */
        ctx->stats.child_descents++;
        uint8_t examined_move = mirrored ? mirror_move(children[best].move) : children[best].move;
        uint32_t examined_bit = UINT32_C(1) << examined_move;
        if (!(entry->examined_mask & examined_bit)) {
            if (entry->examined_mask == 0) {
                ctx->stats.first_pick_nodes++;
                ctx->stats.first_pick_rank_sum += (uint64_t)(best + 1);
                if (best < 3) ctx->stats.first_pick_top3++;
            }
            entry->examined_mask |= examined_bit;
            ctx->stats.unique_children_examined++;
            ctx->stats.unique_rank_sum += (uint64_t)(best + 1);
            if (best < 3) ctx->stats.unique_top3_examined++;
        }

        bool profile_here = ctx->frontier_enabled && ctx->frontier.initialized && depth <= 1;
        WorkStamp frontier_before = {0}, frontier_after = {0};
        int frontier_ci = -1, frontier_ri = -1;
        if (profile_here) {
            frontier_before = work_stamp(ctx);
            if (depth == 0) {
                frontier_ci = frontier_child_by_move(ctx, children[best].move);
            } else {
                frontier_ci = frontier_child_by_board(ctx, b);
                if (frontier_ci >= 0) frontier_ri = frontier_reply_get(ctx, frontier_ci, &children[best]);
            }
        }

        /* v9.8.2 deep prefix accounting: attribute the full recursive work to
           every selected edge up to --deep-frontier-depth.  Active stamps
           make long in-flight descents visible to periodic snapshots. */
        int32_t deep_child_idx = -1;
        int32_t prev_deep_path = -1, prev_deep_active = -1;
        WorkStamp deep_before = {0}, prev_deep_start = {0};
        bool deep_profile_here = ctx->deep_frontier_enabled &&
                                 depth < ctx->deep_frontier_depth &&
                                 depth < DEEP_FRONTIER_HARD_MAX_DEPTH;
        if (deep_profile_here) {
            int32_t deep_parent = ctx->deep_path_node[depth];
            deep_child_idx = deep_frontier_child(ctx, deep_parent, &children[best], depth + 1);
            if (deep_child_idx >= 0) {
                unsigned dd = depth + 1;
                deep_before = work_stamp(ctx);
                prev_deep_path = ctx->deep_path_node[dd];
                prev_deep_active = ctx->deep_active_node[dd];
                prev_deep_start = ctx->deep_active_start[dd];
                ctx->deep_path_node[dd] = deep_child_idx;
                ctx->deep_active_node[dd] = deep_child_idx;
                ctx->deep_active_start[dd] = deep_before;
            }
        }

        /* v9.8.2: expose in-flight root/depth-2 work to the periodic global
           scheduler.  The committed counters below are still updated only
           after the recursive call returns, preserving the v8 accounting. */
        int prev_active_ci = ctx->active_frontier_ci;
        int prev_active_ri = ctx->active_frontier_ri;
        uint64_t prev_child_start = ctx->active_child_start_exp;
        uint64_t prev_reply_start = ctx->active_reply_start_exp;
        if (profile_here && depth == 0 && frontier_ci >= 0) {
            ctx->active_frontier_ci = frontier_ci;
            ctx->active_frontier_ri = -1;
            ctx->active_child_start_exp = ctx->stats.expansions;
        } else if (profile_here && depth == 1 && frontier_ci >= 0 && frontier_ri >= 0) {
            ctx->active_frontier_ci = frontier_ci;
            ctx->active_frontier_ri = frontier_ri;
            ctx->active_reply_start_exp = ctx->stats.expansions;
        }

        if (sp_job_budget_hit(ctx)) {
            if (deep_child_idx >= 0) {
                unsigned dd = depth + 1;
                ctx->deep_path_node[dd] = prev_deep_path;
                ctx->deep_active_node[dd] = prev_deep_active;
                ctx->deep_active_start[dd] = prev_deep_start;
            }
            ctx->active_frontier_ci = prev_active_ci;
            ctx->active_frontier_ri = prev_active_ri;
            ctx->active_child_start_exp = prev_child_start;
            ctx->active_reply_start_exp = prev_reply_start;
            return;
        }

        int hot_result = -1;
        if (depth == 1 && frontier_ci >= 0 && frontier_ri >= 0)
            hot_result = maybe_hot_exact(ctx, frontier_ci, frontier_ri, &children[best].board);
        if (hot_result < 0)
            dfpn(ctx, &children[best].board, c_thpn, c_thdn, depth + 1);

        if (deep_child_idx >= 0) {
            unsigned dd = depth + 1;
            WorkStamp deep_after = work_stamp(ctx);
            deep_frontier_add_work(ctx, deep_child_idx, deep_before, deep_after);
            ctx->deep_path_node[dd] = prev_deep_path;
            ctx->deep_active_node[dd] = prev_deep_active;
            ctx->deep_active_start[dd] = prev_deep_start;
        }

        ctx->active_frontier_ci = prev_active_ci;
        ctx->active_frontier_ri = prev_active_ri;
        ctx->active_child_start_exp = prev_child_start;
        ctx->active_reply_start_exp = prev_reply_start;

        if (profile_here) {
            frontier_after = work_stamp(ctx);
            if (depth == 0 && frontier_ci >= 0)
                frontier_add_child_work(ctx, frontier_ci, frontier_before, frontier_after);
            else if (depth == 1 && frontier_ci >= 0 && frontier_ri >= 0)
                frontier_add_reply_work(ctx, frontier_ci, frontier_ri, frontier_before, frontier_after);
        }

        /* A fresh reroot proof can be discovered while this descent is many
           frames below the solved prefix.  Unwind promptly to the root so the
           exact overlay value is reaggregated instead of continuing stale
           threshold work below an already-solved obligation. */
        if (ctx->deep_reroot_unwind) {
            if (depth > 0) return;
            ctx->deep_reroot_unwind = false;
            continue;
        }
    }
}

static void reset_phase(Context *ctx, uint8_t prover, const char *name)
{
    /* Persistent reroot sessions are proposition-specific.  Do not retain a
       W-phase subsolver when the prover changes for the L phase. */
    if (ctx->generation != 0 && ctx->prover != prover)
        deep_reroot_sessions_destroy(ctx);
    memset(&ctx->stats, 0, sizeof(ctx->stats));
    ctx->prover = prover;
    ctx->phase_name = name;
    ctx->generation++;
    if (ctx->generation == 0) ctx->generation = 1;
    ctx->phase_start = wall_time();
    ctx->next_report = ctx->phase_start + ctx->report_interval;
    ctx->root_pn = 1;
    ctx->root_dn = 1;
    ctx->root_best_move = 0;
    ctx->hot_tss_gate_valid = false;
    ctx->deep_reroot_unwind = false;
    memset(ctx->reroot_overlay, 0, sizeof(ctx->reroot_overlay));
    ctx->active_frontier_ci = -1;
    ctx->active_frontier_ri = -1;
    ctx->active_child_start_exp = 0;
    ctx->active_reply_start_exp = 0;
    frontier_reset(ctx);
}

/* 1 proven, 0 disproven, -1 stopped/unknown. */
static int run_phase(Context *ctx, const Board *root, uint8_t prover, const char *name,
                     PhaseSummary *summary)
{
    reset_phase(ctx, prover, name);
    if (ctx->exact_db) {
        size_t known=0,proved=0,disproved=0;
        uint8_t pb=exact_prop_proved_bit(prover), dbit=exact_prop_disproved_bit(prover);
        for (size_t i=0;i<ctx->exact_db->sets;++i) if (ctx->exact_db->slot[i].valid) {
            uint8_t m=ctx->exact_db->slot[i].prop_mask;
            if (m & pb) { proved++; known++; }
            else if (m & dbit) { disproved++; known++; }
        }
        printf("[exact-db] phase=%s prover=%s states=%zu known=%zu propositionProved=%zu propositionDisproved=%zu immutableLookup=on\n",
               name,player_name(prover),ctx->exact_db->count,known,proved,disproved);
    }
    deep_frontier_reset_phase(ctx, root);
    printf("\nPhase %s: prove that %s can force a win.\n", name, player_name(prover));
    if (ctx->exact_lemma_depth > 0 && ctx->exact_db) {
        uint64_t eb=ctx->exact_lemma_budget; uint32_t epn=1,edn=1;
        double e0=wall_time();
        int er=exact_lemma_closure(ctx,root,ctx->exact_lemma_depth,&eb,&epn,&edn);
        printf("[exact-lemma] depth=%u used=%" PRIu64 "/%" PRIu64 " sec=%.3f result=%s root=(%u,%u)\n",
               ctx->exact_lemma_depth,ctx->exact_lemma_budget-eb,ctx->exact_lemma_budget,wall_time()-e0,
               er==1?"PROVED":er==0?"DISPROVED":"UNKNOWN",epn,edn);
        if (er>=0) { ctx->root_pn=epn; ctx->root_dn=edn; }
        else dfpn(ctx, root, PN_INF, PN_INF, 0);
    } else dfpn(ctx, root, PN_INF, PN_INF, 0);

    if (!ctx->stop) {
        uint32_t pn, dn;
        tt_pair(ctx, root, &pn, &dn);
        ctx->root_pn = pn; ctx->root_dn = dn;
    }
    log_stats(ctx, ctx->stop ? "stopped" : "final");
    int verdict = -1;
    if (!ctx->stop && ctx->root_pn == 0) verdict = 1;
    else if (!ctx->stop && ctx->root_dn == 0) verdict = 0;
    if (summary) {
        summary->verdict = verdict;
        summary->elapsed = wall_time() - ctx->phase_start;
        summary->stats = ctx->stats;
        summary->root_pn = ctx->root_pn;
        summary->root_dn = ctx->root_dn;
        summary->root_best_move = ctx->root_best_move;
    }
    return verdict;
}


static bool sp_worker_context_init(Context *w, const Context *base, ParallelShared *sp,
                                   unsigned worker_id, const Options *opt)
{
    *w=*base;
    memset(&w->tt,0,sizeof(w->tt));
    memset(&w->child_cache,0,sizeof(w->child_cache));
    memset(&w->hot_tt,0,sizeof(w->hot_tt));
    memset(&w->frontier,0,sizeof(w->frontier));
    memset(&w->deep_frontier,0,sizeof(w->deep_frontier));
    memset(&w->stats,0,sizeof(w->stats));
    memset(w->reroot_overlay,0,sizeof(w->reroot_overlay));
    memset(w->reroot_session,0,sizeof(w->reroot_session));
    memset(w->deep_path_node,0xff,sizeof(w->deep_path_node));
    memset(w->deep_active_node,0xff,sizeof(w->deep_active_node));
    w->generation=0;
    w->frontier_enabled=false;
    w->deep_frontier_enabled=false;
    w->hot_exact_enabled=false;
    w->deep_reroot_enabled=false;
    w->parallel_shared=sp;
    w->parallel_stop_shared=sp;
    w->parallel_worker=true;
    w->parallel_worker_id=worker_id;
    w->parallel_publish_interval=sp->publish_work;
    w->parallel_next_publish_calls=sp->publish_work;
    w->parallel_publish_depth=sp->publish_depth;
    memset(w->parallel_path,0,sizeof(w->parallel_path));
    /* The parallel job budget is measured globally in DFPN calls.  Disable the
       older expansion-based work-quantum inside workers so every thread uses
       the same bounded-job semantics, including the unified 1-thread mode. */
    w->work_quantum=0;
    w->log_path=NULL; w->frontier_log_path=NULL; w->deep_frontier_log_path=NULL;
    w->log_fp=NULL; w->frontier_fp=NULL; w->deep_frontier_fp=NULL;
    w->report_interval=1.0e30; w->next_report=1.0e30;
    w->start_time=sp->start_time; w->deadline=sp->deadline;
    w->stop=false;
    if(!tt_init(&w->tt,opt->parallel_worker_tt_mb/1024.0)) return false;
    w->child_cache_enabled=base->child_cache_enabled && opt->parallel_worker_cache_mb>0.0;
    if(w->child_cache_enabled && !child_cache_init(&w->child_cache,opt->parallel_worker_cache_mb)){
        tt_destroy(&w->tt); return false;
    }
    reset_phase(w,sp->prover,"SP");
    w->start_time=sp->start_time; w->deadline=sp->deadline; w->next_report=1.0e30;
    return true;
}

static void sp_worker_context_destroy(Context *w)
{
    deep_reroot_sessions_destroy(w);
    child_cache_destroy(&w->child_cache);
    tt_destroy(&w->tt);
}

static void sp_summary_add_stats(Stats *dst, const Stats *src)
{
#define ADDF(x) dst->x += src->x
    ADDF(dfpn_calls); ADDF(expansions); ADDF(child_generations); ADDF(moves_generated);
    ADDF(child_cache_probes); ADDF(child_cache_hits); ADDF(child_cache_stores); ADDF(child_cache_restored);
    ADDF(tt_probes); ADDF(tt_hits); ADDF(tt_replacements);
    ADDF(tss_calls); ADDF(tss_nodes); ADDF(tss_proofs); ADDF(tss_retries); ADDF(tss_deep_proofs);
    ADDF(tss_cache_hits); ADDF(tss_forcing_candidates); ADDF(tss_candidates_kept); ADDF(tss_signature_pruned);
    ADDF(sibling_dedup_pruned); ADDF(dep_maps); ADDF(dep_segments_tested); ADDF(dep_viable_green); ADDF(dep_viable_yellow);
    ADDF(dep_ordered_moves); ADDF(dep_dead_prunes); ADDF(dep_oracle_calls); ADDF(dep_oracle_nodes); ADDF(dep_oracle_proofs);
    ADDF(dep_oracle_candidates); ADDF(dep_oracle_defender_replies); ADDF(dep_oracle_budget_exhausted);
    ADDF(bridge_calls); ADDF(bridge_nodes); ADDF(bridge_proofs); ADDF(bridge_candidates); ADDF(bridge_defender_replies);
    ADDF(bridge_budget_exhausted); ADDF(oracle_collapses); ADDF(expanded_children); ADDF(child_descents);
    ADDF(unique_children_examined); ADDF(unique_top3_examined); ADDF(unique_rank_sum); ADDF(first_pick_nodes);
    ADDF(first_pick_top3); ADDF(first_pick_rank_sum); ADDF(dep_light_nodes); ADDF(dep_light_segments_tested);
    ADDF(dep_hint_uses); ADDF(dep_dead_checks); ADDF(tactical_calls); ADDF(tactical_nodes); ADDF(tactical_children_solved);
    ADDF(tactical_wins); ADDF(tactical_losses); ADDF(tactical_draws); ADDF(tactical_budget_exhausted);
    ADDF(work_quantum_yields); ADDF(work_quantum_yield_expansions);
    if(src->work_quantum_max_call_exp > dst->work_quantum_max_call_exp) dst->work_quantum_max_call_exp=src->work_quantum_max_call_exp;
    ADDF(forced_defenses); ADDF(double_threat_wins); ADDF(double_threat_losses); ADDF(unsafe_pruned);
    ADDF(symmetry_pruned); ADDF(immediate_terminals);
    if(src->max_ply > dst->max_ply) dst->max_ply=src->max_ply;
#undef ADDF
}

static uint8_t sp_root_best_move(Context *sched, ParallelShared *sp, uint32_t *pn, uint32_t *dn)
{
    Child children[MAX_MOVES]; int n=0; uint32_t second; uint8_t bm=0;
    sp_compute_pair(sched,sp,&sp->root,pn,dn,&bm,&second,children,&n);
    sp_shared_publish(sp,&sp->root,*pn,*dn);
    return bm;
}

static void sp_report_locked(Context *sched, ParallelShared *sp, const char *phase)
{
    double now=wall_time();
    if(now < sp->next_report) return;
    uint32_t pn=1,dn=1; uint8_t bm=sp_root_best_move(sched,sp,&pn,&dn);
    unsigned active=0; for(unsigned i=0;i<sp->threads;++i) if(sp->active[i].active) active++;
    uint64_t probes=atomic_load(&sp->shared_probes), hits=atomic_load(&sp->shared_hits);
    uint64_t exp=atomic_load(&sp->total_worker_expansions), calls=atomic_load(&sp->total_worker_calls);
    uint64_t js=atomic_load(&sp->jobs_started), jf=atomic_load(&sp->jobs_finished);
    char dur[32]; format_duration(now-sp->start_time,dur);
    printf("[SP %s %s] th=%u active=%u jobs=%" PRIu64 "/%" PRIu64
           " calls=%" PRIu64 " exp=%" PRIu64 " cps=%.0f sharedTT=%.1f%% root=(%u,%u)",
           phase,dur,sp->threads,active,jf,js,calls,exp,
           (now>sp->start_time)?(double)calls/(now-sp->start_time):0.0,
           probes?100.0*(double)hits/(double)probes:0.0,pn,dn);
    if(bm) printf(" mpc=%d%c",move_tile(bm),'A'+move_col(bm));
    printf(" sched=%" PRIu64 " waits=%" PRIu64 " sync=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " conc=%" PRIu64
           " hotAge=%.1f hotWork=%" PRIu64 " hotDelta=%.1f/%.1f hotRegion=%u hotDepth=%u hotBusy=%u rrCandidate=%u"
           " rr=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " rrSl=%" PRIu64 " rrExp=%" PRIu64 " rrAct=%u"
           " rrHarvest=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " hExp=%" PRIu64
           " hFocusEx=%" PRIu64 " hBudgetEx=%" PRIu64 " hGateSkip=%" PRIu64 " hPressureSkip=%" PRIu64
           " rrPromote=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " rrPconf=%" PRIu64
           " xdbHit=%" PRIu64 " xdbConf=%" PRIu64
           " fail(e/t/x/p)=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 "\n",
           atomic_load(&sp->scheduler_descents),atomic_load(&sp->scheduler_waits),
           atomic_load(&sp->periodic_sync_events),atomic_load(&sp->periodic_sync_nodes),
           atomic_load(&sp->periodic_sync_read_hits),atomic_load(&sp->periodic_sync_updates),
           atomic_load(&sp->concentrated_assignments),
           sp->hot_diag_valid?sp->hot_diag_age_sec:0.0,sp->hot_diag_valid?sp->hot_diag_work:0,
           sp->hot_diag_valid?100.0*sp->hot_diag_delta_pn:0.0,sp->hot_diag_valid?100.0*sp->hot_diag_delta_dn:0.0,
           sp->hot_diag_valid?sp->hot_diag_region_depth:0u,sp->hot_diag_valid?sp->hot_diag_depth:0u,sp->hot_diag_valid?sp->hot_diag_busy:0u,sp->hot_diag_candidate?1u:0u,
           atomic_load(&sp->reroot_triggers),atomic_load(&sp->reroot_episodes),
           atomic_load(&sp->reroot_exact),atomic_load(&sp->reroot_unknown),
           atomic_load(&sp->reroot_slices),atomic_load(&sp->reroot_sub_expansions),sp->reroot_active?1u:0u,
           atomic_load(&sp->reroot_harvest_triggers),atomic_load(&sp->reroot_harvest_attempts),
           atomic_load(&sp->reroot_harvest_exact),atomic_load(&sp->reroot_harvest_unknown),
           atomic_load(&sp->reroot_harvest_sub_expansions),
           atomic_load(&sp->reroot_harvest_focus_exhausted),atomic_load(&sp->reroot_harvest_budget_exhausted),
           atomic_load(&sp->reroot_harvest_deep_gate_skips),
           atomic_load(&sp->reroot_harvest_pressure_skips),
           atomic_load(&sp->reroot_promote_found),atomic_load(&sp->reroot_promote_inserted),
           atomic_load(&sp->reroot_promote_ancestor_changed),atomic_load(&sp->reroot_promote_conflicts),
           atomic_load(&sp->exact_db_hits),atomic_load(&sp->exact_db_conflicts),
           atomic_load(&sp->assign_fail_exact),atomic_load(&sp->assign_fail_threshold),
           atomic_load(&sp->assign_fail_terminal),atomic_load(&sp->assign_fail_pathmax));
    fflush(stdout);
    sp->next_report=now+sp->report_interval;
}

/* Shared-memory SPDFPN-style phase.  State assignment is serialized briefly
   under job_lock; actual bounded DFPN jobs run concurrently.  Exact values are
   shared immediately.  Finite PN/DN/work is exchanged at job boundaries and
   scheduler recomputation points to avoid turning the shared table into a
   lock-contention hotspot.  Active-subtree occupancy supplies a lightweight
   virtual scheduling metric only; it can change work allocation but can never
   create an exact proof/disproof. */
static int run_phase_parallel(Context *sched, const Options *opt, const Board *root,
                              uint8_t prover, const char *name, PhaseSummary *summary)
{
#ifndef _OPENMP
    (void)sched;(void)opt;(void)root;(void)prover;(void)name;(void)summary;
    fprintf(stderr,"Parallel mode requires an OpenMP build (-fopenmp).\n");
    return -1;
#else
    ParallelShared sp; memset(&sp,0,sizeof(sp));
    atomic_flag_clear(&sp.job_lock);
    if(!sp_shared_tt_init(&sp.tt,opt->parallel_shared_tt_gb)){
        fprintf(stderr,"Could not allocate v9.8.2 shared TT. Try smaller --shared-tt-gb.\n");
        return -1;
    }
    sp.root=*root; sp.prover=prover; sp.threads=opt->parallel_threads; sp.exact_db=sched->exact_db;
    sp.max_work_per_job=opt->parallel_job_work; sp.min_job_depth=opt->parallel_min_job_depth;
    sp.publish_work=opt->parallel_publish_work; sp.publish_depth=opt->parallel_publish_depth;
    sp.busy_penalty=opt->parallel_busy_penalty;
    sp.start_time=wall_time(); sp.deadline=sched->deadline;
    sp.report_interval=opt->report_interval; sp.next_report=sp.start_time+sp.report_interval;
    sp.hot_next_probe=sp.start_time;
    atomic_init(&sp.solved,0); atomic_init(&sp.jobs_started,0); atomic_init(&sp.jobs_finished,0);
    atomic_init(&sp.exact_db_hits,0); atomic_init(&sp.exact_db_conflicts,0);
    atomic_init(&sp.shared_probes,0); atomic_init(&sp.shared_hits,0); atomic_init(&sp.shared_publishes,0);
    atomic_init(&sp.scheduler_descents,0); atomic_init(&sp.scheduler_waits,0);
    atomic_init(&sp.assign_fail_exact,0); atomic_init(&sp.assign_fail_threshold,0);
    atomic_init(&sp.assign_fail_terminal,0); atomic_init(&sp.assign_fail_pathmax,0);
    atomic_init(&sp.total_worker_calls,0); atomic_init(&sp.total_worker_expansions,0);
    atomic_init(&sp.periodic_sync_events,0); atomic_init(&sp.periodic_sync_nodes,0);
    atomic_init(&sp.periodic_sync_read_hits,0); atomic_init(&sp.periodic_sync_updates,0);
    atomic_init(&sp.concentrated_assignments,0);
    atomic_init(&sp.reroot_triggers,0); atomic_init(&sp.reroot_episodes,0);
    atomic_init(&sp.reroot_exact,0); atomic_init(&sp.reroot_unknown,0);
    atomic_init(&sp.reroot_sub_calls,0); atomic_init(&sp.reroot_sub_expansions,0);
    atomic_init(&sp.reroot_sub_ms,0); atomic_init(&sp.reroot_fresh_slices,0);
    atomic_init(&sp.reroot_resume_slices,0); atomic_init(&sp.reroot_slices,0);
    atomic_init(&sp.reroot_harvest_triggers,0); atomic_init(&sp.reroot_harvest_attempts,0);
    atomic_init(&sp.reroot_harvest_exact,0); atomic_init(&sp.reroot_harvest_unknown,0);
    atomic_init(&sp.reroot_harvest_sub_expansions,0); atomic_init(&sp.reroot_harvest_sub_ms,0);
    atomic_init(&sp.reroot_harvest_fresh,0); atomic_init(&sp.reroot_harvest_resume,0);
    atomic_init(&sp.reroot_harvest_focus_exhausted,0); atomic_init(&sp.reroot_harvest_budget_exhausted,0);
    atomic_init(&sp.reroot_harvest_deep_gate_skips,0);
    atomic_init(&sp.reroot_harvest_pressure_skips,0);
    atomic_init(&sp.reroot_promote_found,0); atomic_init(&sp.reroot_promote_inserted,0);
    atomic_init(&sp.reroot_promote_ancestor_changed,0); atomic_init(&sp.reroot_promote_conflicts,0);

    reset_phase(sched,prover,name);
    sched->parallel_shared=&sp;
    if (sched->exact_db) (void)exact_db_report_phase(&sp,sched->exact_db,prover,name);
    if (sched->exact_lemma_depth > 0 && sched->exact_db) {
        uint64_t eb=sched->exact_lemma_budget; uint32_t epn=1,edn=1; double e0=wall_time();
        int er=exact_lemma_closure(sched,root,sched->exact_lemma_depth,&eb,&epn,&edn);
        printf("[exact-lemma] depth=%u used=%" PRIu64 "/%" PRIu64 " sec=%.3f result=%s root=(%u,%u)\n",
               sched->exact_lemma_depth,sched->exact_lemma_budget-eb,sched->exact_lemma_budget,wall_time()-e0,
               er==1?"PROVED":er==0?"DISPROVED":"UNKNOWN",epn,edn);
        if (er>=0) {
            if(summary){ memset(summary,0,sizeof(*summary)); summary->verdict=er; summary->elapsed=wall_time()-sp.start_time;
                summary->root_pn=epn; summary->root_dn=edn; }
            sched->root_pn=epn; sched->root_dn=edn; sched->parallel_shared=NULL;
            sp_shared_tt_destroy(&sp.tt); return er;
        }
    }
    /* v9.8.2: the scheduler owns hybrid reroot configuration and persistent
       sessions; normal SP workers themselves keep serial reroot disabled. */
    sched->hot_exact_enabled=false;
    sched->stop=false;
    printf("\nPhase %s: v9.8.2 shared-TT SPDFPN + proof-pressure/evidence-gated harvest ladder with %u workers.\n",name,sp.threads);
    printf("SPDFPN: sharedTT=%.3f GiB; workerTT=%.1f MiB; workerCache=%.1f MiB; MaxWorkPerJob=%" PRIu64 " DFPN-calls; minJobDepth=%u; hotSync=%" PRIu64 "/depth%u; busyPenalty=%.3f\n",
           sp.tt.allocated_gb,opt->parallel_worker_tt_mb,opt->parallel_worker_cache_mb,
           sp.max_work_per_job,sp.min_job_depth,sp.publish_work,sp.publish_depth,sp.busy_penalty);
    printf("SP-reroot: %s concurrent 1-worker persistent sessions=%u TT=%.3fGiB cache=%.1fMiB slice=%.1fs focus=%ux "
           "HOT-REGION total>=%" PRIu64 " age>=%.1fs hotWork>=%" PRIu64 " dPN/dDN-ref=%.1f%% minBusy=%u regionDepth<=%u gap=%.1fs probe=%.2fs targetWork>=%" PRIu64 " retry=%" PRIu64
           " minDepth=%u attempts=%u nesting=%u\n",
           sched->deep_reroot_enabled?"on":"off",sched->deep_reroot_session_slots,sched->deep_reroot_memory_gb,
           sched->deep_reroot_child_cache_mb,sched->deep_reroot_time_sec,sched->deep_reroot_focus_slices,
           sched->deep_reroot_min_total_exp,sched->deep_reroot_hot_age_sec,sched->deep_reroot_hot_work,
           100.0*sched->deep_reroot_hot_max_change,sched->deep_reroot_hot_min_busy,sched->deep_reroot_hot_region_depth,sched->deep_reroot_hot_gap_sec,sched->deep_reroot_hot_probe_sec,
           sched->deep_reroot_min_work,sched->deep_reroot_retry_work,sched->deep_reroot_min_depth,sched->deep_reroot_max_attempts,
           sched->deep_reroot_max_nesting);
    printf("SP-harvest: %s after=%u persistent outer UNKNOWN slice(s); shallow-first MPN +%u..%u; minWork=%" PRIu64 "; +4=fresh %.1fs then resume iff min(pn,dn)<=%u (budget=%u); +5=1x%.1fs; +6 gate(work>=%" PRIu64 ",stable>=%.0f%%)=1x%.1fs; exact-only promotion\n",
           sched->deep_reroot_harvest_enabled ? "on" : "off",
           sched->deep_reroot_harvest_after_slices,
           sched->deep_reroot_harvest_min_depth,sched->deep_reroot_harvest_max_depth,
           sched->deep_reroot_harvest_min_work,sched->deep_reroot_harvest_time_sec,sched->deep_reroot_harvest_resume_pressure,
           sched->deep_reroot_harvest_attempts,sched->deep_reroot_harvest_time_sec,
           sched->deep_reroot_harvest_deep_min_work,100.0*sched->deep_reroot_harvest_deep_min_stability,
           sched->deep_reroot_harvest_time_sec);

    Context rrctx=*sched;
    memset(&rrctx.tt,0,sizeof(rrctx.tt)); memset(&rrctx.child_cache,0,sizeof(rrctx.child_cache));
    memset(&rrctx.hot_tt,0,sizeof(rrctx.hot_tt)); memset(&rrctx.frontier,0,sizeof(rrctx.frontier));
    memset(&rrctx.deep_frontier,0,sizeof(rrctx.deep_frontier)); memset(&rrctx.stats,0,sizeof(rrctx.stats));
    memset(rrctx.reroot_overlay,0,sizeof(rrctx.reroot_overlay)); memset(rrctx.reroot_session,0,sizeof(rrctx.reroot_session));
    rrctx.parallel_shared=NULL; rrctx.parallel_stop_shared=&sp; rrctx.parallel_exact_shared=&sp;
    rrctx.parallel_promote_path_len=0; rrctx.parallel_worker=false; rrctx.prover=prover; rrctx.generation=1; rrctx.phase_name="SPR";
    rrctx.start_time=sp.start_time; rrctx.deadline=sp.deadline; rrctx.stop=false; rrctx.deep_reroot_unwind=false;
    rrctx.log_fp=NULL; rrctx.frontier_fp=NULL; rrctx.deep_frontier_fp=NULL; rrctx.report_interval=1.0e30; rrctx.next_report=1.0e30;

    Context *wc=(Context*)calloc(sp.threads,sizeof(Context));
    uint8_t *ok=(uint8_t*)calloc(sp.threads,1);
    if(!wc || !ok){ free(wc);free(ok);sp_shared_tt_destroy(&sp.tt); return -1; }
    omp_set_dynamic(0);
#pragma omp parallel num_threads(sp.threads)
    {
        unsigned id=(unsigned)omp_get_thread_num();
        if(sp_worker_context_init(&wc[id],sched,&sp,id,opt)){
            ok[id]=1;
            while(!g_interrupted && atomic_load_explicit(&sp.solved,memory_order_acquire)==0){
                double now=wall_time();
                if(sp.deadline>0.0 && now>=sp.deadline) break;
                SPActiveJob job; bool got=false, do_reroot=false; SPRerootRequest rrjob; memset(&rrjob,0,sizeof(rrjob));
                sp_job_lock(&sp);
                if(atomic_load(&sp.solved)==0 && !(sp.deadline>0.0 && wall_time()>=sp.deadline)){
                    if (sched->deep_reroot_enabled && !sp.reroot_pending && !sp.reroot_active)
                        (void)sp_maybe_queue_reroot(sched,&sp);
                    if (sp.reroot_pending && !sp.reroot_active) {
                        rrjob=sp.reroot_request; sp.reroot_pending=false; sp.reroot_active=true; sp.reroot_owner=id; do_reroot=true;
                    } else {
                        got=sp_assign_job(sched,&sp,id,&job);
                        if(got){
                            sp.active[id]=job; sp.active[id].active=true;
                            atomic_fetch_add_explicit(&sp.jobs_started,1,memory_order_relaxed);
                        }else atomic_fetch_add_explicit(&sp.scheduler_waits,1,memory_order_relaxed);
                    }
                    sp_report_locked(sched,&sp,name);
                }
                sp_job_unlock(&sp);

                if (do_reroot) {
                    uint32_t rpn=1,rdn=1; int rrres=sp_run_reroot_episode(&rrctx,&sp,&rrjob,&rpn,&rdn);
                    sp_job_lock(&sp);
                    if (rrres==-2) sched->deep_reroot_enabled=false;
                    if (rrres>=0 && (rpn==0 || rdn==0)) {
                        sp_shared_publish(&sp,&rrjob.target,rpn,rdn);
                        SPActiveJob fake; memset(&fake,0,sizeof(fake)); fake.active=true; fake.path_len=rrjob.path_len;
                        memcpy(fake.path,rrjob.path,sizeof(Board)*rrjob.path_len);
                        sp_recompute_true_path(sched,&sp,&fake);
                        uint32_t xpn=1,xdn=1; uint64_t xw=0;
                        if (sp_shared_read(&sp,&sp.root,&xpn,&xdn,&xw)) {
                            if (xpn==0) atomic_store_explicit(&sp.solved,1,memory_order_release);
                            else if (xdn==0) atomic_store_explicit(&sp.solved,2,memory_order_release);
                        }
                    }
                    sp.reroot_active=false; sp.reroot_owner=0; sp.reroot_request.valid=false;
                    sp_report_locked(sched,&sp,name); sp_job_unlock(&sp);
                    continue;
                }
                if(!got){ sp_thread_yield(); continue; }

                uint64_t c0=wc[id].stats.dfpn_calls, e0=wc[id].stats.expansions;
                wc[id].stop=false;
                wc[id].parallel_job_start_calls=wc[id].stats.dfpn_calls;
                wc[id].parallel_job_budget=sp.max_work_per_job;
                wc[id].parallel_next_publish_calls=wc[id].stats.dfpn_calls + sp.publish_work;
                const Board *cand=&job.path[job.path_len-1];
                (void)sp_worker_import_shared(&wc[id],cand);
                dfpn(&wc[id],cand,job.thpn,job.thdn,1);
                wc[id].parallel_job_budget=0;
                uint64_t dc=wc[id].stats.dfpn_calls-c0, de=wc[id].stats.expansions-e0;
                uint32_t cpn=1,cdn=1; tt_pair(&wc[id],cand,&cpn,&cdn);
                sp_shared_publish(&sp,cand,cpn,cdn);
                atomic_fetch_add_explicit(&sp.total_worker_calls,dc,memory_order_relaxed);
                atomic_fetch_add_explicit(&sp.total_worker_expansions,de,memory_order_relaxed);

                sp_job_lock(&sp);
                sp_finish_job(sched,&sp,&sp.active[id],dc?dc:1);
                atomic_fetch_add_explicit(&sp.jobs_finished,1,memory_order_relaxed);
                sp_report_locked(sched,&sp,name);
                sp_job_unlock(&sp);
            }
            sp_worker_context_destroy(&wc[id]);
        }
    }

    uint32_t rpn=1,rdn=1; uint8_t bm=0;
    sp_job_lock(&sp);
    if (sp.exact_db && exact_db_lookup_pair(sp.exact_db,&sp.root,sp.prover,&rpn,&rdn)) {
        atomic_fetch_add_explicit(&sp.exact_db_hits,1,memory_order_relaxed);
        bm=0;
    } else {
        bm=sp_root_best_move(sched,&sp,&rpn,&rdn);
    }
    sp_job_unlock(&sp);
    int solved=atomic_load(&sp.solved);
    if(!solved){ if(rpn==0)solved=1; else if(rdn==0)solved=2; }
    int verdict=solved==1?1:solved==2?0:-1;
    if(summary){
        memset(summary,0,sizeof(*summary)); summary->verdict=verdict;
        summary->elapsed=wall_time()-sp.start_time; summary->root_pn=rpn; summary->root_dn=rdn; summary->root_best_move=bm;
        for(unsigned i=0;i<sp.threads;++i) if(ok[i]) sp_summary_add_stats(&summary->stats,&wc[i].stats);
    }
    printf("[SP %s FINAL] result=%s root=(%u,%u) jobs=%" PRIu64 "/%" PRIu64
           " exp=%" PRIu64 " shared-hit=%.1f%% sync=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64 " conc=%" PRIu64
           " xdbHit=%" PRIu64 " xdbConf=%" PRIu64 "\n",name,
           verdict==1?"PROVED":verdict==0?"DISPROVED":"UNKNOWN",rpn,rdn,
           atomic_load(&sp.jobs_finished),atomic_load(&sp.jobs_started),
           atomic_load(&sp.total_worker_expansions),
           atomic_load(&sp.shared_probes)?100.0*(double)atomic_load(&sp.shared_hits)/(double)atomic_load(&sp.shared_probes):0.0,
           atomic_load(&sp.periodic_sync_events),atomic_load(&sp.periodic_sync_nodes),
           atomic_load(&sp.periodic_sync_read_hits),atomic_load(&sp.periodic_sync_updates),
           atomic_load(&sp.concentrated_assignments),atomic_load(&sp.exact_db_hits),atomic_load(&sp.exact_db_conflicts));
    printf("[SP-reroot FINAL] triggers=%" PRIu64 " episodes=%" PRIu64 " exact=%" PRIu64 " unknown=%" PRIu64
           " slices=%" PRIu64 " fresh/resume=%" PRIu64 "/%" PRIu64 " calls=%" PRIu64 " exp=%" PRIu64 " sec=%.1f"
           " harvest=%" PRIu64 "/%" PRIu64 "/%" PRIu64 "/%" PRIu64
           " hFresh/resume=%" PRIu64 "/%" PRIu64 " hExp=%" PRIu64 " hSec=%.1f"
           " hFocusEx=%" PRIu64 " hBudgetEx=%" PRIu64 " hGateSkip=%" PRIu64 " hPressureSkip=%" PRIu64
           " promote=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " conflicts=%" PRIu64 "\n",
           atomic_load(&sp.reroot_triggers),atomic_load(&sp.reroot_episodes),atomic_load(&sp.reroot_exact),
           atomic_load(&sp.reroot_unknown),atomic_load(&sp.reroot_slices),atomic_load(&sp.reroot_fresh_slices),
           atomic_load(&sp.reroot_resume_slices),atomic_load(&sp.reroot_sub_calls),atomic_load(&sp.reroot_sub_expansions),
           (double)atomic_load(&sp.reroot_sub_ms)/1000.0,
           atomic_load(&sp.reroot_harvest_triggers),atomic_load(&sp.reroot_harvest_attempts),
           atomic_load(&sp.reroot_harvest_exact),atomic_load(&sp.reroot_harvest_unknown),
           atomic_load(&sp.reroot_harvest_fresh),atomic_load(&sp.reroot_harvest_resume),
           atomic_load(&sp.reroot_harvest_sub_expansions),(double)atomic_load(&sp.reroot_harvest_sub_ms)/1000.0,
           atomic_load(&sp.reroot_harvest_focus_exhausted),atomic_load(&sp.reroot_harvest_budget_exhausted),
           atomic_load(&sp.reroot_harvest_deep_gate_skips),atomic_load(&sp.reroot_harvest_pressure_skips),
           atomic_load(&sp.reroot_promote_found),atomic_load(&sp.reroot_promote_inserted),
           atomic_load(&sp.reroot_promote_ancestor_changed),atomic_load(&sp.reroot_promote_conflicts));
    deep_reroot_sessions_destroy(&rrctx);
    sched->parallel_shared=NULL;
    free(ok); free(wc); sp_shared_tt_destroy(&sp.tt);
    return verdict;
#endif
}

static int brute_value(Board *b)
{
    if (board_full(b)) return 0;
    bool saw_draw = false;
    bool any = false;
    for (int col = 0; col < COLS; ++col) {
        for (int tile = 1; tile <= 3; ++tile) {
            if (!legal_tile_at(b, col, tile)) continue;
            any = true;
            if (winning_move_class(b, b->turn, col, tile) != WIN_NONE) return 1;
            Board child = *b;
            board_drop(&child, col, tile);
            int v = -brute_value(&child);
            if (v > 0) return 1;
            if (v == 0) saw_draw = true;
        }
    }
    if (!any || saw_draw) return 0;
    return -1;
}

static void make_near_full_twos(Board *b, uint8_t floor_row, uint8_t empty_mask)
{
    board_init(b, floor_row);
    b->p[0] = b->p[1] = b->twos = b->threes = 0;
    int occupied = 0;
    for (int c = 0; c < COLS; ++c) {
        int top = (empty_mask & (1u << c)) ? 6 : 7;
        b->height[c] = (uint8_t)top;
        for (int r = floor_row; r < top; ++r) {
            uint64_t bit = bit_at(c, r);
            int owner = (c + r) & 1;
            b->p[owner] |= bit;
            b->twos |= bit;
            occupied++;
        }
    }
    b->ply = (uint8_t)occupied;
    b->turn = (uint8_t)(occupied & 1);
}

static int dfpn_value_for_test(Context *ctx, const Board *root)
{
    ctx->stop = false;
    reset_phase(ctx, root->turn, "TEST-W");
    dfpn(ctx, root, PN_INF, PN_INF, 0);
    uint32_t pn, dn;
    tt_pair(ctx, root, &pn, &dn);
    if (pn == 0) return 1;
    if (dn != 0) return 99;

    reset_phase(ctx, root->turn ^ 1u, "TEST-L");
    dfpn(ctx, root, PN_INF, PN_INF, 0);
    tt_pair(ctx, root, &pn, &dn);
    if (pn == 0) return -1;
    if (dn == 0) return 0;
    return 99;
}

static uint64_t rng_state = UINT64_C(0x54d3e7a9b18c2f61);
static uint64_t xrng(void)
{
    uint64_t x = rng_state;
    x ^= x << 13; x ^= x >> 7; x ^= x << 17;
    rng_state = x;
    return x;
}

static int self_test(void)
{
    puts("Running self-test...");
    uint64_t candidates = 0, double_cases = 0, forced_cases = 0;

    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        for (int sample = 0; sample < 2500; ++sample) {
            Board b;
            board_init(&b, floor_row);
            int target_plies = (int)(xrng() % 25);
            for (int p = 0; p < target_plies; ++p) {
                bool moved = false;
                for (int tries = 0; tries < 30; ++tries) {
                    int col = (int)(xrng() % 7);
                    int tile = 1 + (int)(xrng() % 3);
                    if (!legal_tile_at(&b, col, tile)) continue;
                    if (winning_move_class(&b, b.turn, col, tile) != WIN_NONE) continue;
                    Board c = b;
                    board_drop(&c, col, tile);
                    if (has_win_for_player(&c, b.turn)) continue;
                    b = c; moved = true; break;
                }
                if (!moved) break;
            }

            /* Fast hypothetical win detection versus independent board scan. */
            for (int col = 0; col < 7; ++col) {
                for (int tile = 1; tile <= 3; ++tile) {
                    if (!legal_tile_at(&b, col, tile)) continue;
                    int mover = b.turn;
                    bool fast = winning_move_class(&b, mover, col, tile) != WIN_NONE;
                    Board c = b;
                    board_drop(&c, col, tile);
                    bool brute = has_win_for_player(&c, mover);
                    candidates++;
                    if (fast != brute) {
                        fprintf(stderr, "SELF-TEST FAIL: win detector rows=%d sample=%d move=%d%c\n",
                                rows, sample, tile, 'A' + col);
                        return 1;
                    }
                }
            }

            ThreatInfo own = immediate_info(&b, b.turn);
            ThreatInfo opp = immediate_info(&b, b.turn ^ 1u);
            unsigned no = popcount7(opp.colmask);
            if (!own.colmask && no >= 2) {
                double_cases++;
                for (int col = 0; col < 7; ++col) {
                    for (int tile = 1; tile <= 3; ++tile) {
                        if (!legal_tile_at(&b, col, tile)) continue;
                        Board c = b;
                        board_drop(&c, col, tile);
                        if (!immediate_info(&c, c.turn).colmask) {
                            fprintf(stderr, "SELF-TEST FAIL: double-threat pruning unsound\n");
                            return 1;
                        }
                    }
                }
            } else if (!own.colmask && no == 1) {
                forced_cases++;
                int forced = 0;
                while (!(opp.colmask & (1u << forced))) ++forced;
                for (int col = 0; col < 7; ++col) {
                    if (col == forced) continue;
                    for (int tile = 1; tile <= 3; ++tile) {
                        if (!legal_tile_at(&b, col, tile)) continue;
                        Board c = b;
                        board_drop(&c, col, tile);
                        if (!immediate_info(&c, c.turn).colmask) {
                            fprintf(stderr, "SELF-TEST FAIL: forced-defense pruning unsound\n");
                            return 1;
                        }
                    }
                }
            }

            uint64_t k1[4], k2[4]; bool m1, m2;
            canonical_key(&b, k1, &m1);
            Board mb = b;
            mb.p[0] = mirror_bits(b.p[0]); mb.p[1] = mirror_bits(b.p[1]);
            mb.twos = mirror_bits(b.twos); mb.threes = mirror_bits(b.threes);
            for (int c = 0; c < 7; ++c) mb.height[c] = b.height[6 - c];
            canonical_key(&mb, k2, &m2);
            if (memcmp(k1, k2, sizeof(k1)) != 0) {
                fprintf(stderr, "SELF-TEST FAIL: symmetry canonicalization\n");
                return 1;
            }
        }
    }

    /* Validate the DF-PN thresholds against exhaustive minimax on near-full states. */
    Context tc;
    memset(&tc, 0, sizeof(tc));
    tc.tss_enabled = false;
    tc.report_interval = 1e30;
    tc.deadline = 0.0;
    tc.phase_name = "TEST";
    if (!tt_init(&tc.tt, 0.05)) {
        fprintf(stderr, "SELF-TEST FAIL: could not allocate test TT\n");
        return 1;
    }
    tc.child_cache_enabled = true;
    if (!child_cache_init(&tc.child_cache, 8.0)) {
        fprintf(stderr, "SELF-TEST FAIL: could not allocate test child cache\n");
        tt_destroy(&tc.tt);
        return 1;
    }
    const uint8_t masks[] = {0x01, 0x02, 0x04, 0x03, 0x05, 0x09, 0x0b, 0x15};
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b;
            make_near_full_twos(&b, floor_row, masks[i]);
            if (has_win_for_player(&b, 0) || has_win_for_player(&b, 1)) {
                fprintf(stderr, "SELF-TEST FAIL: constructed state unexpectedly won\n");
                tt_destroy(&tc.tt);
                return 1;
            }
            Board copy = b;
            int brute = brute_value(&copy);
            int proof = dfpn_value_for_test(&tc, &b);
            if (proof != brute) {
                fprintf(stderr, "SELF-TEST FAIL: DF-PN=%d brute=%d rows=%d mask=0x%x\n",
                        proof, brute, rows, masks[i]);
                tt_destroy(&tc.tt);
                return 1;
            }
        }
    }
    /* Repeat the same exact minimax comparison with adaptive TSS enabled. */
    tc.tss_enabled = true;
    tc.tss_depth = 16;
    tc.tss_budget = 5000;
    tc.tss_start_ply = 0;
    tc.tss_levels = 3;
    tc.tss_depth_step = 8;
    tc.tss_budget_mult = 4;
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b;
            make_near_full_twos(&b, floor_row, masks[i]);
            Board copy = b;
            int brute = brute_value(&copy);
            int proof = dfpn_value_for_test(&tc, &b);
            if (proof != brute) {
                fprintf(stderr, "SELF-TEST FAIL: adaptive TSS DF-PN=%d brute=%d rows=%d mask=0x%x\n",
                        proof, brute, rows, masks[i]);
                tt_destroy(&tc.tt);
                return 1;
            }
        }
    }

    /* Repeat once more with the v3 bounded tactical pre-solver and bridge enabled. */
    tc.tss_compress = true;
    tc.tactical_enabled = true;
    tc.tactical_start_ply = 0;
    tc.tactical_depth = 4;
    tc.tactical_candidates = 6;
    tc.tactical_budget = 800;
    tc.bridge_enabled = true;
    tc.bridge_start_ply = 0;
    tc.bridge_layers = 1;
    tc.bridge_candidates = 4;
    tc.bridge_levels = 2;
    tc.bridge_budget = 12000;
    tc.bridge_budget_mult = 4;
    tc.dependency_enabled = true;
    tc.dep_full_enabled = true;
    tc.dep_oracle_enabled = true;
    tc.dep_start_ply = 0;
    tc.dep_full_start_ply = 0;
    tc.dep_full_tss_level = 0;
    tc.dep_dead_empty = 49;
    tc.dep_deadcheck_ply = 0;
    tc.dep_oracle_start_ply = 0;
    tc.dep_oracle_layers = 2;
    tc.dep_oracle_candidates = 4;
    tc.dep_oracle_budget = 3000;
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b;
            make_near_full_twos(&b, floor_row, masks[i]);
            Board copy = b;
            int brute = brute_value(&copy);
            int proof = dfpn_value_for_test(&tc, &b);
            if (proof != brute) {
                fprintf(stderr, "SELF-TEST FAIL: v5 selective dependency DF-PN=%d brute=%d rows=%d mask=0x%x\n",
                        proof, brute, rows, masks[i]);
                tt_destroy(&tc.tt);
                return 1;
            }
        }
    }
    /* v7 OR-only dependency ordering must remain exact as well. */
    tc.dep_oracle_enabled = false;
    tc.dep_full_enabled = false;
    tc.tactical_enabled = false;
    tc.bridge_enabled = false;
    tc.legacy_quiet_order = false;
    tc.or_dependency_only = true;
    tc.frontier_enabled = true; /* v8 diagnostics must be search-neutral */
    tc.frontier_top = 5;
    tc.dep_start_ply = 0;
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b;
            make_near_full_twos(&b, floor_row, masks[i]);
            Board copy = b;
            int brute = brute_value(&copy);
            int proof = dfpn_value_for_test(&tc, &b);
            if (proof != brute) {
                fprintf(stderr, "SELF-TEST FAIL: v7 OR-only/cache DF-PN=%d brute=%d rows=%d mask=0x%x\n",
                        proof, brute, rows, masks[i]);
                return 1;
            }
        }
    }

    /* v9.8.2 regression: tiny work quanta must preserve the exact minimax
       verdict.  The root remains unlimited while non-root calls repeatedly
       yield after a few fresh expansions. */
    tc.work_quantum = 4;
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows);
        tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b;
            make_near_full_twos(&b, floor_row, masks[i]);
            Board copy = b;
            int brute = brute_value(&copy);
            int proof = dfpn_value_for_test(&tc, &b);
            if (proof != brute) {
                fprintf(stderr, "SELF-TEST FAIL: v9.8.2 work-quantum DF-PN=%d brute=%d rows=%d mask=0x%x\n",
                        proof, brute, rows, masks[i]);
                child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
            }
        }
    }
    tc.work_quantum = 0;

    /* v9.8.2 regression: reroot eligibility must use the deep NODE's PN/DN
       stall, not the age of one recursive frame.  Simulate a depth-6 node
       with 300k committed expansions since its last pair change, then mark a
       just-entered active frame (age ~= 0).  Candidate selection must still
       recognize the node as stalled. */
    {
        Context rc;
        memset(&rc, 0, sizeof(rc));
        Board rb;
        make_near_full_twos(&rb, 0, 0x01);
        rc.floor_row = 0;
        rc.frontier_enabled = true;
        rc.frontier.initialized = true;
        rc.frontier.expansions_at_root_change = 0;
        rc.deep_frontier_enabled = true;
        rc.deep_frontier_depth = 6;
        rc.deep_frontier_top = 1;
        rc.deep_frontier_max_nodes = 16;
        rc.deep_reroot_enabled = true;
        rc.deep_reroot_min_total_exp = 500000;
        rc.deep_reroot_stall_exp = 250000;
        rc.deep_reroot_min_work = 250000;
        rc.deep_reroot_retry_work = 1000000;
        rc.deep_reroot_min_depth = 6;
        rc.deep_reroot_max_attempts = 3;
        rc.stats.expansions = 600000;
        if (!tt_init(&rc.tt, 0.01) || !deep_frontier_init(&rc, &rb)) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 node-stall fixture allocation\n");
            deep_frontier_destroy(&rc);
            tt_destroy(&rc.tt);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        DeepFrontierNode *rn = &rc.deep_frontier.node[1];
        memset(rn, 0, sizeof(*rn));
        rn->valid = true;
        rn->depth = 6;
        rn->move = move_encode(1, 3);
        rn->board = rb;
        rn->parent = 0;
        rn->first_child = -1;
        rn->next_sibling = -1;
        rn->pair_valid = 1;
        rn->last_pn = 17;
        rn->last_dn = 9;
        rn->work_expansions = 300000;
        rn->work_at_pair_change = 0;
        rc.deep_frontier.node[0].first_child = 1;
        rc.deep_frontier.count = 2;
        rc.deep_active_node[6] = 1;
        rc.deep_active_start[6] = work_stamp(&rc); /* fresh re-entry: frame age zero */

        double rs = 0.0, ls = 0.0;
        uint64_t lw = 0, ns = 0;
        int32_t cand = deep_reroot_candidate(&rc, &rs, &ls, &lw, &ns);
        if (cand != 1 || ns != 300000) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 node-stall trigger cand=%d stall=%" PRIu64 "\n",
                    (int)cand, ns);
            deep_frontier_destroy(&rc);
            tt_destroy(&rc.tt);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        deep_frontier_destroy(&rc);
        tt_destroy(&rc.tt);
    }

    /* v9.8.2 regression: reacquiring the same reroot board must return the
       same live sub-context with the same TT generation and retained entries. */
    {
        Context pc;
        memset(&pc, 0, sizeof(pc));
        Board pb;
        make_near_full_twos(&pb, 0, 0x03);
        pc.floor_row = 0;
        pc.prover = pb.turn;
        pc.deep_reroot_memory_gb = 0.05;
        pc.deep_reroot_child_cache_mb = 0.0;
        pc.deep_reroot_max_nesting = 1;
        pc.deep_reroot_focus_slices = 4;
        pc.deep_reroot_level = 0;
        pc.report_interval = 1e30;
        bool resumed1 = true, resumed2 = false;
        RerootSession *a = deep_reroot_session_acquire(&pc, &pb, NULL, 0, &resumed1);
        if (!a || resumed1 || !a->sub_context) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 persistent reroot fresh acquire\n");
            deep_reroot_sessions_destroy(&pc);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        Context *sub1 = (Context *)a->sub_context;
        uint32_t gen = sub1->generation;
        bool hit = false, mir = false;
        TTEntry *te = tt_access(sub1, &pb, true, &hit, &mir);
        if (!te) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 persistent reroot TT create\n");
            deep_reroot_sessions_destroy(&pc);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        te->pn = 123; te->dn = 456;
        RerootSession *b = deep_reroot_session_acquire(&pc, &pb, NULL, 0, &resumed2);
        uint32_t pn = 0, dn = 0;
        Context *sub2 = b ? (Context *)b->sub_context : NULL;
        if (!b || !resumed2 || sub2 != sub1 || sub2->generation != gen ||
            !tt_pair_peek(sub2, &pb, &pn, &dn) || pn != 123 || dn != 456 ||
            sub2->deep_reroot_focus_slices != 4 ||
            pc.stats.deep_reroot_fresh_sessions != 1 || pc.stats.deep_reroot_resumes != 1) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 persistent reroot resume pn/dn=(%u,%u)\n", pn, dn);
            deep_reroot_sessions_destroy(&pc);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        deep_reroot_sessions_destroy(&pc);
    }

    /* v9.8.2 regression: an exact descendant promoted from a reroot
       context must become immediately visible in the global shared TT and
       force a true ancestor reaggregation.  The fixture computes the child's
       exact prover proposition by exhaustive near-end minimax (or by an
       immediate winning move), so no heuristic value is injected. */
    {
        Board pr = {0}, pc = {0}; bool found = false; uint32_t exact_pn=1, exact_dn=1;
        const uint8_t pmasks[] = {0x01,0x02,0x04,0x08,0x10,0x20,0x40,0x03,0x05,0x09,0x11,0x21};
        for (size_t mi = 0; mi < sizeof(pmasks)/sizeof(pmasks[0]) && !found; ++mi) {
            make_near_full_twos(&pr, 0, pmasks[mi]);
            if (has_win_for_player(&pr,0) || has_win_for_player(&pr,1)) continue;
            for (int col=0; col<COLS && !found; ++col) for (int tile=1; tile<=3 && !found; ++tile) {
                if (!legal_tile_at(&pr,col,tile)) continue;
                int wc = winning_move_class(&pr,pr.turn,col,tile);
                pc = pr; board_drop(&pc,col,tile);
                bool prover_wins = false;
                if (wc != WIN_NONE) prover_wins = true;
                else {
                    Board cp = pc; int rel = brute_value(&cp);
                    /* pc.turn is the opponent of pr.turn. */
                    prover_wins = rel < 0;
                }
                exact_pn = prover_wins ? 0 : PN_INF;
                exact_dn = prover_wins ? PN_INF : 0;
                found = true;
            }
        }
        if (!found) {
            fprintf(stderr,"SELF-TEST FAIL: v9.8.2 could not construct promotion fixture\n");
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        ParallelShared ps; memset(&ps,0,sizeof(ps)); atomic_flag_clear(&ps.job_lock);
        atomic_init(&ps.solved,0); atomic_init(&ps.shared_probes,0); atomic_init(&ps.shared_hits,0);
        atomic_init(&ps.shared_publishes,0); atomic_init(&ps.reroot_promote_found,0);
        atomic_init(&ps.reroot_promote_inserted,0); atomic_init(&ps.reroot_promote_ancestor_changed,0);
        atomic_init(&ps.reroot_promote_conflicts,0);
        ps.root=pr; ps.prover=pr.turn; ps.threads=1;
        if (!sp_shared_tt_init(&ps.tt,0.01)) {
            fprintf(stderr,"SELF-TEST FAIL: v9.8.2 promotion shared TT allocation\n");
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        Context px; memset(&px,0,sizeof(px)); px.floor_row=0; px.prover=pr.turn;
        px.report_interval=1e30; px.parallel_exact_shared=&ps;
        if (!tt_init(&px.tt,0.01)) {
            fprintf(stderr,"SELF-TEST FAIL: v9.8.2 promotion scheduler TT allocation\n");
            sp_shared_tt_destroy(&ps.tt); child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        reset_phase(&px,pr.turn,"PROMOTE-TEST");
        sp_shared_publish(&ps,&pr,123,456); /* force an observable ancestor update */
        Board ppath[2]={pr,pc};
        sp_promote_exact_descendant(&px,&pc,exact_pn,exact_dn,ppath,2);
        uint32_t ppn=1,pdn=1,cpn=1,cdn=1; uint64_t pw=0,cw=0;
        bool pok=sp_shared_read(&ps,&pr,&ppn,&pdn,&pw);
        bool cok=sp_shared_read(&ps,&pc,&cpn,&cdn,&cw);
        if (!pok || !cok || cpn!=exact_pn || cdn!=exact_dn || (ppn==123 && pdn==456) ||
            atomic_load(&ps.reroot_promote_found)!=1 || atomic_load(&ps.reroot_promote_inserted)!=1 ||
            atomic_load(&ps.reroot_promote_ancestor_changed)<1 || atomic_load(&ps.reroot_promote_conflicts)!=0) {
            fprintf(stderr,"SELF-TEST FAIL: v9.8.2 promotion child=(%u,%u) expected=(%u,%u) root=(%u,%u) promote=%" PRIu64 "/%" PRIu64 "/%" PRIu64 " conf=%" PRIu64 "\n",
                    cpn,cdn,exact_pn,exact_dn,ppn,pdn,
                    atomic_load(&ps.reroot_promote_found),atomic_load(&ps.reroot_promote_inserted),
                    atomic_load(&ps.reroot_promote_ancestor_changed),atomic_load(&ps.reroot_promote_conflicts));
            tt_destroy(&px.tt); sp_shared_tt_destroy(&ps.tt);
            child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
        tt_destroy(&px.tt); sp_shared_tt_destroy(&ps.tt);
    }

    if (!hot_tt_init(&tc.hot_tt, 8.0)) {
        fprintf(stderr, "SELF-TEST FAIL: could not allocate hot exact TT\n");
        child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
    }
    for (int rows = 5; rows <= 7; ++rows) {
        uint8_t floor_row = (uint8_t)(7 - rows); tc.floor_row = floor_row;
        for (size_t i = 0; i < sizeof(masks)/sizeof(masks[0]); ++i) {
            Board b; make_near_full_twos(&b, floor_row, masks[i]);
            Board cp = b; int rel = brute_value(&cp);
            int expected = rel > 0 ? b.turn : (rel < 0 ? (b.turn ^ 1u) : 2);
            uint64_t budget = UINT64_C(2000000);
            int got = hot_exact_dfs(&tc, &b, 12, &budget);
            if (got != expected) {
                fprintf(stderr, "SELF-TEST FAIL: hot exact=%d expected=%d rows=%d mask=0x%x left=%" PRIu64 "\n",
                        got, expected, rows, masks[i], budget);
                hot_tt_destroy(&tc.hot_tt); child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
            }
        }
    }
    /* v9.8.2 regression: the periodic global hot scheduler must see in-flight
       work that has not yet been committed by a returning recursive descent. */
    {
        Board sb;
        make_near_full_twos(&sb, 0, 0x01);
        memset(&tc.frontier, 0, sizeof(tc.frontier));
        tc.frontier_enabled = true;
        tc.frontier.initialized = true;
        tc.frontier.child_count = 1;
        tc.frontier.child[0].valid = true;
        tc.frontier.child[0].move = move_encode(1, 3);
        tc.frontier.child[0].board = sb;
        tc.frontier.child[0].work_expansions = 10000;
        tc.frontier.child[0].reply_count = 1;
        tc.frontier.child[0].reply[0].valid = true;
        tc.frontier.child[0].reply[0].move = move_encode(1, 1);
        tc.frontier.child[0].reply[0].board = sb;
        tc.frontier.child[0].reply[0].work_expansions = 10000;
        tc.frontier.expansions_at_root_change = 0;
        tc.active_frontier_ci = 0;
        tc.active_frontier_ri = 0;
        tc.active_child_start_exp = 100000;
        tc.active_reply_start_exp = 120000;
        tc.stats.expansions = 1000000;
        tc.hot_exact_enabled = true;
        tc.hot_exact_min_total_exp = 1000;
        tc.hot_exact_stall_exp = 1000;
        tc.hot_exact_min_reply_exp = 1000;
        tc.hot_exact_child_share = 0.70;
        tc.hot_exact_reply_share = 0.70;
        tc.hot_exact_levels = 1;
        tc.hot_exact_depth = 12;
        tc.hot_exact_depth_step = 4;
        tc.hot_exact_budget = 2000000;
        tc.hot_exact_budget_mult = 4;
        tc.hot_exact_retry_work = 1000;
        tc.prover = sb.turn;
        uint64_t a0 = tc.stats.hot_exact_attempts;
        (void)maybe_schedule_hot_exact(&tc);
        if (tc.stats.hot_exact_attempts != a0 + 1) {
            fprintf(stderr, "SELF-TEST FAIL: v9.8.2 live-work hot scheduler did not fire\n");
            hot_tt_destroy(&tc.hot_tt); child_cache_destroy(&tc.child_cache); tt_destroy(&tc.tt); return 1;
        }
    }

    hot_tt_destroy(&tc.hot_tt);
    child_cache_destroy(&tc.child_cache);
    tt_destroy(&tc.tt);

    printf("Self-test OK: %" PRIu64 " candidate moves checked; %" PRIu64
           " double-threat and %" PRIu64 " forced-defense positions verified; "
           "24 near-end DF-PN positions matched exhaustive minimax in baseline, adaptive-TSS, "
           "full dependency/cache, OR-only dependency/cache + frontier, v9.8.2 work-quantum, v9.8.2 parallel hot-region + persistent focused/recursive reroot + exact-descendant promotion, v9 hot-exact modes, and the v9.8.2 live-work scheduler regression.\n",
           candidates, double_cases, forced_cases);
    return 0;
}

static void usage(const char *argv0)
{
    printf(
        "Make Seven DF-PN v9.8.3 - SPDFPN + Persistent Exact Proposition DB research variant\n\n"
        "Usage: %s [options] [MOVESEQUENCE]\n\n"
        "Options:\n"
        "  --rows N             Play 7xN using original top geometry; N=5,6,7 (default 7)\n"
        "  --position SEQ       Position such as 2D2D1C (naked sequence also accepted)\n"
        "  --exact-db FILE      Read-only exact proposition DB (outcome or GREEN|YELLOW PROVED|DISPROVED facts)\n"
        "  --exact-lemma-depth N Exact-only bottom-up closure depth, 0=off, max 12 (default 0)\n"
        "  --exact-lemma-budget N Node budget for exact lemma closure (default 1000000)\n"
        "  --perspective P      side|green|yellow (default side)\n"
        "  --phase-only P       W|L: solve only that exact phase proposition (default full game)\n"
        "  --label TEXT         Label written to final summary (e.g. 1D)\n"
        "  --memory-gb G        TT memory request in GiB (default 2)\n"
        "  --child-cache-mb M   Exact child-generation cache per process (default 32 MiB)\n"
        "  --threads N          Unified v9.8.2 SPDFPN scheduler workers, including N=1 (default 1)\n"
        "  --shared-tt-gb G     Shared PN/DN/work TT for parallel workers (default 1 GiB)\n"
        "  --worker-tt-mb M     Private rich TT per worker in parallel mode (default 128 MiB)\n"
        "  --worker-cache-mb M  Private child cache per worker (default 8 MiB)\n"
        "  --job-work N         DFPN calls per bounded scheduler job (default 4096)\n"
        "  --min-job-depth N    Never assign jobs shallower than this depth (default 1)\n"
        "  --publish-work N     Hot-path shared sync interval in DFPN calls; 0=off (default 0)\n"
        "  --publish-depth N    Hot-path nodes synced per event, 1..50 (default 8)\n"
        "  --busy-penalty X     Soft busy penalty per already-active worker, >=0 (default .02)\n"
        "  --legacy-serial      Use pre-SPDFPN serial path; requires --threads 1\n"
        "  --no-child-cache     Disable exact child-generation cache for A/B testing\n"
        "  --time-limit SEC     Total wall-clock limit; 0 = unlimited (default 0)\n"
        "  --report SEC         Console/CSV status interval (default 60)\n"
        "  --log FILE           Periodic CSV benchmark file (default dfpn_v9_8_3_benchmark.csv)\n"
        "  --summary FILE       One-row final CSV for orchestration (default none)\n"
        "  --frontier-log FILE  v9 root/depth-2 proof-frontier CSV (default dfpn_v9_8_3_frontier.csv)\n"
        "  --no-frontier        Disable v9 proof-frontier diagnostics\n"
        "  --frontier-top N     Root/depth-2 display/tooling width (default 5)\n"
        "  --deep-frontier-log FILE  v9.8.2 deep prefix profiler CSV (default dfpn_v9_8_3_deep_frontier.csv)\n"
        "  --no-deep-frontier   Disable v9.8.2 deep proof-frontier profiling\n"
        "  --deep-frontier      Enable deep proof-frontier profiling (default on)\n"
        "  --deep-frontier-depth N  Prefix depth to profile, 1..10 (default 6)\n"
        "  --deep-frontier-top N  Top siblings logged at each hot prefix (default 5)\n"
        "  --deep-frontier-max-nodes N  Prefix-node cap (default 1000000)\n"
        "  --no-hot-exact       Disable legacy bounded hot-subtree exact DFS (default off)\n"
        "  --hot-exact          Enable legacy bounded hot-subtree exact DFS\n"
        "  --hot-exact-tt-mb M  Independent exact-result TT (default 64 MiB)\n"
        "  --hot-exact-depth N  First exact DFS depth limit (default 12)\n"
        "  --hot-exact-depth-step N Added plies per retry (default 4)\n"
        "  --hot-exact-levels N Maximum hot exact attempts per reply (default 3)\n"
        "  --hot-exact-budget N First hot DFS node budget (default 500000)\n"
        "  --hot-exact-budget-mult N Retry budget multiplier (default 4)\n"
        "  --hot-min-total-exp N Require total DF-PN expansions (default 500000)\n"
        "  --hot-stall-exp N    Require unchanged-root expansions (default 250000)\n"
        "  --hot-min-reply-exp N Require hot reply work (default 250000)\n"
        "  --hot-retry-work N   New reply work before retry (default 1000000)\n"
        "  --hot-child-share X  Root-child work concentration 0..1 (default .70)\n"
        "  --hot-reply-share X  Depth-2 reply concentration 0..1 (default .70)\n"
        "  --no-hot-tss-gate    Disable low-yield TSS retry throttling\n"
        "  --hot-tss-level-cap N Max TSS levels in gated descendants (default 1)\n"
        "  --hot-tss-min-calls N Evidence before gating (default 100000)\n"
        "  --hot-tss-max-rate X Max TSS proof rate as fraction (default .002)\n"
        "  --work-quantum N     Max fresh DF-PN expansions per non-root call (default 0=unlimited)\n"
        "  --no-work-quantum    Classic unlimited recursive DF-PN for A/B baseline\n"
        "  --deep-reroot         Enable v9.8.2 persistent focused/recursive reroot DF-PN oracle (default on)\n"
        "  --no-deep-reroot      Disable v9.8.2 persistent focused/recursive reroot DF-PN oracle\n"
        "  --reroot-memory-gb G  Per-session subsolver TT request (default 0.25 GiB)\n"
        "  --reroot-cache-mb M   Per-session subsolver child cache (default 16 MiB)\n"
        "  --reroot-time SEC     Wall-time slice per reroot/resume attempt (default 180)\n"
        "  --reroot-min-total-exp N Main-search work before reroot (default 500000; parallel: aggregate worker expansions)\n"
        "  --reroot-stall-exp N  Legacy serial root-stall threshold (default 250000; ignored by parallel hot detector)\n"
        "  --reroot-min-work N   Serial: deep-node stall; parallel: minimum deep target shared work (default 250000)\n"
        "  --reroot-retry-work N Serial: retry work; parallel hot detector: aggregate new work before retry (default 1000000)\n"
        "  --reroot-hot-age SEC Parallel: active-job region hot exposure required (default 20)\n"
        "  --reroot-hot-work N  Parallel: aggregate shared-path DFPN calls during hot window (default 1000000)\n"
        "  --reroot-hot-max-change X Parallel: max region PN/DN drift before stability window resets (default .10)\n"
        "  --reroot-hot-probe SEC Parallel detector sampling interval (default .50)\n"
        "  --reroot-hot-gap SEC  Region absence before exposure resets; 0=disabled (default 5)\n"
        "  --reroot-hot-min-busy N Parallel: minimum active workers sharing a hot region (default 2)\n"
        "  --reroot-hot-region-depth N Parallel: deepest prefix depth used to define a region (default 3)\n"
        "  --reroot-root-share X Legacy diagnostic share only; no trigger gate (default .70)\n"
        "  --reroot-inner-share X Legacy diagnostic share only; no trigger gate (default .95)\n"
        "  --reroot-min-depth N  Minimum extracted prefix depth (default 6)\n"
        "  --reroot-attempts N   Maximum attempts per deep-prefix node (default 3)\n"
        "  --reroot-nesting N    Maximum recursive persistent-reroot levels (default 2)\n"
        "  --reroot-sessions N    Persistent session slots per Context, 1..4 (default 2)\n"
        "  --reroot-profile-nodes N Deep-profiler node cap inside reroot solver (default 600000)\n"
        "  --reroot-focus-slices N Consecutive persistent slices per outer trigger (default 4)\n"
        "  --reroot-harvest       Enable proof-pressure/evidence-gated shallow-first exact-harvest ladder (default on)\n"
        "  --no-reroot-harvest    Disable proof-pressure/evidence-gated exact-harvest ladder\n"
        "  --reroot-harvest-after N Start harvesting after N outer UNKNOWN slices (default 2)\n"
        "  --reroot-harvest-min-depth N Minimum extra MPN-chain plies (default 4)\n"
        "  --reroot-harvest-max-depth N Maximum extra MPN-chain plies (default 6)\n"
        "  --reroot-harvest-attempts N Shallow-rung slice budget (default 2; deeper rungs use 1)\n"
        "  --reroot-harvest-focus-slices N Consecutive shallow-rung sticky slices (default 2)\n"
        "  --reroot-harvest-min-work N Minimum profiled work for +4/+5 candidates (default 1000)\n"
        "  --reroot-harvest-deep-min-work N +6 minimum profiled work (default 50000)\n"
        "  --reroot-harvest-deep-min-stability X +6 minimum stall/work fraction (default .50)\n"
        "  --reroot-harvest-resume-pressure N Resume +4 only when min(pn,dn)<=N (default 128)\n"
        "  --reroot-harvest-time SEC Nested harvesting slice time (default 5)\n"
        "  --reroot-tss          Enable TSS inside reroot subsolver (default off)\n"
        "  --no-reroot-tss       Disable TSS inside reroot subsolver\n"
        "  --tss-depth N        Base threat-chain depth (default 16)\n"
        "  --tss-budget N       Base TSS nodes per oracle call (default 5000)\n"
        "  --tss-levels N       Adaptive TSS levels per TT residency (default 3)\n"
        "  --tss-depth-step N   Added depth per retry level (default 8)\n"
        "  --tss-budget-mult N  Budget multiplier per retry level (default 4)\n"
        "  --tss-start-ply N    Do not call TSS before this ply (default 6)\n"
        "  --no-tss             Disable threat-chain proof oracle\n"
        "  --no-tss-compress    Disable selective TSS threat-signature compression\n"
        "  --tactical           Enable bounded exact tactical pre-solver (off by default)\n"
        "  --tactical-start-ply N Start bounded exact tactical scan (default 12)\n"
        "  --tactical-depth N   Exact tactical lookahead plies (default 4)\n"
        "  --tactical-candidates N Scan top N children on first expansion (default 6)\n"
        "  --tactical-budget N  Shared tactical nodes per first expansion (default 800)\n"
        "  --no-tactical        Disable bounded exact tactical child pre-solving\n"
        "  --bridge             Enable experimental quiet-setup bridge oracle (off by default)\n"
        "  --bridge-start-ply N Start quiet-setup bridge oracle at this ply (default 6)\n"
        "  --bridge-layers N    Quiet setup layers at bridge level 0 (default 1)\n"
        "  --bridge-candidates N Top relevance-ranked setup moves tried (default 3)\n"
        "  --bridge-levels N    Adaptive bridge retry levels per TT residency (default 2)\n"
        "  --bridge-budget N    Base bridge/TSS node budget per call (default 12000)\n"
        "  --bridge-budget-mult N Budget multiplier per retry (default 4)\n"
        "  --no-bridge          Explicitly disable quiet-setup bridge proof oracle\n"
        "  --no-dependency      Disable all dependency ordering/dead checks/oracle\n"
        "  --dep-start-ply N    Start cheap local dependency ordering (default 6)\n"
        "  --no-full-dependency Disable cached full-map dependency hints\n"
        "  --full-dependency    Enable cached full-map dependency hints (default on)\n"
        "  --dep-full-start-ply N Earliest ply for full-map hint (default 12)\n"
        "  --dep-full-tss-level N Require this many adaptive TSS attempts first (default 3)\n"
        "  --dep-dead-empty N   Run exact dead-prover only with <=N empty cells (default 8; 0=off)\n"
        "  --dep-deadcheck-ply N Additional minimum ply for dead check (default 0)\n"
        "  --legacy-quiet-order Use center/tile-first quiet ordering (default; v6 first-top3 ~98-99%%)\n"
        "  --dependency-first-quiet Experimental: dependency score ranks all quiet nodes\n"
        "  --or-dependency-first Experimental: dependency-first only at prover/OR nodes\n"
        "  --dep-oracle         Enable dependency quiet-proof oracle (off by default)\n"
        "  --no-dep-oracle      Disable dependency quiet-proof oracle\n"
        "  --dep-oracle-start-ply N Start dependency oracle (default 6)\n"
        "  --dep-oracle-layers N Quiet setup layers (default 2)\n"
        "  --dep-oracle-candidates N Attacker top moves per OR node (default 4)\n"
        "  --dep-oracle-budget N Node budget per dependency-oracle call (default 3000)\n"
        "  --list-legal-moves   Print physical legal moves for --position and exit\n"
        "  --self-test          Run exactness checks and exit\n"
        "  -h, --help           Show this help\n\n"
        "Use run_rootsplit_v9_8_2.py for 1-ply exact process-parallel root splitting; run_openings_v9_8_2.py remains available for opening suites.\n"
        "W/L/D is exact if both required proof phases finish; UNKNOWN means unfinished.\n",
        argv0);
}

static bool parse_u64(const char *s, uint64_t *out)
{
    errno = 0; char *end = NULL;
    unsigned long long v = strtoull(s, &end, 10);
    if (errno || !end || *end) return false;
    *out = (uint64_t)v; return true;
}

static bool parse_double(const char *s, double *out)
{
    errno = 0; char *end = NULL;
    double v = strtod(s, &end);
    if (errno || !end || *end) return false;
    *out = v; return true;
}

static int parse_options(int argc, char **argv, Options *o)
{
    *o = (Options){
        .memory_gb = 2.0,
        .child_cache_mb = CHILD_CACHE_DEFAULT_MB,
        .time_limit = 0.0,
        .report_interval = 60.0,
        .rows = 7,
        .tss_enabled = true,
        .tss_compress = true,
        .tss_depth = 16,
        .tss_budget = 5000,
        .tss_start_ply = 6,
        .tss_levels = 3,
        .tss_depth_step = 8,
        .tss_budget_mult = 4,
        .tactical_enabled = false,
        .tactical_start_ply = 12,
        .tactical_depth = 4,
        .tactical_candidates = 6,
        .tactical_budget = 800,
        .bridge_enabled = false,
        .bridge_start_ply = 6,
        .bridge_layers = 1,
        .bridge_candidates = 3,
        .bridge_levels = 2,
        .bridge_budget = 12000,
        .bridge_budget_mult = 4,
        .dependency_enabled = true,
        .dep_full_enabled = true,
        .dep_oracle_enabled = false,
        .legacy_quiet_order = true,
        .or_dependency_only = false,
        .child_cache_enabled = true,
        .frontier_enabled = true,
        .frontier_top = 5,
        .deep_frontier_enabled = true,
        .deep_frontier_depth = DEEP_FRONTIER_DEFAULT_DEPTH,
        .deep_frontier_top = 5,
        .deep_frontier_max_nodes = DEEP_FRONTIER_DEFAULT_NODES,
        .hot_exact_enabled = false,
        .hot_exact_tt_mb = 64.0,
        .hot_exact_depth = 12,
        .hot_exact_depth_step = 4,
        .hot_exact_levels = 3,
        .hot_exact_budget = 500000,
        .hot_exact_budget_mult = 4,
        .hot_exact_min_total_exp = 500000,
        .hot_exact_stall_exp = 250000,
        .hot_exact_min_reply_exp = 250000,
        .hot_exact_retry_work = 1000000,
        .hot_exact_child_share = 0.70,
        .hot_exact_reply_share = 0.70,
        .hot_tss_gate_enabled = true,
        .hot_tss_min_calls = 100000,
        .hot_tss_max_proof_rate = 0.002,
        .hot_tss_level_cap = 1,
        .work_quantum = 0,
        .exact_lemma_depth = 0,
        .exact_lemma_budget = 1000000,
        .deep_reroot_enabled = true,
        .deep_reroot_memory_gb = REROOT_DEFAULT_TT_GB,
        .deep_reroot_child_cache_mb = REROOT_DEFAULT_CACHE_MB,
        .deep_reroot_time_sec = 180.0,
        .deep_reroot_min_total_exp = 500000,
        .deep_reroot_stall_exp = 250000,
        .deep_reroot_min_work = 250000,
        .deep_reroot_retry_work = 1000000,
        .deep_reroot_hot_age_sec = SP_DEFAULT_HOT_AGE_SEC,
        .deep_reroot_hot_work = SP_DEFAULT_HOT_WORK,
        .deep_reroot_hot_max_change = SP_DEFAULT_HOT_MAX_CHANGE,
        .deep_reroot_hot_probe_sec = SP_DEFAULT_HOT_PROBE_SEC,
        .deep_reroot_hot_gap_sec = SP_DEFAULT_HOT_GAP_SEC,
        .deep_reroot_hot_min_busy = SP_DEFAULT_HOT_MIN_BUSY,
        .deep_reroot_hot_region_depth = SP_DEFAULT_HOT_REGION_DEPTH,
        .deep_reroot_root_share = 0.70,
        .deep_reroot_inner_share = 0.95,
        .deep_reroot_min_depth = 6,
        .deep_reroot_max_attempts = 3,
        .deep_reroot_max_nesting = 2,
        .deep_reroot_session_slots = REROOT_SESSION_DEFAULT_SLOTS,
        .deep_reroot_profile_nodes = 600000,
        .deep_reroot_focus_slices = REROOT_DEFAULT_FOCUS_SLICES,
        .deep_reroot_harvest_enabled = true,
        .deep_reroot_harvest_after_slices = REROOT_HARVEST_AFTER_SLICES,
        .deep_reroot_harvest_min_depth = REROOT_HARVEST_MIN_DEPTH,
        .deep_reroot_harvest_max_depth = REROOT_HARVEST_MAX_DEPTH,
        .deep_reroot_harvest_attempts = REROOT_HARVEST_ATTEMPTS,
        .deep_reroot_harvest_focus_slices = REROOT_HARVEST_FOCUS_SLICES,
        .deep_reroot_harvest_min_work = REROOT_HARVEST_MIN_WORK,
        .deep_reroot_harvest_deep_min_work = REROOT_HARVEST_DEEP_MIN_WORK,
        .deep_reroot_harvest_deep_min_stability = REROOT_HARVEST_DEEP_MIN_STABILITY,
        .deep_reroot_harvest_resume_pressure = REROOT_HARVEST_RESUME_PRESSURE,
        .deep_reroot_harvest_time_sec = REROOT_HARVEST_TIME_SEC,
        .deep_reroot_tss = false,
        .dep_start_ply = 6,
        .dep_full_start_ply = 12,
        .dep_full_tss_level = 3,
        .dep_dead_empty = 8,
        .dep_deadcheck_ply = 0,
        .dep_oracle_start_ply = 6,
        .dep_oracle_layers = 2,
        .dep_oracle_candidates = 4,
        .dep_oracle_budget = 3000,
        .parallel_threads = SP_DEFAULT_THREADS,
        .parallel_shared_tt_gb = SP_DEFAULT_SHARED_TT_GB,
        .parallel_worker_tt_mb = SP_DEFAULT_WORKER_TT_MB,
        .parallel_worker_cache_mb = SP_DEFAULT_WORKER_CACHE_MB,
        .parallel_job_work = SP_DEFAULT_JOB_WORK,
        .parallel_min_job_depth = SP_DEFAULT_MIN_JOB_DEPTH,
        .parallel_publish_work = SP_DEFAULT_PUBLISH_WORK,
        .parallel_publish_depth = SP_DEFAULT_PUBLISH_DEPTH,
        .parallel_busy_penalty = SP_DEFAULT_BUSY_PENALTY,
        .legacy_serial = false,
        .perspective = -1,
        .phase_only = 0,
        .position = "",
        .log_path = "dfpn_v9_8_3_benchmark.csv",
        .frontier_log_path = "dfpn_v9_8_3_frontier.csv",
        .deep_frontier_log_path = "dfpn_v9_8_3_deep_frontier.csv",
        .summary_path = NULL,
        .label = "",
        .self_test = false,
        .list_legal_moves = false
    };
    for (int i = 1; i < argc; ++i) {
        const char *a = argv[i];
        if (!strcmp(a, "-h") || !strcmp(a, "--help")) { usage(argv[0]); return 1; }
        else if (!strcmp(a, "--self-test")) o->self_test = true;
        else if (!strcmp(a, "--list-legal-moves")) o->list_legal_moves = true;
        else if (!strcmp(a, "--legacy-serial")) o->legacy_serial = true;
        else if (!strcmp(a, "--no-tss")) o->tss_enabled = false;
        else if (!strcmp(a, "--no-tss-compress")) o->tss_compress = false;
        else if (!strcmp(a, "--tactical")) o->tactical_enabled = true;
        else if (!strcmp(a, "--no-tactical")) o->tactical_enabled = false;
        else if (!strcmp(a, "--bridge")) o->bridge_enabled = true;
        else if (!strcmp(a, "--no-bridge")) o->bridge_enabled = false;
        else if (!strcmp(a, "--no-dependency")) o->dependency_enabled = false;
        else if (!strcmp(a, "--no-full-dependency")) o->dep_full_enabled = false;
        else if (!strcmp(a, "--full-dependency")) o->dep_full_enabled = true;
        else if (!strcmp(a, "--dep-oracle")) o->dep_oracle_enabled = true;
        else if (!strcmp(a, "--no-dep-oracle")) o->dep_oracle_enabled = false;
        else if (!strcmp(a, "--legacy-quiet-order")) { o->legacy_quiet_order = true; o->or_dependency_only = false; }
        else if (!strcmp(a, "--dependency-first-quiet")) { o->legacy_quiet_order = false; o->or_dependency_only = false; }
        else if (!strcmp(a, "--or-dependency-first")) { o->legacy_quiet_order = false; o->or_dependency_only = true; }
        else if (!strcmp(a, "--no-child-cache")) o->child_cache_enabled = false;
        else if (!strcmp(a, "--child-cache")) o->child_cache_enabled = true;
        else if (!strcmp(a, "--no-frontier")) o->frontier_enabled = false;
        else if (!strcmp(a, "--frontier")) o->frontier_enabled = true;
        else if (!strcmp(a, "--no-deep-frontier")) o->deep_frontier_enabled = false;
        else if (!strcmp(a, "--deep-frontier")) o->deep_frontier_enabled = true;
        else if (!strcmp(a, "--no-hot-exact")) o->hot_exact_enabled = false;
        else if (!strcmp(a, "--hot-exact")) o->hot_exact_enabled = true;
        else if (!strcmp(a, "--no-hot-tss-gate")) o->hot_tss_gate_enabled = false;
        else if (!strcmp(a, "--hot-tss-gate")) o->hot_tss_gate_enabled = true;
        else if (!strcmp(a, "--no-work-quantum")) o->work_quantum = 0;
        else if (!strcmp(a, "--deep-reroot")) o->deep_reroot_enabled = true;
        else if (!strcmp(a, "--no-deep-reroot")) o->deep_reroot_enabled = false;
        else if (!strcmp(a, "--reroot-tss")) o->deep_reroot_tss = true;
        else if (!strcmp(a, "--no-reroot-tss")) o->deep_reroot_tss = false;
        else if (!strcmp(a, "--reroot-harvest")) o->deep_reroot_harvest_enabled = true;
        else if (!strcmp(a, "--no-reroot-harvest")) o->deep_reroot_harvest_enabled = false;
        else if (!strcmp(a, "--work-quantum") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->work_quantum)) return -1;
        } else if (!strcmp(a, "--exact-lemma-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 12) return -1; o->exact_lemma_depth=(unsigned)v;
        } else if (!strcmp(a, "--exact-lemma-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->exact_lemma_budget) || o->exact_lemma_budget < 1) return -1;
        } else if (!strcmp(a, "--reroot-memory-gb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_memory_gb) || o->deep_reroot_memory_gb <= 0.0) return -1;
        } else if (!strcmp(a, "--reroot-cache-mb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_child_cache_mb) || o->deep_reroot_child_cache_mb < 0.0) return -1;
        } else if (!strcmp(a, "--reroot-time") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_time_sec) || o->deep_reroot_time_sec < 0.0) return -1;
        } else if (!strcmp(a, "--reroot-min-total-exp") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_min_total_exp)) return -1;
        } else if (!strcmp(a, "--reroot-stall-exp") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_stall_exp)) return -1;
        } else if (!strcmp(a, "--reroot-min-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_min_work)) return -1;
        } else if (!strcmp(a, "--reroot-retry-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_retry_work)) return -1;
        } else if (!strcmp(a, "--reroot-hot-age") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_hot_age_sec) || o->deep_reroot_hot_age_sec < 0.0) return -1;
        } else if (!strcmp(a, "--reroot-hot-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_hot_work)) return -1;
        } else if (!strcmp(a, "--reroot-hot-max-change") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_hot_max_change) || o->deep_reroot_hot_max_change < 0.0 || o->deep_reroot_hot_max_change > 10.0) return -1;
        } else if (!strcmp(a, "--reroot-hot-probe") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_hot_probe_sec) || o->deep_reroot_hot_probe_sec <= 0.0) return -1;
        } else if (!strcmp(a, "--reroot-hot-gap") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_hot_gap_sec) || o->deep_reroot_hot_gap_sec < 0.0) return -1;
        } else if (!strcmp(a, "--reroot-hot-min-busy") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > SP_MAX_THREADS) return -1; o->deep_reroot_hot_min_busy=(unsigned)v;
        } else if (!strcmp(a, "--reroot-hot-region-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > DEEP_FRONTIER_HARD_MAX_DEPTH) return -1; o->deep_reroot_hot_region_depth=(unsigned)v;
        } else if (!strcmp(a, "--reroot-root-share") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_root_share) || o->deep_reroot_root_share < 0.0 || o->deep_reroot_root_share > 1.0) return -1;
        } else if (!strcmp(a, "--reroot-inner-share") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_inner_share) || o->deep_reroot_inner_share < 0.0 || o->deep_reroot_inner_share > 1.0) return -1;
        } else if (!strcmp(a, "--reroot-min-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > DEEP_FRONTIER_HARD_MAX_DEPTH) return -1; o->deep_reroot_min_depth = (unsigned)v;
        } else if (!strcmp(a, "--reroot-attempts") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 32) return -1; o->deep_reroot_max_attempts = (unsigned)v;
        } else if (!strcmp(a, "--reroot-nesting") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 8) return -1; o->deep_reroot_max_nesting = (unsigned)v;
        } else if (!strcmp(a, "--reroot-sessions") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > REROOT_SESSION_MAX_SLOTS) return -1; o->deep_reroot_session_slots = (unsigned)v;
        } else if (!strcmp(a, "--reroot-profile-nodes") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 100 || v > UINT32_MAX) return -1; o->deep_reroot_profile_nodes = (uint32_t)v;
        } else if (!strcmp(a, "--reroot-focus-slices") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 32) return -1; o->deep_reroot_focus_slices = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-after") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 32) return -1; o->deep_reroot_harvest_after_slices = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-min-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > DEEP_FRONTIER_HARD_MAX_DEPTH) return -1; o->deep_reroot_harvest_min_depth = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-max-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > DEEP_FRONTIER_HARD_MAX_DEPTH) return -1; o->deep_reroot_harvest_max_depth = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-attempts") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 32) return -1; o->deep_reroot_harvest_attempts = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-focus-slices") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 32) return -1; o->deep_reroot_harvest_focus_slices = (unsigned)v;
        } else if (!strcmp(a, "--reroot-harvest-min-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_harvest_min_work)) return -1;
        } else if (!strcmp(a, "--reroot-harvest-deep-min-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->deep_reroot_harvest_deep_min_work)) return -1;
        } else if (!strcmp(a, "--reroot-harvest-deep-min-stability") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_harvest_deep_min_stability) ||
                o->deep_reroot_harvest_deep_min_stability < 0.0 || o->deep_reroot_harvest_deep_min_stability > 1.0) return -1;
        } else if (!strcmp(a, "--reroot-harvest-resume-pressure") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > PN_INF) return -1;
            o->deep_reroot_harvest_resume_pressure = (uint32_t)v;
        } else if (!strcmp(a, "--reroot-harvest-time") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->deep_reroot_harvest_time_sec) || o->deep_reroot_harvest_time_sec < 0.0) return -1;
        }
        else if (!strcmp(a, "--hot-exact-tt-mb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->hot_exact_tt_mb) || o->hot_exact_tt_mb < 0) return -1;
        } else if (!strcmp(a, "--hot-exact-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 49) return -1; o->hot_exact_depth = (unsigned)v;
        } else if (!strcmp(a, "--hot-exact-depth-step") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 49) return -1; o->hot_exact_depth_step = (unsigned)v;
        } else if (!strcmp(a, "--hot-exact-levels") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 16) return -1; o->hot_exact_levels = (unsigned)v;
        } else if (!strcmp(a, "--hot-exact-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_exact_budget)) return -1;
        } else if (!strcmp(a, "--hot-exact-budget-mult") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 100) return -1; o->hot_exact_budget_mult = (unsigned)v;
        } else if (!strcmp(a, "--hot-min-total-exp") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_exact_min_total_exp)) return -1;
        } else if (!strcmp(a, "--hot-stall-exp") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_exact_stall_exp)) return -1;
        } else if (!strcmp(a, "--hot-min-reply-exp") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_exact_min_reply_exp)) return -1;
        } else if (!strcmp(a, "--hot-retry-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_exact_retry_work)) return -1;
        } else if (!strcmp(a, "--hot-child-share") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->hot_exact_child_share) || o->hot_exact_child_share < 0.0 || o->hot_exact_child_share > 1.0) return -1;
        } else if (!strcmp(a, "--hot-reply-share") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->hot_exact_reply_share) || o->hot_exact_reply_share < 0.0 || o->hot_exact_reply_share > 1.0) return -1;
        } else if (!strcmp(a, "--hot-tss-min-calls") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->hot_tss_min_calls)) return -1;
        } else if (!strcmp(a, "--hot-tss-max-rate") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->hot_tss_max_proof_rate) || o->hot_tss_max_proof_rate < 0.0 || o->hot_tss_max_proof_rate > 1.0) return -1;
        } else if (!strcmp(a, "--hot-tss-level-cap") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 16) return -1; o->hot_tss_level_cap = (unsigned)v;
        } else if (!strcmp(a, "--threads") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > SP_MAX_THREADS) return -1; o->parallel_threads = (unsigned)v;
        } else if (!strcmp(a, "--shared-tt-gb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->parallel_shared_tt_gb) || o->parallel_shared_tt_gb <= 0.0) return -1;
        } else if (!strcmp(a, "--worker-tt-mb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->parallel_worker_tt_mb) || o->parallel_worker_tt_mb <= 0.0) return -1;
        } else if (!strcmp(a, "--worker-cache-mb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->parallel_worker_cache_mb) || o->parallel_worker_cache_mb < 0.0) return -1;
        } else if (!strcmp(a, "--job-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->parallel_job_work) || o->parallel_job_work < 16) return -1;
        } else if (!strcmp(a, "--min-job-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > SP_PATH_MAX-1) return -1; o->parallel_min_job_depth = (unsigned)v;
        } else if (!strcmp(a, "--publish-work") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->parallel_publish_work)) return -1;
        } else if (!strcmp(a, "--publish-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v >= SP_PATH_MAX) return -1; o->parallel_publish_depth = (unsigned)v;
        } else if (!strcmp(a, "--busy-penalty") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->parallel_busy_penalty) || o->parallel_busy_penalty < 0.0 || o->parallel_busy_penalty > 16.0) return -1;
        } else if (!strcmp(a, "--rows") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 5 || v > 7) return -1; o->rows = (unsigned)v;
        } else if (!strcmp(a, "--memory-gb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->memory_gb) || o->memory_gb <= 0) return -1;
        } else if (!strcmp(a, "--child-cache-mb") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->child_cache_mb) || o->child_cache_mb < 0) return -1;
        } else if (!strcmp(a, "--time-limit") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->time_limit) || o->time_limit < 0) return -1;
        } else if (!strcmp(a, "--report") && i + 1 < argc) {
            if (!parse_double(argv[++i], &o->report_interval) || o->report_interval <= 0) return -1;
        } else if (!strcmp(a, "--log") && i + 1 < argc) {
            o->log_path = argv[++i];
        } else if (!strcmp(a, "--frontier-log") && i + 1 < argc) {
            o->frontier_log_path = argv[++i];
        } else if (!strcmp(a, "--frontier-top") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > MAX_MOVES) return -1; o->frontier_top = (unsigned)v;
        } else if (!strcmp(a, "--deep-frontier-log") && i + 1 < argc) {
            o->deep_frontier_log_path = argv[++i];
        } else if (!strcmp(a, "--deep-frontier-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > DEEP_FRONTIER_HARD_MAX_DEPTH) return -1; o->deep_frontier_depth = (unsigned)v;
        } else if (!strcmp(a, "--deep-frontier-top") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > MAX_MOVES) return -1; o->deep_frontier_top = (unsigned)v;
        } else if (!strcmp(a, "--deep-frontier-max-nodes") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 100 || v > UINT32_MAX) return -1; o->deep_frontier_max_nodes = (uint32_t)v;
        } else if (!strcmp(a, "--summary") && i + 1 < argc) {
            o->summary_path = argv[++i];
        } else if (!strcmp(a, "--label") && i + 1 < argc) {
            o->label = argv[++i];
        } else if (!strcmp(a, "--position") && i + 1 < argc) {
            o->position = argv[++i];
        } else if (!strcmp(a, "--exact-db") && i + 1 < argc) {
            o->exact_db_path = argv[++i];
        } else if (!strcmp(a, "--perspective") && i + 1 < argc) {
            const char *v = argv[++i];
            if (!strcmp(v, "side")) o->perspective = -1;
            else if (!strcmp(v, "green") || !strcmp(v, "0")) o->perspective = 0;
            else if (!strcmp(v, "yellow") || !strcmp(v, "1")) o->perspective = 1;
            else return -1;
        } else if (!strcmp(a, "--phase-only") && i + 1 < argc) {
            const char *v=argv[++i];
            if (!strcmp(v,"W") || !strcmp(v,"w")) o->phase_only=1;
            else if (!strcmp(v,"L") || !strcmp(v,"l")) o->phase_only=2;
            else return -1;
        } else if (!strcmp(a, "--tss-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->tss_depth = (unsigned)v;
        } else if (!strcmp(a, "--tss-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->tss_budget)) return -1;
        } else if (!strcmp(a, "--tss-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->tss_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--tss-levels") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 16) return -1; o->tss_levels = (unsigned)v;
        } else if (!strcmp(a, "--tss-depth-step") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->tss_depth_step = (unsigned)v;
        } else if (!strcmp(a, "--tss-budget-mult") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 1000) return -1; o->tss_budget_mult = (unsigned)v;
        } else if (!strcmp(a, "--tactical-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->tactical_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--tactical-depth") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 20) return -1; o->tactical_depth = (unsigned)v;
        } else if (!strcmp(a, "--tactical-candidates") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > MAX_MOVES) return -1; o->tactical_candidates = (unsigned)v;
        } else if (!strcmp(a, "--tactical-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->tactical_budget)) return -1;
        } else if (!strcmp(a, "--bridge-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->bridge_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--bridge-layers") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 4) return -1; o->bridge_layers = (unsigned)v;
        } else if (!strcmp(a, "--bridge-candidates") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > MAX_MOVES) return -1; o->bridge_candidates = (unsigned)v;
        } else if (!strcmp(a, "--bridge-levels") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 8) return -1; o->bridge_levels = (unsigned)v;
        } else if (!strcmp(a, "--bridge-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->bridge_budget)) return -1;
        } else if (!strcmp(a, "--bridge-budget-mult") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 1000) return -1; o->bridge_budget_mult = (unsigned)v;
        } else if (!strcmp(a, "--dep-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->dep_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--dep-full-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->dep_full_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--dep-full-tss-level") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 16) return -1; o->dep_full_tss_level = (unsigned)v;
        } else if (!strcmp(a, "--dep-dead-empty") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 49) return -1; o->dep_dead_empty = (unsigned)v;
        } else if (!strcmp(a, "--dep-deadcheck-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->dep_deadcheck_ply = (unsigned)v;
        } else if (!strcmp(a, "--dep-oracle-start-ply") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v > 1000) return -1; o->dep_oracle_start_ply = (unsigned)v;
        } else if (!strcmp(a, "--dep-oracle-layers") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > 4) return -1; o->dep_oracle_layers = (unsigned)v;
        } else if (!strcmp(a, "--dep-oracle-candidates") && i + 1 < argc) {
            uint64_t v; if (!parse_u64(argv[++i], &v) || v < 1 || v > MAX_MOVES) return -1; o->dep_oracle_candidates = (unsigned)v;
        } else if (!strcmp(a, "--dep-oracle-budget") && i + 1 < argc) {
            if (!parse_u64(argv[++i], &o->dep_oracle_budget)) return -1;
        } else if (a[0] != '-' && sequence_is_syntax(a)) {
            o->position = a;
        } else {
            fprintf(stderr, "Unknown/invalid option: %s\n", a);
            return -1;
        }
    }
    if (o->deep_reroot_harvest_min_depth > o->deep_reroot_harvest_max_depth) return -1;
    /* Harvesting needs the reroot subsolver profiler to reach the requested
       descendant depth.  Raising the profile depth here is safe and avoids a
       silent no-candidate configuration. */
    if (o->deep_reroot_harvest_enabled &&
        o->deep_reroot_harvest_max_depth > o->deep_frontier_depth)
        o->deep_frontier_depth = o->deep_reroot_harvest_max_depth;
    return 0;
}

static const char *outcome_name(int outcome)
{
    return outcome == 0 ? "WIN" : outcome == 1 ? "LOSS" : outcome == 2 ? "DRAW" : "UNKNOWN";
}

static void write_summary_csv(const Options *opt, const Board *root, int perspective,
                              int outcome, double total,
                              const PhaseSummary *w, const PhaseSummary *l)
{
    if (!opt->summary_path || !strcmp(opt->summary_path, "-")) return;
    FILE *fp = fopen(opt->summary_path, "w");
    if (!fp) {
        fprintf(stderr, "Warning: could not write summary %s\n", opt->summary_path);
        return;
    }
    fputs("label,rows,position,perspective,side_to_move,result,total_s,"
          "w_verdict,w_s,w_calls,w_exp,w_expanded_children,w_avg_branch,w_moves,"
          "w_tss_proofs,w_tss_calls,w_tss_cache,w_bridge_proofs,w_bridge_calls,w_collapses,"
          "w_root_pn,w_root_dn,w_maxply,"
          "l_verdict,l_s,l_calls,l_exp,l_expanded_children,l_avg_branch,l_moves,"
          "l_tss_proofs,l_tss_calls,l_tss_cache,l_bridge_proofs,l_bridge_calls,l_collapses,"
          "l_root_pn,l_root_dn,l_maxply\n", fp);

    double wb = (w && w->stats.expansions) ?
        (double)w->stats.expanded_children / (double)w->stats.expansions : 0.0;
    double lb = (l && l->stats.expansions) ?
        (double)l->stats.expanded_children / (double)l->stats.expansions : 0.0;

    fprintf(fp, "%s,%u,%s,%s,%s,%s,%.6f,"
            "%d,%.6f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.8f,%" PRIu64 ","
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u,"
            "%d,%.6f,%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%.8f,%" PRIu64 ","
            "%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%" PRIu64 ",%u,%u,%u\n",
            opt->label ? opt->label : "", opt->rows, opt->position ? opt->position : "",
            player_name(perspective), player_name(root->turn), outcome_name(outcome), total,
            w ? w->verdict : -2, w ? w->elapsed : 0.0,
            w ? w->stats.dfpn_calls : 0, w ? w->stats.expansions : 0,
            w ? w->stats.expanded_children : 0, wb, w ? w->stats.moves_generated : 0,
            w ? w->stats.tss_proofs : 0, w ? w->stats.tss_calls : 0,
            w ? w->stats.tss_cache_hits : 0, w ? w->stats.bridge_proofs : 0,
            w ? w->stats.bridge_calls : 0, w ? w->stats.oracle_collapses : 0,
            w ? w->root_pn : 0, w ? w->root_dn : 0, w ? w->stats.max_ply : 0,
            l ? l->verdict : -2, l ? l->elapsed : 0.0,
            l ? l->stats.dfpn_calls : 0, l ? l->stats.expansions : 0,
            l ? l->stats.expanded_children : 0, lb, l ? l->stats.moves_generated : 0,
            l ? l->stats.tss_proofs : 0, l ? l->stats.tss_calls : 0,
            l ? l->stats.tss_cache_hits : 0, l ? l->stats.bridge_proofs : 0,
            l ? l->stats.bridge_calls : 0, l ? l->stats.oracle_collapses : 0,
            l ? l->root_pn : 0, l ? l->root_dn : 0, l ? l->stats.max_ply : 0);
    fclose(fp);
}

int main(int argc, char **argv)
{
    signal(SIGINT, on_interrupt);
    Options opt;
    int po = parse_options(argc, argv, &opt);
    if (po == 1) return 0;
    if (po < 0) { usage(argv[0]); return 2; }
    if (opt.legacy_serial && opt.parallel_threads != 1) {
        fprintf(stderr,"--legacy-serial requires --threads 1.\n");
        return 2;
    }
    if (opt.self_test) return self_test();

    uint8_t floor_row = (uint8_t)(7 - opt.rows);
    Board root;
    int initial_status = parse_position(&root, floor_row, opt.position);
    if (initial_status == -2) return 2;

    if (opt.list_legal_moves) {
        if (initial_status >= 0) {
            printf("LEGAL_MOVES turn=%s count=0 moves=- terminal=%s\n",
                   player_name(root.turn), initial_status == 2 ? "Draw" : player_name(initial_status));
            return 0;
        }
        unsigned count = 0;
        char moves[256];
        size_t used = 0;
        moves[0] = '\0';
        for (int tile = 1; tile <= 3; ++tile) {
            for (int col = 0; col < COLS; ++col) {
                if (!legal_tile_at(&root, col, tile)) continue;
                char mbuf[3] = {(char)('0' + tile), (char)('A' + col), '\0'};
                int n = snprintf(moves + used, sizeof(moves) - used, "%s%s",
                                 count ? "," : "", mbuf);
                if (n < 0 || (size_t)n >= sizeof(moves) - used) {
                    fprintf(stderr, "Internal error formatting legal moves.\n");
                    return 3;
                }
                used += (size_t)n;
                count++;
            }
        }
        printf("LEGAL_MOVES turn=%s count=%u moves=%s\n",
               player_name(root.turn), count, count ? moves : "-");
        return 0;
    }

    puts("Make Seven DF-PN v9.8.3 SPDFPN + Persistent Exact Proposition DB solver");
    puts("Variant: unlimited tiles; exact sliding sum-to-7; red-cell restriction for tile 3");
    printf("Board: 7x%u (bottom %u original row%s removed)\n", opt.rows,
           (unsigned)floor_row, floor_row == 1 ? "" : "s");
    board_print(&root, floor_row);

    int perspective = opt.perspective >= 0 ? opt.perspective : root.turn;
    printf("Evaluation perspective: %s%s\n", player_name(perspective),
           opt.perspective < 0 ? " (side to move)" : " (fixed)");
    if (opt.exact_db_path) printf("Persistent exact proposition DB: %s (read-only exact phase facts)\n",opt.exact_db_path);
    if (opt.legacy_serial)
        printf("Search mode: legacy serial DF-PN (deep-reroot available).\n");
    else
        printf("Unified SPDFPN: %u worker%s; shared TT request=%.3f GiB; private TT/worker=%.1f MiB; job work=%" PRIu64 "; hot sync=%" PRIu64 "/d%u; busy penalty=%.3f\n",
               opt.parallel_threads,opt.parallel_threads==1?"":"s",opt.parallel_shared_tt_gb,opt.parallel_worker_tt_mb,opt.parallel_job_work,
               opt.parallel_publish_work,opt.parallel_publish_depth,opt.parallel_busy_penalty);

    if (initial_status == 0 || initial_status == 1 || initial_status == 2) {
        int outcome = initial_status == 2 ? 2 : (initial_status == perspective ? 0 : 1);
        if (initial_status == 2) puts("Position is already terminal: draw.");
        else printf("Position is already terminal: %s has won.\n", player_name(initial_status));
        printf("Result from %s perspective: %s\n", player_name(perspective), outcome_name(outcome));
        write_summary_csv(&opt, &root, perspective, outcome, 0.0, NULL, NULL);
        return 0;
    }

    ExactDB exact_db; memset(&exact_db,0,sizeof(exact_db));
    if (opt.exact_db_path && !exact_db_load(&exact_db,opt.exact_db_path,opt.rows)) return 2;

    Context ctx;
    memset(&ctx, 0, sizeof(ctx));
    ctx.exact_db = opt.exact_db_path ? &exact_db : NULL;
    ctx.floor_row = floor_row;
    ctx.tss_enabled = opt.tss_enabled;
    ctx.tss_compress = opt.tss_compress;
    ctx.tss_depth = opt.tss_depth;
    ctx.tss_budget = opt.tss_budget;
    ctx.tss_start_ply = opt.tss_start_ply;
    ctx.tss_levels = opt.tss_levels;
    ctx.tss_depth_step = opt.tss_depth_step;
    ctx.tss_budget_mult = opt.tss_budget_mult;
    ctx.tactical_enabled = opt.tactical_enabled;
    ctx.tactical_start_ply = opt.tactical_start_ply;
    ctx.tactical_depth = opt.tactical_depth;
    ctx.tactical_candidates = opt.tactical_candidates;
    ctx.tactical_budget = opt.tactical_budget;
    ctx.bridge_enabled = opt.bridge_enabled;
    ctx.bridge_start_ply = opt.bridge_start_ply;
    ctx.bridge_layers = opt.bridge_layers;
    ctx.bridge_candidates = opt.bridge_candidates;
    ctx.bridge_levels = opt.bridge_levels;
    ctx.bridge_budget = opt.bridge_budget;
    ctx.bridge_budget_mult = opt.bridge_budget_mult;
    ctx.dependency_enabled = opt.dependency_enabled;
    ctx.dep_full_enabled = opt.dep_full_enabled;
    ctx.dep_oracle_enabled = opt.dep_oracle_enabled;
    ctx.legacy_quiet_order = opt.legacy_quiet_order;
    ctx.or_dependency_only = opt.or_dependency_only;
    ctx.child_cache_enabled = opt.child_cache_enabled && opt.child_cache_mb > 0.0;
    ctx.frontier_enabled = opt.frontier_enabled;
    ctx.frontier_top = opt.frontier_top;
    ctx.deep_frontier_enabled = opt.deep_frontier_enabled;
    ctx.deep_frontier_depth = opt.deep_frontier_depth;
    ctx.deep_frontier_top = opt.deep_frontier_top;
    ctx.deep_frontier_max_nodes = opt.deep_frontier_max_nodes;
    ctx.hot_exact_enabled = opt.hot_exact_enabled;
    ctx.hot_exact_tt_mb = opt.hot_exact_tt_mb;
    ctx.hot_exact_depth = opt.hot_exact_depth;
    ctx.hot_exact_depth_step = opt.hot_exact_depth_step;
    ctx.hot_exact_levels = opt.hot_exact_levels;
    ctx.hot_exact_budget = opt.hot_exact_budget;
    ctx.hot_exact_budget_mult = opt.hot_exact_budget_mult;
    ctx.hot_exact_min_total_exp = opt.hot_exact_min_total_exp;
    ctx.hot_exact_stall_exp = opt.hot_exact_stall_exp;
    ctx.hot_exact_min_reply_exp = opt.hot_exact_min_reply_exp;
    ctx.hot_exact_retry_work = opt.hot_exact_retry_work;
    ctx.hot_exact_child_share = opt.hot_exact_child_share;
    ctx.hot_exact_reply_share = opt.hot_exact_reply_share;
    ctx.hot_tss_gate_enabled = opt.hot_tss_gate_enabled;
    ctx.hot_tss_min_calls = opt.hot_tss_min_calls;
    ctx.hot_tss_max_proof_rate = opt.hot_tss_max_proof_rate;
    ctx.hot_tss_level_cap = opt.hot_tss_level_cap;
    ctx.work_quantum = opt.work_quantum;
    ctx.exact_lemma_depth = opt.exact_lemma_depth;
    ctx.exact_lemma_budget = opt.exact_lemma_budget;
    ctx.deep_reroot_enabled = opt.deep_reroot_enabled;
    ctx.deep_reroot_memory_gb = opt.deep_reroot_memory_gb;
    ctx.deep_reroot_child_cache_mb = opt.deep_reroot_child_cache_mb;
    ctx.deep_reroot_time_sec = opt.deep_reroot_time_sec;
    ctx.deep_reroot_min_total_exp = opt.deep_reroot_min_total_exp;
    ctx.deep_reroot_stall_exp = opt.deep_reroot_stall_exp;
    ctx.deep_reroot_min_work = opt.deep_reroot_min_work;
    ctx.deep_reroot_retry_work = opt.deep_reroot_retry_work;
    ctx.deep_reroot_hot_age_sec = opt.deep_reroot_hot_age_sec;
    ctx.deep_reroot_hot_work = opt.deep_reroot_hot_work;
    ctx.deep_reroot_hot_max_change = opt.deep_reroot_hot_max_change;
    ctx.deep_reroot_hot_probe_sec = opt.deep_reroot_hot_probe_sec;
    ctx.deep_reroot_hot_gap_sec = opt.deep_reroot_hot_gap_sec;
    ctx.deep_reroot_hot_min_busy = opt.deep_reroot_hot_min_busy;
    ctx.deep_reroot_hot_region_depth = opt.deep_reroot_hot_region_depth;
    ctx.deep_reroot_root_share = opt.deep_reroot_root_share;
    ctx.deep_reroot_inner_share = opt.deep_reroot_inner_share;
    ctx.deep_reroot_min_depth = opt.deep_reroot_min_depth;
    ctx.deep_reroot_max_attempts = opt.deep_reroot_max_attempts;
    ctx.deep_reroot_max_nesting = opt.deep_reroot_max_nesting;
    ctx.deep_reroot_session_slots = opt.deep_reroot_session_slots;
    ctx.deep_reroot_profile_nodes = opt.deep_reroot_profile_nodes;
    ctx.deep_reroot_focus_slices = opt.deep_reroot_focus_slices;
    ctx.deep_reroot_harvest_enabled = opt.deep_reroot_harvest_enabled;
    ctx.deep_reroot_harvest_after_slices = opt.deep_reroot_harvest_after_slices;
    ctx.deep_reroot_harvest_min_depth = opt.deep_reroot_harvest_min_depth;
    ctx.deep_reroot_harvest_max_depth = opt.deep_reroot_harvest_max_depth;
    ctx.deep_reroot_harvest_attempts = opt.deep_reroot_harvest_attempts;
    ctx.deep_reroot_harvest_focus_slices = opt.deep_reroot_harvest_focus_slices;
    ctx.deep_reroot_harvest_min_work = opt.deep_reroot_harvest_min_work;
    ctx.deep_reroot_harvest_deep_min_work = opt.deep_reroot_harvest_deep_min_work;
    ctx.deep_reroot_harvest_deep_min_stability = opt.deep_reroot_harvest_deep_min_stability;
    ctx.deep_reroot_harvest_resume_pressure = opt.deep_reroot_harvest_resume_pressure;
    ctx.deep_reroot_harvest_time_sec = opt.deep_reroot_harvest_time_sec;
    ctx.deep_reroot_level = 0;
    ctx.deep_reroot_tss = opt.deep_reroot_tss;
    ctx.dep_start_ply = opt.dep_start_ply;
    ctx.dep_full_start_ply = opt.dep_full_start_ply;
    ctx.dep_full_tss_level = opt.dep_full_tss_level;
    ctx.dep_dead_empty = opt.dep_dead_empty;
    ctx.dep_deadcheck_ply = opt.dep_deadcheck_ply;
    ctx.dep_oracle_start_ply = opt.dep_oracle_start_ply;
    ctx.dep_oracle_layers = opt.dep_oracle_layers;
    ctx.dep_oracle_candidates = opt.dep_oracle_candidates;
    ctx.dep_oracle_budget = opt.dep_oracle_budget;
    ctx.report_interval = opt.report_interval;
    ctx.log_path = opt.log_path;
    ctx.frontier_log_path = opt.frontier_log_path;
    ctx.deep_frontier_log_path = opt.deep_frontier_log_path;
    ctx.start_time = wall_time();
    ctx.deadline = opt.time_limit > 0.0 ? ctx.start_time + opt.time_limit : 0.0;

    if (!tt_init(&ctx.tt, opt.memory_gb)) {
        fprintf(stderr, "Could not allocate transposition table. Try a smaller --memory-gb.\n");
        exact_db_destroy(&exact_db); return 3;
    }
    if (ctx.child_cache_enabled && !child_cache_init(&ctx.child_cache, opt.child_cache_mb)) {
        fprintf(stderr, "Could not allocate child-generation cache. Try a smaller --child-cache-mb.\n");
        tt_destroy(&ctx.tt);
        exact_db_destroy(&exact_db); return 3;
    }
    if (ctx.hot_exact_enabled && !hot_tt_init(&ctx.hot_tt, opt.hot_exact_tt_mb)) {
        fprintf(stderr, "Could not allocate hot-exact TT. Try a smaller --hot-exact-tt-mb.\n");
        child_cache_destroy(&ctx.child_cache);
        tt_destroy(&ctx.tt);
        exact_db_destroy(&exact_db); return 3;
    }
    if (ctx.deep_frontier_enabled && !deep_frontier_init(&ctx, &root)) {
        fprintf(stderr, "Could not allocate deep-frontier profiler. Try a smaller --deep-frontier-max-nodes or --no-deep-frontier.\n");
        hot_tt_destroy(&ctx.hot_tt);
        child_cache_destroy(&ctx.child_cache);
        tt_destroy(&ctx.tt);
        exact_db_destroy(&exact_db); return 3;
    }
    printf("TT: %.3f GiB, %zu sets, 2-way, exact state keys (%zu-byte entries)\n",
           ctx.tt.allocated_gb, ctx.tt.sets, sizeof(TTEntry));
    printf("Child-generation cache: %s", ctx.child_cache_enabled ? "on" : "off");
    if (ctx.child_cache_enabled) printf("; %.1f MiB, %zu direct-mapped exact-key entries",
                                        ctx.child_cache.allocated_mb, ctx.child_cache.sets);
    putchar('\n');
    printf("Proof-frontier diagnostics: %s; root + depth-2 work attribution; top=%u\n",
           ctx.frontier_enabled ? "on" : "off", ctx.frontier_top);
    printf("Deep proof-frontier profiler: %s; depth=%u; top=%u; node cap=%u (%.1f MiB reserved)\n",
           ctx.deep_frontier_enabled ? "on" : "off", ctx.deep_frontier_depth,
           ctx.deep_frontier_top, ctx.deep_frontier_max_nodes,
           ctx.deep_frontier_enabled ? ((double)ctx.deep_frontier.capacity * sizeof(DeepFrontierNode)) / (1024.0 * 1024.0) : 0.0);
    printf("Hot exact: %s; TT=%.1f MiB; depth=%u +%u; levels=%u; budget=%" PRIu64 " x%u; trigger total=%" PRIu64 " stall=%" PRIu64 " reply=%" PRIu64 " shares=%.0f%%/%.0f%%; TSS gate=%s cap=%u <=%.3f%% after %" PRIu64 " calls\n",
           ctx.hot_exact_enabled ? "on" : "off", ctx.hot_tt.allocated_mb,
           ctx.hot_exact_depth, ctx.hot_exact_depth_step, ctx.hot_exact_levels,
           ctx.hot_exact_budget, ctx.hot_exact_budget_mult, ctx.hot_exact_min_total_exp,
           ctx.hot_exact_stall_exp, ctx.hot_exact_min_reply_exp,
           100.0 * ctx.hot_exact_child_share, 100.0 * ctx.hot_exact_reply_share,
           ctx.hot_tss_gate_enabled ? "on" : "off", ctx.hot_tss_level_cap,
           100.0 * ctx.hot_tss_max_proof_rate, ctx.hot_tss_min_calls);
    printf("Work quantum: %s; W=%" PRIu64 " fresh DF-PN expansions/non-root call; root scheduler unlimited\n",
           ctx.work_quantum ? "on" : "off", ctx.work_quantum);
    printf("Deep reroot: %s; PERSISTENT+FOCUSED sessions=%u/level TT=%.3f GiB cache=%.1f MiB slice=%.1fs focus=%ux; "
           "serial NODE-STALL total=%" PRIu64 " root=%" PRIu64 " node=%" PRIu64 "; "
           "parallel HOT-REGION age=%.1fs work=%" PRIu64 " dPN/dDN-ref=%.1f%% minBusy=%u regionDepth<=%u probe=%.2fs gap=%.1fs targetWork>=%" PRIu64 " retry=%" PRIu64
           " minDepth=%u episodes=%u nesting=%u profile=%u TSS=%s\n",
           ctx.deep_reroot_enabled ? "on" : "off", ctx.deep_reroot_session_slots, ctx.deep_reroot_memory_gb,
           ctx.deep_reroot_child_cache_mb, ctx.deep_reroot_time_sec, ctx.deep_reroot_focus_slices,
           ctx.deep_reroot_min_total_exp, ctx.deep_reroot_stall_exp, ctx.deep_reroot_min_work,
           ctx.deep_reroot_hot_age_sec, ctx.deep_reroot_hot_work, 100.0*ctx.deep_reroot_hot_max_change,ctx.deep_reroot_hot_min_busy,ctx.deep_reroot_hot_region_depth,
           ctx.deep_reroot_hot_probe_sec, ctx.deep_reroot_hot_gap_sec,ctx.deep_reroot_min_work, ctx.deep_reroot_retry_work,
           ctx.deep_reroot_min_depth, ctx.deep_reroot_max_attempts,
           ctx.deep_reroot_max_nesting, ctx.deep_reroot_profile_nodes,
           ctx.deep_reroot_tss ? "on" : "off");
    if (ctx.deep_reroot_enabled && opt.legacy_serial && (!ctx.frontier_enabled || !ctx.deep_frontier_enabled)) {
        fprintf(stderr, "Warning: legacy-serial deep reroot requires frontier + deep-frontier profiling; disabling deep reroot.\n");
        ctx.deep_reroot_enabled = false;
    }
    if (ctx.deep_reroot_enabled && !opt.legacy_serial && (!ctx.frontier_enabled || !ctx.deep_frontier_enabled)) {
        printf("Parallel hybrid reroot: frontier profilers may stay off; candidate detection uses the v9.8.2 active-job reservation hot-region detector.\n");
    }
    printf("Adaptive TSS: %s; signature compression=%s; base depth=%u; base budget=%" PRIu64
           "; levels=%u; depth step=%u; budget x%u; starts at ply %u\n",
           ctx.tss_enabled ? "on" : "off", ctx.tss_compress ? "on" : "off",
           ctx.tss_depth, ctx.tss_budget, ctx.tss_levels, ctx.tss_depth_step,
           ctx.tss_budget_mult, ctx.tss_start_ply);
    printf("Bounded tactical pre-solver: %s; depth=%u; candidates=%u; budget=%" PRIu64
           "; starts at ply %u\n",
           ctx.tactical_enabled ? "on" : "off", ctx.tactical_depth,
           ctx.tactical_candidates, ctx.tactical_budget, ctx.tactical_start_ply);
    printf("Quiet-setup bridge: %s; layers=%u; candidates=%u; levels=%u; base budget=%" PRIu64
           "; budget x%u; starts at ply %u\n",
           ctx.bridge_enabled ? "on" : "off", ctx.bridge_layers, ctx.bridge_candidates,
           ctx.bridge_levels, ctx.bridge_budget, ctx.bridge_budget_mult, ctx.bridge_start_ply);
    printf("Dependency engine: %s; quiet order=%s; light ordering starts ply %u; full-map hints=%s "
           "from ply %u after TSS level %u; exact dead check <=%u empties (min ply %u); "
           "oracle=%s\n",
           ctx.dependency_enabled ? "on" : "off",
           ctx.legacy_quiet_order ? "center/tile-fast" : (ctx.or_dependency_only ? "OR-only-dependency" : "dependency-first-exp"), ctx.dep_start_ply,
           (ctx.dependency_enabled && ctx.dep_full_enabled && !ctx.legacy_quiet_order) ? "on" : "off",
           ctx.dep_full_start_ply, ctx.dep_full_tss_level, ctx.dep_dead_empty,
           ctx.dep_deadcheck_ply,
           (ctx.dependency_enabled && ctx.dep_oracle_enabled) ? "on" : "off");
    if (opt.time_limit > 0) printf("Total time limit: %.1f seconds\n", opt.time_limit);
    else puts("Total time limit: none");

    if (ctx.log_path && strcmp(ctx.log_path, "-") != 0) {
        ctx.log_fp = fopen(ctx.log_path, "w");
        if (!ctx.log_fp) fprintf(stderr, "Warning: could not open CSV log %s\n", ctx.log_path);
        else { csv_header(ctx.log_fp); printf("Benchmark CSV: %s\n", ctx.log_path); }
    }
    if (ctx.frontier_enabled && ctx.frontier_log_path && strcmp(ctx.frontier_log_path, "-") != 0) {
        ctx.frontier_fp = fopen(ctx.frontier_log_path, "w");
        if (!ctx.frontier_fp) fprintf(stderr, "Warning: could not open frontier CSV %s\n", ctx.frontier_log_path);
        else { frontier_csv_header(ctx.frontier_fp); printf("Frontier CSV: %s\n", ctx.frontier_log_path); }
    }
    if (ctx.deep_frontier_enabled && ctx.deep_frontier_log_path && strcmp(ctx.deep_frontier_log_path, "-") != 0) {
        ctx.deep_frontier_fp = fopen(ctx.deep_frontier_log_path, "w");
        if (!ctx.deep_frontier_fp) fprintf(stderr, "Warning: could not open deep-frontier CSV %s\n", ctx.deep_frontier_log_path);
        else { deep_frontier_csv_header(ctx.deep_frontier_fp); printf("Deep frontier CSV: %s\n", ctx.deep_frontier_log_path); }
    }

    if (opt.phase_only) {
        const char *pname = opt.phase_only==1 ? "W" : "L";
        uint8_t prover = (uint8_t)(opt.phase_only==1 ? perspective : (perspective ^ 1));
        PhaseSummary ps={0}; ps.verdict=-2;
        int pv = opt.legacy_serial ?
            run_phase(&ctx,&root,prover,pname,&ps) :
            run_phase_parallel(&ctx,&opt,&root,prover,pname,&ps);
        if (pv >= 0) {
            printf("[PHASE %s EXACT] prover=%s result=%s root=(%u,%u) elapsed=%.3f\n",
                   pname,player_name(prover),pv==1?"PROVED":"DISPROVED",
                   ps.root_pn,ps.root_dn,ps.elapsed);
            fflush(stdout);
        }
        double total = wall_time() - ctx.start_time;
        char dur[32]; format_duration(total,dur);
        puts("\n============= PHASE-ONLY FINAL =============");
        printf("Perspective: %s\n",player_name(perspective));
        printf("Phase-only: %s\n",pname);
        printf("Prover: %s\n",player_name(prover));
        printf("Proposition: %s\n",pv==1?"PROVED":pv==0?"DISPROVED":"UNKNOWN");
        printf("Total wall time: %s (%.3f s)\n",dur,total);
        puts("============================================");
        if (ctx.log_fp) fclose(ctx.log_fp);
        if (ctx.frontier_fp) fclose(ctx.frontier_fp);
        if (ctx.deep_frontier_fp) fclose(ctx.deep_frontier_fp);
        deep_reroot_sessions_destroy(&ctx);
        deep_frontier_destroy(&ctx);
        hot_tt_destroy(&ctx.hot_tt);
        child_cache_destroy(&ctx.child_cache);
        tt_destroy(&ctx.tt);
        exact_db_destroy(&exact_db);
        return pv < 0 ? 4 : 0;
    }

    PhaseSummary sw = {0}, sl = {0};
    sw.verdict = sl.verdict = -2;
    /* v9.8.2: by default all thread counts, including 1, use the same SPDFPN
       scheduler so scaling comparisons are algorithmically controlled. */
    int phase_w = opt.legacy_serial ?
        run_phase(&ctx, &root, (uint8_t)perspective, "W", &sw) :
        run_phase_parallel(&ctx, &opt, &root, (uint8_t)perspective, "W", &sw);
    if (phase_w >= 0) {
        printf("[PHASE W EXACT] prover=%s result=%s root=(%u,%u) elapsed=%.3f\n",
               player_name((uint8_t)perspective),phase_w==1?"PROVED":"DISPROVED",
               sw.root_pn,sw.root_dn,sw.elapsed);
        fflush(stdout);
    }
    uint8_t continuation = sw.root_best_move;
    int outcome = 3; /* unknown */
    if (phase_w == 1) {
        outcome = 0;
    } else if (phase_w == 0 && !ctx.stop) {
        int phase_l = opt.legacy_serial ?
            run_phase(&ctx, &root, (uint8_t)(perspective ^ 1), "L", &sl) :
            run_phase_parallel(&ctx, &opt, &root, (uint8_t)(perspective ^ 1), "L", &sl);
        if (phase_l >= 0) {
            printf("[PHASE L EXACT] prover=%s result=%s root=(%u,%u) elapsed=%.3f\n",
                   player_name((uint8_t)(perspective ^ 1)),phase_l==1?"PROVED":"DISPROVED",
                   sl.root_pn,sl.root_dn,sl.elapsed);
            fflush(stdout);
        }
        continuation = sl.root_best_move;
        if (phase_l == 1) outcome = 1;
        else if (phase_l == 0) outcome = 2;
    }

    double total = wall_time() - ctx.start_time;
    char dur[32]; format_duration(total, dur);
    puts("\n================ FINAL ================");
    printf("Perspective: %s\n", player_name(perspective));
    printf("Result: %s\n", outcome_name(outcome));
    if (continuation) printf("DF-PN root continuation: %d%c\n", move_tile(continuation), 'A' + move_col(continuation));
    printf("Total wall time: %s (%.3f s)\n", dur, total);
    puts("=======================================");

    write_summary_csv(&opt, &root, perspective, outcome, total, &sw,
                      sl.verdict == -2 ? NULL : &sl);
    if (ctx.log_fp) fclose(ctx.log_fp);
    if (ctx.frontier_fp) fclose(ctx.frontier_fp);
    if (ctx.deep_frontier_fp) fclose(ctx.deep_frontier_fp);
    deep_reroot_sessions_destroy(&ctx);
    deep_frontier_destroy(&ctx);
    hot_tt_destroy(&ctx.hot_tt);
    child_cache_destroy(&ctx.child_cache);
    tt_destroy(&ctx.tt);
    exact_db_destroy(&exact_db);
    return outcome == 3 ? 4 : 0;
}

