/*
 * Batched bit-parallel Myers edit distance, infix (HW) semantics, for aarch64 NEON.
 *
 * Semantics match edlib's mode="HW", task="distance": the minimum edit distance
 * between the full query and any substring of the target. Query length must be
 * <= 64 so the bitvector fits one word, which covers adapters, primers,
 * barcodes, UMIs and probes.
 *
 * Recurrence: Hyyro's formulation of Myers (1999). Per target character:
 *   Xv = Eq | VN
 *   Xh = (((Eq & VP) + VP) ^ VP) | Eq
 *   Ph = VN | ~(Xh | VP)
 *   Mh = VP & Xh
 *   score += (Ph >> (m-1)) & 1;  score -= (Mh >> (m-1)) & 1
 *   VP = (Mh << 1) | ~(Xv | (Ph << 1))
 *   VN = (Ph << 1) & Xv
 *
 * The horizontal carry into the shift is 0, which is what selects infix (HW)
 * semantics: the target may start matching anywhere. A carry-in of 1 would
 * instead give global (NW) semantics. This mirrors edlib's calculateBlock with
 * hin == 0.
 *
 * Ph and Mh are provably disjoint (Mh is a subset of VP, and Ph & VP == 0
 * because VP & VN == 0), so the two score updates can be applied
 * unconditionally. That is what makes the recurrence branch-free and therefore
 * vectorizable across lanes.
 *
 * Every operation is a per-lane 64-bit AND/OR/XOR/NOT/ADD/SHIFT, so two
 * independent alignments fit in one 128-bit NEON register with no cross-lane
 * traffic. The only per-lane divergence is the Peq table lookup, which stays an
 * L1-resident scalar load pair.
 */

#include <stddef.h>
#include <stdint.h>

#if defined(__aarch64__) && defined(__ARM_NEON)
#define MYERS_HAVE_NEON 1
#include <arm_neon.h>
#else
#define MYERS_HAVE_NEON 0
#endif

#define ALPHA 256 /* one entry per byte; the binding folds canonical DNA case */

/* ---------------------------------------------------------------- helpers */

static inline void build_peq(const uint8_t *pat, int m, uint64_t peq[ALPHA]) {
    for (int c = 0; c < ALPHA; c++) peq[c] = 0;
    for (int j = 0; j < m; j++) peq[pat[j]] |= (uint64_t)1 << j;
}

#if MYERS_HAVE_NEON
static inline uint64x2_t vnotq_u64(uint64x2_t x) {
    return vreinterpretq_u64_u8(vmvnq_u8(vreinterpretq_u8_u64(x)));
}
#endif

/* ----------------------------------------------------------------- scalar */

typedef struct {
    uint64_t VP, VN;
    int32_t score, best;
} myers_state;

static inline void state_init(myers_state *s, int m) {
    s->VP = ~(uint64_t)0;
    s->VN = 0;
    s->score = m;
    s->best = m; /* distance against the empty prefix is m deletions */
}

static inline void scalar_run(myers_state *s, const uint64_t peq[ALPHA], uint64_t mask, int shift,
                              const uint8_t *t, int n) {
    uint64_t VP = s->VP, VN = s->VN;
    int32_t score = s->score, best = s->best;
    for (int i = 0; i < n; i++) {
        const uint64_t Eq = peq[t[i]];
        const uint64_t Xv = Eq | VN;
        const uint64_t Xh = (((Eq & VP) + VP) ^ VP) | Eq;
        const uint64_t Ph = VN | ~(Xh | VP);
        const uint64_t Mh = VP & Xh;
        score += (int32_t)((Ph & mask) >> shift);
        score -= (int32_t)((Mh & mask) >> shift);
        const uint64_t Ph1 = Ph << 1;
        const uint64_t Mh1 = Mh << 1;
        VP = Mh1 | ~(Xv | Ph1);
        VN = Ph1 & Xv;
        if (score < best) best = score;
    }
    s->VP = VP;
    s->VN = VN;
    s->score = score;
    s->best = best;
}

/* One query against many targets, scalar reference path. */
void hw_batch_scalar(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
                     const int32_t *tlen, int32_t n_targets, int32_t *out) {
    uint64_t peq[ALPHA];
    build_peq(pat, m, peq);
    const uint64_t mask = (uint64_t)1 << (m - 1);
    const int shift = m - 1;
    for (int32_t k = 0; k < n_targets; k++) {
        myers_state s;
        state_init(&s, m);
        scalar_run(&s, peq, mask, shift, targets + toff[k], tlen[k]);
        out[k] = s.best;
    }
}

#if MYERS_HAVE_NEON

/* ------------------------------------------------------------------- NEON */

/*
 * Two targets advance in lockstep for min(n0, n1) characters; whichever target
 * is longer finishes on the scalar path from the extracted lane state. Equal
 * lengths, the norm for a sequencer run, take the vector path end to end.
 */
static void hw_pair_neon(const uint64_t peq[ALPHA], int m, const uint8_t *t0, int32_t n0,
                         const uint8_t *t1, int32_t n1, int32_t *o0, int32_t *o1) {
    const int shift = m - 1;
    const uint64_t maskw = (uint64_t)1 << shift;

    uint64x2_t VP = vdupq_n_u64(~(uint64_t)0);
    uint64x2_t VN = vdupq_n_u64(0);
    int64x2_t score = vdupq_n_s64(m);
    int64x2_t best = vdupq_n_s64(m);
    const uint64x2_t maskv = vdupq_n_u64(maskw);
    const int64x2_t shrv = vdupq_n_s64(-(int64_t)shift);


    const int32_t nmin = n0 < n1 ? n0 : n1;
    for (int32_t i = 0; i < nmin; i++) {
        const uint64x2_t Eq = vcombine_u64(vcreate_u64(peq[t0[i]]), vcreate_u64(peq[t1[i]]));
        const uint64x2_t Xv = vorrq_u64(Eq, VN);
        const uint64x2_t sum = vaddq_u64(vandq_u64(Eq, VP), VP);
        const uint64x2_t Xh = vorrq_u64(veorq_u64(sum, VP), Eq);
        const uint64x2_t Ph = vorrq_u64(VN, vnotq_u64(vorrq_u64(Xh, VP)));
        const uint64x2_t Mh = vandq_u64(VP, Xh);

        const int64x2_t pinc = vreinterpretq_s64_u64(vshlq_u64(vandq_u64(Ph, maskv), shrv));
        const int64x2_t minc = vreinterpretq_s64_u64(vshlq_u64(vandq_u64(Mh, maskv), shrv));
        score = vsubq_s64(vaddq_s64(score, pinc), minc);
        best = vreinterpretq_s64_u64(vbslq_u64(vcltq_s64(score, best),
                                              vreinterpretq_u64_s64(score),
                                              vreinterpretq_u64_s64(best)));

        const uint64x2_t Ph1 = vshlq_n_u64(Ph, 1);
        const uint64x2_t Mh1 = vshlq_n_u64(Mh, 1);
        VP = vorrq_u64(Mh1, vnotq_u64(vorrq_u64(Xv, Ph1)));
        VN = vandq_u64(Ph1, Xv);
    }

    myers_state s0 = {vgetq_lane_u64(VP, 0), vgetq_lane_u64(VN, 0), (int32_t)vgetq_lane_s64(score, 0),
                      (int32_t)vgetq_lane_s64(best, 0)};
    myers_state s1 = {vgetq_lane_u64(VP, 1), vgetq_lane_u64(VN, 1), (int32_t)vgetq_lane_s64(score, 1),
                      (int32_t)vgetq_lane_s64(best, 1)};
    if (n0 > nmin) scalar_run(&s0, peq, maskw, shift, t0 + nmin, n0 - nmin);
    if (n1 > nmin) scalar_run(&s1, peq, maskw, shift, t1 + nmin, n1 - nmin);
    *o0 = s0.best;
    *o1 = s1.best;
}

/* One query against many targets, two alignments per NEON register. */
void hw_batch_neon(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
                   const int32_t *tlen, int32_t n_targets, int32_t *out) {
    uint64_t peq[ALPHA];
    build_peq(pat, m, peq);
    const uint64_t mask = (uint64_t)1 << (m - 1);
    const int shift = m - 1;

    int32_t k = 0;
    for (; k + 1 < n_targets; k += 2) {
        hw_pair_neon(peq, m, targets + toff[k], tlen[k], targets + toff[k + 1], tlen[k + 1], out + k,
                     out + k + 1);
    }
    for (; k < n_targets; k++) {
        myers_state s;
        state_init(&s, m);
        scalar_run(&s, peq, mask, shift, targets + toff[k], tlen[k]);
        out[k] = s.best;
    }
}

/* Many queries against many targets; returns a query-major distance matrix. */
void hw_matrix_neon(const uint8_t *pats, const int32_t *poff, const int32_t *plen, int32_t n_pats,
                    const uint8_t *targets, const int32_t *toff, const int32_t *tlen,
                    int32_t n_targets, int32_t *out) {
    for (int32_t p = 0; p < n_pats; p++) {
        hw_batch_neon(pats + poff[p], plen[p], targets, toff, tlen, n_targets,
                      out + (size_t)p * n_targets);
    }
}

/* --------------------------------------------------- NEON, two chains deep */

/*
 * The recurrence above is one long dependency chain: the add feeds the xor,
 * which feeds the or, which feeds the next iteration. Two lanes therefore leave
 * most of the core's issue width idle. This variant advances four targets as two
 * register-independent chains, so the scheduler can overlap them.
 */

#define MYERS_STEP(VP, VN, score, best, e0, e1)                                              \
    do {                                                                                     \
        const uint64x2_t Eq = vcombine_u64(vcreate_u64(e0), vcreate_u64(e1));                \
        const uint64x2_t Xv = vorrq_u64(Eq, VN);                                             \
        const uint64x2_t sum = vaddq_u64(vandq_u64(Eq, VP), VP);                             \
        const uint64x2_t Xh = vorrq_u64(veorq_u64(sum, VP), Eq);                             \
        const uint64x2_t Ph = vorrq_u64(VN, vnotq_u64(vorrq_u64(Xh, VP)));                   \
        const uint64x2_t Mh = vandq_u64(VP, Xh);                                             \
        const int64x2_t pinc = vreinterpretq_s64_u64(vshlq_u64(vandq_u64(Ph, maskv), shrv)); \
        const int64x2_t minc = vreinterpretq_s64_u64(vshlq_u64(vandq_u64(Mh, maskv), shrv)); \
        score = vsubq_s64(vaddq_s64(score, pinc), minc);                                     \
        best = vreinterpretq_s64_u64(vbslq_u64(vcltq_s64(score, best),                       \
                                              vreinterpretq_u64_s64(score),                  \
                                              vreinterpretq_u64_s64(best)));                 \
        const uint64x2_t Ph1 = vshlq_n_u64(Ph, 1);                                           \
        const uint64x2_t Mh1 = vshlq_n_u64(Mh, 1);                                           \
        VP = vorrq_u64(Mh1, vnotq_u64(vorrq_u64(Xv, Ph1)));                                  \
        VN = vandq_u64(Ph1, Xv);                                                             \
    } while (0)

static void hw_quad_neon(const uint64_t peq[ALPHA], int m, const uint8_t *const t[4],
                         const int32_t n[4], int32_t *out) {
    const int shift = m - 1;
    const uint64_t maskw = (uint64_t)1 << shift;
    const uint64x2_t maskv = vdupq_n_u64(maskw);
    const int64x2_t shrv = vdupq_n_s64(-(int64_t)shift);

    uint64x2_t VPa = vdupq_n_u64(~(uint64_t)0), VNa = vdupq_n_u64(0);
    uint64x2_t VPb = vdupq_n_u64(~(uint64_t)0), VNb = vdupq_n_u64(0);
    int64x2_t sa = vdupq_n_s64(m), ba = vdupq_n_s64(m);
    int64x2_t sb = vdupq_n_s64(m), bb = vdupq_n_s64(m);

    int32_t nmin = n[0];
    for (int i = 1; i < 4; i++)
        if (n[i] < nmin) nmin = n[i];

    const uint8_t *t0 = t[0], *t1 = t[1], *t2 = t[2], *t3 = t[3];
    for (int32_t i = 0; i < nmin; i++) {
        MYERS_STEP(VPa, VNa, sa, ba, peq[t0[i]], peq[t1[i]]);
        MYERS_STEP(VPb, VNb, sb, bb, peq[t2[i]], peq[t3[i]]);
    }

    myers_state s[4] = {
        {vgetq_lane_u64(VPa, 0), vgetq_lane_u64(VNa, 0), (int32_t)vgetq_lane_s64(sa, 0),
         (int32_t)vgetq_lane_s64(ba, 0)},
        {vgetq_lane_u64(VPa, 1), vgetq_lane_u64(VNa, 1), (int32_t)vgetq_lane_s64(sa, 1),
         (int32_t)vgetq_lane_s64(ba, 1)},
        {vgetq_lane_u64(VPb, 0), vgetq_lane_u64(VNb, 0), (int32_t)vgetq_lane_s64(sb, 0),
         (int32_t)vgetq_lane_s64(bb, 0)},
        {vgetq_lane_u64(VPb, 1), vgetq_lane_u64(VNb, 1), (int32_t)vgetq_lane_s64(sb, 1),
         (int32_t)vgetq_lane_s64(bb, 1)},
    };
    for (int j = 0; j < 4; j++) {
        if (n[j] > nmin) scalar_run(&s[j], peq, maskw, shift, t[j] + nmin, n[j] - nmin);
        out[j] = s[j].best;
    }
}

/* One query against many targets, four alignments per iteration. */
void hw_batch_neon4(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
                    const int32_t *tlen, int32_t n_targets, int32_t *out) {
    uint64_t peq[ALPHA];
    build_peq(pat, m, peq);
    const uint64_t mask = (uint64_t)1 << (m - 1);
    const int shift = m - 1;

    int32_t k = 0;
    for (; k + 3 < n_targets; k += 4) {
        const uint8_t *t[4] = {targets + toff[k], targets + toff[k + 1], targets + toff[k + 2],
                               targets + toff[k + 3]};
        const int32_t n[4] = {tlen[k], tlen[k + 1], tlen[k + 2], tlen[k + 3]};
        hw_quad_neon(peq, m, t, n, out + k);
    }
    for (; k < n_targets; k++) {
        myers_state s;
        state_init(&s, m);
        scalar_run(&s, peq, mask, shift, targets + toff[k], tlen[k]);
        out[k] = s.best;
    }
}

/* -------------------------------------------------- NEON, four chains deep */

/*
 * Four independent chains, eight targets per iteration. Live vector state is
 * 16 registers (VP, VN, score, best per chain) out of the 32 the ISA provides,
 * so the constants stay resident and nothing spills.
 */
static void hw_oct_neon(const uint64_t peq[ALPHA], int m, const uint8_t *const t[8],
                        const int32_t n[8], int32_t *out) {
    const int shift = m - 1;
    const uint64_t maskw = (uint64_t)1 << shift;
    const uint64x2_t maskv = vdupq_n_u64(maskw);
    const int64x2_t shrv = vdupq_n_s64(-(int64_t)shift);

    uint64x2_t VP[4], VN[4];
    int64x2_t sc[4], bs[4];
    for (int c = 0; c < 4; c++) {
        VP[c] = vdupq_n_u64(~(uint64_t)0);
        VN[c] = vdupq_n_u64(0);
        sc[c] = vdupq_n_s64(m);
        bs[c] = vdupq_n_s64(m);
    }

    int32_t nmin = n[0];
    for (int i = 1; i < 8; i++)
        if (n[i] < nmin) nmin = n[i];

    for (int32_t i = 0; i < nmin; i++) {
        MYERS_STEP(VP[0], VN[0], sc[0], bs[0], peq[t[0][i]], peq[t[1][i]]);
        MYERS_STEP(VP[1], VN[1], sc[1], bs[1], peq[t[2][i]], peq[t[3][i]]);
        MYERS_STEP(VP[2], VN[2], sc[2], bs[2], peq[t[4][i]], peq[t[5][i]]);
        MYERS_STEP(VP[3], VN[3], sc[3], bs[3], peq[t[6][i]], peq[t[7][i]]);
    }

    for (int j = 0; j < 8; j++) {
        const int c = j >> 1, l = j & 1;
        myers_state s = {l ? vgetq_lane_u64(VP[c], 1) : vgetq_lane_u64(VP[c], 0),
                         l ? vgetq_lane_u64(VN[c], 1) : vgetq_lane_u64(VN[c], 0),
                         (int32_t)(l ? vgetq_lane_s64(sc[c], 1) : vgetq_lane_s64(sc[c], 0)),
                         (int32_t)(l ? vgetq_lane_s64(bs[c], 1) : vgetq_lane_s64(bs[c], 0))};
        if (n[j] > nmin) scalar_run(&s, peq, maskw, shift, t[j] + nmin, n[j] - nmin);
        out[j] = s.best;
    }
}

/* One query against many targets, eight alignments per iteration. */
void hw_batch_neon8(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
                    const int32_t *tlen, int32_t n_targets, int32_t *out) {
    uint64_t peq[ALPHA];
    build_peq(pat, m, peq);
    const uint64_t mask = (uint64_t)1 << (m - 1);
    const int shift = m - 1;

    int32_t k = 0;
    for (; k + 7 < n_targets; k += 8) {
        const uint8_t *t[8];
        int32_t n[8];
        for (int j = 0; j < 8; j++) {
            t[j] = targets + toff[k + j];
            n[j] = tlen[k + j];
        }
        hw_oct_neon(peq, m, t, n, out + k);
    }
    for (; k < n_targets; k++) {
        myers_state s;
        state_init(&s, m);
        scalar_run(&s, peq, mask, shift, targets + toff[k], tlen[k]);
        out[k] = s.best;
    }
}

#endif /* MYERS_HAVE_NEON */

/* ------------------------------------------------------------- dispatcher */

/* Widest available path: four-chain NEON on aarch64, scalar everywhere else. */
void hw_batch(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
              const int32_t *tlen, int32_t n_targets, int32_t *out) {
#if MYERS_HAVE_NEON
    hw_batch_neon8(pat, m, targets, toff, tlen, n_targets, out);
#else
    hw_batch_scalar(pat, m, targets, toff, tlen, n_targets, out);
#endif
}

int hw_have_neon(void) { return MYERS_HAVE_NEON; }
