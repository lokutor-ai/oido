// On-device CTC prefix beam search with GRU language-model shallow fusion (mirrors train/beam.py).
// LM weights: int8 per-row (tasr_qlin row layout), activations dynamically quantized per step.
#include <math.h>
#include <string.h>
#include "kernels.h"
#include "tinyasr.h"
#include "tinyasr_lm.h"

#define NEG (-1e30f)

struct tasr_lm {
    int V, E, H;
    tasr_qlin_t emb;       // "linear" with n=V rows of E: row i = embedding of token i (scale per row)
    tasr_qlin_t w_ih, w_hh, out;
    int kp_max;
    float dflt_weight, dflt_bonus;  // recommended beam-search weights stored in the header (0 = not set)
};

typedef struct {
    const uint8_t *p, *end;
    int err;
} lcur_t;
static void lalign(lcur_t *c) { c->p = (const uint8_t *)(((uintptr_t)c->p + 15) & ~(uintptr_t)15); }
static const float *lf32(lcur_t *c, size_t n)
{
    lalign(c);
    const float *r = (const float *)c->p;
    c->p += n * 4;
    if (c->p > c->end) c->err = 1;
    return r;
}
static tasr_qlin_t lqlin(lcur_t *c, int n, int k, int *kpm)
{
    tasr_qlin_t L;
    memset(&L, 0, sizeof(L));
    L.n = n; L.k = k; L.bits = 8; L.kp = (k + 15) & ~15;
    lalign(c);
    L.w = (const int8_t *)c->p;
    c->p += (size_t)n * L.kp;
    L.s = lf32(c, n);
    L.b = lf32(c, n);
    if (L.kp > *kpm) *kpm = L.kp;
    return L;
}

tasr_lm_t *tasr_lm_load(const uint8_t *blob, size_t size)
{
    if (size < 32 || memcmp(blob, "TLM1", 4)) return NULL;
    uint32_t hd[7];
    memcpy(hd, blob + 4, sizeof(hd));
    tasr_lm_t *lm = (tasr_lm_t *)tasr_alloc(sizeof(tasr_lm_t), 1);
    lm->V = hd[0]; lm->E = hd[1]; lm->H = hd[2];
    lm->dflt_weight = hd[3] / 100.0f; lm->dflt_bonus = hd[4] / 100.0f;
    lcur_t c = {blob + 32, blob + size, 0};
    int kpm = 0;
    lm->emb = lqlin(&c, lm->V, lm->E, &kpm);
    lm->w_ih = lqlin(&c, 3 * lm->H, lm->E, &kpm);
    lm->w_hh = lqlin(&c, 3 * lm->H, lm->H, &kpm);
    lm->out = lqlin(&c, lm->V, lm->H, &kpm);
    lm->kp_max = kpm;
    if (c.err) { tasr_free(lm); return NULL; }
    return lm;
}
void tasr_lm_free(tasr_lm_t *lm) { tasr_free(lm); }
float tasr_lm_weight(const tasr_lm_t *lm, float fallback) { return lm && lm->dflt_weight > 0.f ? lm->dflt_weight : fallback; }
float tasr_lm_bonus(const tasr_lm_t *lm, float fallback) { return lm && lm->dflt_bonus > 0.f ? lm->dflt_bonus : fallback; }

static size_t lm_move(tasr_qlin_t *L)
{
    size_t wb = (size_t)L->n * L->kp;
    int8_t *p = (int8_t *)tasr_alloc(wb, 0);
    if (!p) return 0;
    memcpy(p, L->w, wb);
    L->w = p;
    return wb;
}
size_t tasr_lm_to_ram(tasr_lm_t *lm)
{
    return lm_move(&lm->emb) + lm_move(&lm->w_ih) + lm_move(&lm->w_hh) + lm_move(&lm->out);
}

static inline float fsig(float x) { return 1.0f / (1.0f + expf(-x)); }

// ---------------------------------------------------------------- decoder
#define MAXB 16
#define MAXK 16

typedef struct {
    int32_t parent;
    int16_t tok, state;  // state: LM slot or -1
    float lm;            // accumulated LM log-prob of the prefix
    int32_t len;
} node_t;

typedef struct {
    int node;       // >= 0: existing prefix node; -1: pending child (parent, tok)
    int parent, tok;
    float pb, pnb, lm;
    int len;
} cand_t;

typedef struct {
    int node;
    float pb, pnb;
} beam_t;

struct tasr_decoder {
    const tasr_lm_t *lm;
    int beam, topk, V1;
    float lm_weight, token_bonus, blank_skip_logp;
    node_t *nodes;
    int n_nodes, max_nodes;
    int *hash;
    int hash_cap;
    float *st_h, *st_lp;  // LM state pool: [slots][H], [slots][V]
    int16_t *free_slots;
    int n_free, max_slots;
    beam_t beams[MAXB];
    int n_beams;
    int8_t *xq;
    float *xs, *gi, *gh, *tmp;
    int8_t *wtmp, *wtmp2;
    int32_t *acc, *acc2;
};

typedef struct {
    tasr_decoder_t *d;
    const tasr_qlin_t *L;
    int T, ldy;
    float *y;
} lmjob_t;
static void lm_gemm_job(void *c, int b, int e, int w)
{
    lmjob_t *j = (lmjob_t *)c;
    tasr_qlin_range(j->L, j->d->xq, j->d->xs, j->T, j->d->lm->kp_max, j->y, j->ldy, w ? j->d->wtmp2 : j->d->wtmp,
                    w ? j->d->acc2 : j->d->acc, b, e);
}
static void lm_qlin(tasr_decoder_t *d, const tasr_qlin_t *L, int T, float *y, int ldy)
{
    lmjob_t j = {d, L, T, ldy, y};
    TASR_PAR(12, lm_gemm_job, &j, L->n);  // split output rows across both cores
}

static int slot_alloc(tasr_decoder_t *d) { return d->n_free > 0 ? d->free_slots[--d->n_free] : -1; }
static void slot_release(tasr_decoder_t *d, int s)
{
    if (s >= 0 && d->n_free < d->max_slots) d->free_slots[d->n_free++] = (int16_t)s;
}

#define LMB 16  // max LM steps batched per frame
// Batched GRU step: for each i, state slot out[i] = GRU(state parent[i] (or zeros if < 0), token tok[i]).
// One GEMM per weight matrix for the whole batch -> each LM weight is read once per frame.
static void lm_steps(tasr_decoder_t *d, const int *parent, const int *tok, const int *out, int n)
{
    const tasr_lm_t *lm = d->lm;
    const int H = lm->H, V = lm->V, E = lm->E, ldq = lm->kp_max;
    if (n <= 0) return;
    for (int i = 0; i < n; i++) {
        float *x = d->tmp + (size_t)i * (E > H ? E : H);
        const int8_t *er = lm->emb.w + (size_t)tok[i] * lm->emb.kp;
        for (int e = 0; e < E; e++) x[e] = (float)er[e] * lm->emb.s[tok[i]];
    }
    const int ldt = E > H ? E : H;
    tasr_quant_rows(d->tmp, n, E, ldt, d->xq, ldq, lm->w_ih.kp, d->xs);
    lm_qlin(d, &lm->w_ih, n, d->gi, 3 * H);
    for (int i = 0; i < n; i++) {  // previous hidden states (zeros for BOS)
        float *hp = d->tmp + (size_t)i * ldt;
        if (parent[i] >= 0) memcpy(hp, d->st_h + (size_t)parent[i] * H, sizeof(float) * H);
        else memset(hp, 0, sizeof(float) * H);
    }
    tasr_quant_rows(d->tmp, n, H, ldt, d->xq, ldq, lm->w_hh.kp, d->xs);
    lm_qlin(d, &lm->w_hh, n, d->gh, 3 * H);
    for (int i = 0; i < n; i++) {
        const float *gi = d->gi + (size_t)i * 3 * H, *gh = d->gh + (size_t)i * 3 * H;
        const float *hp = d->tmp + (size_t)i * ldt;
        float *h = d->st_h + (size_t)out[i] * H;
        for (int j = 0; j < H; j++) {
            float r = fsig(gi[j] + gh[j]);
            float z = fsig(gi[H + j] + gh[H + j]);
            float nn = tanhf(gi[2 * H + j] + r * gh[2 * H + j]);
            h[j] = (1.f - z) * nn + z * hp[j];
        }
        memcpy(d->tmp + (size_t)i * ldt, h, sizeof(float) * H);
    }
    tasr_quant_rows(d->tmp, n, H, ldt, d->xq, ldq, lm->out.kp, d->xs);
    lm_qlin(d, &lm->out, n, d->gi, V);
    for (int i = 0; i < n; i++) {
        const float *lo = d->gi + (size_t)i * V;
        float *lp = d->st_lp + (size_t)out[i] * V;
        float mx = lo[0];
        for (int v = 1; v < V; v++) mx = lo[v] > mx ? lo[v] : mx;
        float sum = 0.f;
        for (int v = 0; v < V; v++) sum += expf(lo[v] - mx);
        const float lz = mx + logf(sum);
        for (int v = 0; v < V; v++) lp[v] = lo[v] - lz;
    }
}

static int lm_step(tasr_decoder_t *d, int parent_slot, int tok)
{
    int slot = slot_alloc(d);
    if (slot < 0) return -1;
    lm_steps(d, &parent_slot, &tok, &slot, 1);
    return slot;
}

static inline unsigned hkey(int parent, int tok) { return ((unsigned)parent * 2654435761u) ^ ((unsigned)tok * 40503u); }
static int find_child(const tasr_decoder_t *d, int parent, int tok)
{
    for (int probe = 0; probe < d->hash_cap; probe++) {
        int n = d->hash[(hkey(parent, tok) + probe) & (d->hash_cap - 1)];
        if (n < 0) return -1;
        if (d->nodes[n].parent == parent && d->nodes[n].tok == tok) return n;
    }
    return -1;
}
static int make_child(tasr_decoder_t *d, int parent, int tok, float lm, int len)
{
    if (d->n_nodes >= d->max_nodes) return -1;
    for (int probe = 0; probe < d->hash_cap; probe++) {
        int i = (hkey(parent, tok) + probe) & (d->hash_cap - 1);
        if (d->hash[i] < 0) {
            int n = d->n_nodes++;
            node_t *nd = &d->nodes[n];
            nd->parent = parent; nd->tok = (int16_t)tok; nd->state = -1; nd->lm = lm; nd->len = len;
            d->hash[i] = n;
            return n;
        }
    }
    return -1;
}

static inline float logadd(float a, float b)
{
    if (a < b) { float t = a; a = b; b = t; }
    if (b <= NEG * 0.5f) return a;
    return a + log1pf(expf(b - a));
}

void tasr_decoder_reset(tasr_decoder_t *d)
{
    d->n_nodes = 1;
    d->nodes[0].parent = -1; d->nodes[0].tok = -1; d->nodes[0].len = 0; d->nodes[0].lm = 0.f; d->nodes[0].state = -1;
    for (int i = 0; i < d->hash_cap; i++) d->hash[i] = -1;
    d->n_free = 0;
    for (int i = d->max_slots - 1; i >= 0; i--) d->free_slots[d->n_free++] = (int16_t)i;
    if (d->lm) d->nodes[0].state = (int16_t)lm_step(d, -1, d->lm->V - 1);  // BOS = last LM token
    d->n_beams = 1;
    d->beams[0].node = 0; d->beams[0].pb = 0.f; d->beams[0].pnb = NEG;
}

tasr_decoder_t *tasr_decoder_create(const tasr_lm_t *lm, int V1, int beam, int topk, float lm_weight, float token_bonus)
{
    tasr_decoder_t *d = (tasr_decoder_t *)tasr_alloc(sizeof(tasr_decoder_t), 1);
    d->lm = lm; d->V1 = V1;
    d->beam = beam > MAXB ? MAXB : beam;
    d->topk = topk > MAXK ? MAXK : topk;
    d->lm_weight = lm_weight; d->token_bonus = token_bonus;
    d->blank_skip_logp = logf(0.999f);
    d->max_nodes = 16384;
    d->nodes = (node_t *)tasr_alloc(sizeof(node_t) * d->max_nodes, 0);
    d->hash_cap = 32768;
    d->hash = (int *)tasr_alloc(sizeof(int) * d->hash_cap, 0);
    d->max_slots = 2 * MAXB + 2;
    d->free_slots = (int16_t *)tasr_alloc(sizeof(int16_t) * d->max_slots, 1);
    if (lm) {
        d->st_h = (float *)tasr_alloc(sizeof(float) * d->max_slots * lm->H, 0);
        d->st_lp = (float *)tasr_alloc(sizeof(float) * d->max_slots * lm->V, 0);
        int big = 3 * lm->H > lm->V ? 3 * lm->H : lm->V;
        d->xq = (int8_t *)tasr_alloc((size_t)LMB * lm->kp_max, 0);
        d->xs = (float *)tasr_alloc(sizeof(float) * LMB, 1);
        d->gi = (float *)tasr_alloc(sizeof(float) * LMB * big, 0);
        d->gh = (float *)tasr_alloc(sizeof(float) * LMB * big, 0);
        d->tmp = (float *)tasr_alloc(sizeof(float) * LMB * (lm->E > lm->H ? lm->E : lm->H), 0);
        d->wtmp = (int8_t *)tasr_alloc(lm->kp_max, 1);
        d->wtmp2 = (int8_t *)tasr_alloc(lm->kp_max, 1);
        d->acc = (int32_t *)tasr_alloc(sizeof(int32_t) * 64 * 16, 1);
        d->acc2 = (int32_t *)tasr_alloc(sizeof(int32_t) * 64 * 16, 1);
    }
    tasr_decoder_reset(d);
    return d;
}

void tasr_decoder_free(tasr_decoder_t *d)
{
    if (!d) return;
    tasr_free(d->nodes); tasr_free(d->hash); tasr_free(d->free_slots);
    if (d->lm) {
        tasr_free(d->st_h); tasr_free(d->st_lp); tasr_free(d->xq); tasr_free(d->xs); tasr_free(d->gi);
        tasr_free(d->gh); tasr_free(d->tmp); tasr_free(d->wtmp); tasr_free(d->acc); tasr_free(d->wtmp2); tasr_free(d->acc2);
    }
    tasr_free(d);
}

static inline float cand_score(const tasr_decoder_t *d, const cand_t *c)
{
    return logadd(c->pb, c->pnb) + d->lm_weight * c->lm + d->token_bonus * (float)c->len;
}

void tasr_decoder_step(tasr_decoder_t *d, const float *lp)
{
    if (lp[0] > d->blank_skip_logp) {  // blank-dominated frame
        for (int i = 0; i < d->n_beams; i++) {
            beam_t *b = &d->beams[i];
            b->pb = logadd(b->pb, b->pnb) + lp[0];
            b->pnb = NEG;
        }
        return;
    }
    int cand[MAXK];
    float cv[MAXK];
    int nc = 0;
    for (int v = 1; v < d->V1; v++) {  // top-k non-blank classes (insertion sort)
        float x = lp[v];
        if (nc < d->topk || x > cv[nc - 1]) {
            int j = nc < d->topk ? nc++ : nc - 1;
            while (j > 0 && cv[j - 1] < x) { cv[j] = cv[j - 1]; cand[j] = cand[j - 1]; j--; }
            cv[j] = x; cand[j] = v;
        }
    }
    cand_t cs[MAXB * (1 + 2 * MAXK)];
    int nn = 0;
    for (int i = 0; i < d->n_beams; i++) {
        const beam_t b = d->beams[i];
        const node_t *bn = &d->nodes[b.node];
        const float ptot = logadd(b.pb, b.pnb);
        const float *plp = (d->lm && bn->state >= 0) ? d->st_lp + (size_t)bn->state * d->lm->V : NULL;
        for (int j = -1; j < nc; j++) {
            // j == -1: blank extension of the same prefix
            int tok = j < 0 ? -1 : cand[j] - 1;
            float p = j < 0 ? lp[0] : cv[j];
            for (int part = 0; part < (tok == bn->tok && j >= 0 ? 2 : 1); part++) {
                cand_t c;
                float add_pb = NEG, add_pnb = NEG;
                if (j < 0) {                     // blank
                    c.node = b.node; add_pb = ptot + p;
                } else if (tok == bn->tok && part == 0) {  // repeat collapses into same prefix
                    c.node = b.node; add_pnb = b.pnb + p;
                } else {                          // new token (or repeat after blank)
                    c.node = find_child(d, b.node, tok);
                    add_pnb = (tok == bn->tok ? b.pb : ptot) + p;
                }
                if (c.node >= 0) {
                    c.parent = d->nodes[c.node].parent; c.tok = d->nodes[c.node].tok;
                    c.lm = d->nodes[c.node].lm; c.len = d->nodes[c.node].len;
                } else {
                    c.parent = b.node; c.tok = tok;
                    c.lm = bn->lm + (plp ? plp[tok] : 0.f);
                    c.len = bn->len + 1;
                }
                int f = -1;
                for (int q = 0; q < nn; q++)
                    if ((c.node >= 0 && cs[q].node == c.node) ||
                        (c.node < 0 && cs[q].node < 0 && cs[q].parent == c.parent && cs[q].tok == c.tok)) { f = q; break; }
                if (f < 0) { f = nn++; cs[f] = c; cs[f].pb = NEG; cs[f].pnb = NEG; }
                cs[f].pb = logadd(cs[f].pb, add_pb);
                cs[f].pnb = logadd(cs[f].pnb, add_pnb);
            }
        }
    }
    // select best `beam` candidates
    beam_t old[MAXB];
    int n_old = d->n_beams;
    memcpy(old, d->beams, sizeof(beam_t) * n_old);
    d->n_beams = 0;
    for (int k = 0; k < d->beam && k < nn; k++) {
        int best = -1;
        float bs = NEG;
        for (int q = 0; q < nn; q++) {
            if (cs[q].tok == -32768) continue;
            float sc = cand_score(d, &cs[q]);
            if (sc > bs) { bs = sc; best = q; }
        }
        if (best < 0) break;
        cand_t *c = &cs[best];
        int node = c->node >= 0 ? c->node : make_child(d, c->parent, c->tok, c->lm, c->len);
        c->tok = -32768;  // mark used
        if (node < 0) continue;
        d->beams[d->n_beams].node = node;
        d->beams[d->n_beams].pb = c->pb;
        d->beams[d->n_beams].pnb = c->pnb;
        d->n_beams++;
    }
    // LM states: compute for new beam nodes, release for dropped ones
    if (d->lm) {
        int par[LMB] = {0}, tk[LMB] = {0}, outs[LMB] = {0}, who[LMB] = {0}, nq = 0;
        for (int i = 0; i < d->n_beams && nq < LMB; i++) {
            node_t *n = &d->nodes[d->beams[i].node];
            if (n->state < 0 && n->parent >= 0) {
                int sl = slot_alloc(d);
                if (sl < 0) continue;
                par[nq] = d->nodes[n->parent].state; tk[nq] = n->tok; outs[nq] = sl; who[nq] = d->beams[i].node; nq++;
            }
        }
        lm_steps(d, par, tk, outs, nq);
        for (int q = 0; q < nq; q++) d->nodes[who[q]].state = (int16_t)outs[q];
        for (int i = 0; i < n_old; i++) {
            int keep = 0;
            for (int j = 0; j < d->n_beams; j++)
                if (d->beams[j].node == old[i].node) keep = 1;
            node_t *n = &d->nodes[old[i].node];
            if (!keep && n->state >= 0) { slot_release(d, n->state); n->state = -1; }
        }
    }
}

int tasr_decoder_best(tasr_decoder_t *d, int *toks, int max)
{
    int best = 0;
    float bs = NEG;
    for (int i = 0; i < d->n_beams; i++) {
        const node_t *n = &d->nodes[d->beams[i].node];
        float sc = logadd(d->beams[i].pb, d->beams[i].pnb) + d->lm_weight * n->lm + d->token_bonus * (float)n->len;
        if (sc > bs) { bs = sc; best = i; }
    }
    int n = d->beams[best].node, len = d->nodes[n].len;
    int out = len > max ? max : len;
    for (int i = len - 1; n > 0; n = d->nodes[n].parent, i--)
        if (i < out) toks[i] = d->nodes[n].tok;
    return out;
}
