/*
 * CPython bindings for the batched infix Myers kernel.
 *
 * The GIL is released around the kernel call, so callers can shard a batch
 * across a thread pool and scale past a single core.
 */

#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>

#define MAX_QUERY 64

void hw_batch(const uint8_t *pat, int32_t m, const uint8_t *const *targets,
              const int32_t *tlen, int32_t n_targets, int32_t *out);
void hw_batch_scalar(const uint8_t *pat, int32_t m, const uint8_t *const *targets,
                     const int32_t *tlen, int32_t n_targets, int32_t *out);
int hw_have_neon(void);
int hw_simd_backend(void);

#define EXACT_LENGTH_BUCKETS 4096
#define LOG_LENGTH_BUCKETS (19 * 32)
#define LENGTH_BUCKETS (EXACT_LENGTH_BUCKETS + LOG_LENGTH_BUCKETS)

typedef struct {
    const uint8_t **target;
    int32_t *len;
    int32_t *orig;
    uint8_t *snapshot_buf;
    Py_ssize_t n;
    int uniform_length;
    Py_buffer *bufs;
} packed;

static void packed_free(packed *p) {
    if (p->bufs) {
        for (Py_ssize_t i = 0; i < p->n; i++)
            if (p->bufs[i].obj) PyBuffer_Release(&p->bufs[i]);
        free(p->bufs);
    }
    free(p->target);
    free(p->len);
    free(p->orig);
    free(p->snapshot_buf);
    p->target = NULL;
    p->len = p->orig = NULL;
    p->snapshot_buf = NULL;
    p->bufs = NULL;
}

static unsigned length_bucket(int32_t n) {
    if (n < EXACT_LENGTH_BUCKETS) return (unsigned)n;
    unsigned log2 = 0;
    for (unsigned value = (unsigned)n; value >>= 1;) log2++;
    const unsigned fraction = ((unsigned)n >> (log2 - 5U)) & 31U;
    return EXACT_LENGTH_BUCKETS + (log2 - 12U) * 32U + fraction;
}

static int group_targets_by_length(packed *p) {
    size_t *next = (size_t *)calloc(LENGTH_BUCKETS, sizeof(size_t));
    const uint8_t **target = (const uint8_t **)malloc((size_t)p->n * sizeof(*target));
    int32_t *len = (int32_t *)malloc((size_t)p->n * sizeof(*len));
    int32_t *orig = (int32_t *)malloc((size_t)p->n * sizeof(*orig));
    if (!next || !target || !len || !orig) {
        free(next);
        free(target);
        free(len);
        free(orig);
        PyErr_NoMemory();
        return -1;
    }

    for (Py_ssize_t i = 0; i < p->n; i++) next[length_bucket(p->len[i])]++;
    size_t total = 0;
    for (size_t b = 0; b < LENGTH_BUCKETS; b++) {
        const size_t count = next[b];
        next[b] = total;
        total += count;
    }
    for (Py_ssize_t i = 0; i < p->n; i++) {
        const size_t pos = next[length_bucket(p->len[i])]++;
        target[pos] = p->target[i];
        len[pos] = p->len[i];
        orig[pos] = (int32_t)i;
    }
    free(next);
    free(p->target);
    free(p->len);
    p->target = target;
    p->len = len;
    p->orig = orig;
    return 0;
}

/* Validate targets and retain their buffers. Exact bytes objects are immutable
 * and can be read directly while the GIL is released. Every other exporter is
 * snapshotted because a read-only view may still alias mutable backing storage. */
static int pack_targets(PyObject *seq, packed *p, int group_by_length) {
    PyObject *fast = PySequence_Fast(seq, "targets must be a sequence of bytes-like objects");
    if (!fast) return -1;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > INT32_MAX) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_OverflowError, "too many targets");
        return -1;
    }

    const size_t alloc_n = (size_t)(n > 0 ? n : 1);
    p->n = n;
    p->bufs = (Py_buffer *)calloc(alloc_n, sizeof(Py_buffer));
    p->len = (int32_t *)malloc(alloc_n * sizeof(*p->len));
    if (!p->bufs || !p->len) {
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    Py_ssize_t total = 0;
    Py_ssize_t snapshot_total = 0;
    int uniform_length = 1;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        if (PyObject_GetBuffer(item, &p->bufs[i], PyBUF_SIMPLE) < 0) {
            Py_DECREF(fast);
            return -1;
        }
        const Py_ssize_t item_len = p->bufs[i].len;
        if (item_len > INT32_MAX || total > INT32_MAX - item_len) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_OverflowError, "total target length exceeds 2GB");
            return -1;
        }
        p->len[i] = (int32_t)item_len;
        if (i > 0 && p->len[i] != p->len[0]) uniform_length = 0;
        total += item_len;
        if (!PyBytes_Check(item)) snapshot_total += item_len;
    }

    p->uniform_length = uniform_length;
    if (n == 0 || (p->uniform_length && p->len[0] == 0)) {
        Py_DECREF(fast);
        return 0;
    }
    p->target = (const uint8_t **)malloc((size_t)n * sizeof(*p->target));
    if (!p->target) {
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }
    if (snapshot_total) {
        p->snapshot_buf = (uint8_t *)malloc((size_t)snapshot_total);
        if (!p->snapshot_buf) {
            Py_DECREF(fast);
            PyErr_NoMemory();
            return -1;
        }
    }
    size_t snapshot_pos = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        const uint8_t *src = (const uint8_t *)p->bufs[i].buf;
        if (PyBytes_Check(p->bufs[i].obj) || p->len[i] == 0) {
            p->target[i] = src;
        } else {
            p->target[i] = p->snapshot_buf + snapshot_pos;
            memcpy(p->snapshot_buf + snapshot_pos, src, (size_t)p->len[i]);
            snapshot_pos += (size_t)p->len[i];
        }
    }
    Py_DECREF(fast);
    if (group_by_length && !p->uniform_length && n > 1 && group_targets_by_length(p) < 0)
        return -1;
    return 0;
}

static PyObject *run(PyObject *args, int force_scalar) {
    Py_buffer qb;
    PyObject *seq;
    if (!PyArg_ParseTuple(args, "y*O", &qb, &seq)) return NULL;

    if (qb.len < 1 || qb.len > MAX_QUERY) {
        PyBuffer_Release(&qb);
        return PyErr_Format(PyExc_ValueError, "query length must be 1..%d, got %zd", MAX_QUERY,
                            qb.len);
    }

    uint8_t q[MAX_QUERY];
    const uint8_t *qsrc = (const uint8_t *)qb.buf;
    memcpy(q, qsrc, (size_t)qb.len);
    const int32_t m = (int32_t)qb.len;
    PyBuffer_Release(&qb);

    packed p = {0};
    if (pack_targets(seq, &p, !force_scalar && hw_simd_backend() != 0) < 0) {
        packed_free(&p);
        return NULL;
    }
    if (p.uniform_length && p.n > 0 && p.len[0] == 0) {
        PyObject *list = PyList_New(p.n);
        PyObject *distance = list ? PyLong_FromLong((long)m) : NULL;
        if (!list || !distance) {
            Py_XDECREF(list);
            packed_free(&p);
            return NULL;
        }
        for (Py_ssize_t i = 0; i < p.n; i++) {
            Py_INCREF(distance);
            PyList_SET_ITEM(list, i, distance);
        }
        Py_DECREF(distance);
        packed_free(&p);
        return list;
    }


    int32_t *out = (int32_t *)malloc((size_t)(p.n > 0 ? p.n : 1) * sizeof(int32_t));
    if (!out) {
        packed_free(&p);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS;
    if (force_scalar)
        hw_batch_scalar(q, m, p.target, p.len, (int32_t)p.n, out);
    else
        hw_batch(q, m, p.target, p.len, (int32_t)p.n, out);
    Py_END_ALLOW_THREADS;

    PyObject *list = PyList_New(p.n);
    if (list) {
        for (Py_ssize_t i = 0; i < p.n; i++) {
            const Py_ssize_t source = p.orig ? p.orig[i] : i;
            PyObject *v = PyLong_FromLong((long)out[i]);
            if (!v) {
                Py_CLEAR(list);
                break;
            }
            PyList_SET_ITEM(list, source, v);
        }
    }
    free(out);
    packed_free(&p);
    return list;
}

static PyObject *py_distances(PyObject *self, PyObject *args) {
    (void)self;
    return run(args, 0);
}

static PyObject *py_distances_scalar(PyObject *self, PyObject *args) {
    (void)self;
    return run(args, 1);
}

static PyObject *py_have_neon(PyObject *self, PyObject *noargs) {
    (void)self;
    (void)noargs;
    return PyBool_FromLong(hw_have_neon());
}

static PyObject *py_simd_backend(PyObject *self, PyObject *noargs) {
    (void)self;
    (void)noargs;
    const int backend = hw_simd_backend();
    return PyUnicode_FromString(backend == 2 ? "neon" : backend == 1 ? "avx2" : "scalar");
}

static PyMethodDef methods[] = {
    {"distances", py_distances, METH_VARARGS,
     "distances(query, targets) -> list[int]\n\n"
     "Minimum edit distance between query and any substring of each target\n"
     "(edlib mode='HW'). Query length must be 1..64."},
    {"distances_scalar", py_distances_scalar, METH_VARARGS,
     "distances_scalar(query, targets) -> list[int]\n\n"
     "Same result via the portable scalar path. Used for differential testing."},
    {"have_neon", py_have_neon, METH_NOARGS,
     "have_neon() -> bool\n\nTrue when the aarch64 NEON path is compiled in."},
    {"simd_backend", py_simd_backend, METH_NOARGS,
     "simd_backend() -> str\n\nActive kernel: 'neon', 'avx2', or 'scalar'."},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "myers_batch._kernel",
    "Batched infix (HW) bit-parallel Myers edit distance.", -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__kernel(void) {
    PyObject *mod = PyModule_Create(&moduledef);
    if (!mod) return NULL;
    if (PyModule_AddIntConstant(mod, "MAX_QUERY", MAX_QUERY) < 0) {
        Py_DECREF(mod);
        return NULL;
    }
    return mod;
}
