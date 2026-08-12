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

#define MAX_QUERY 64

void hw_batch(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
              const int32_t *tlen, int32_t n_targets, int32_t *out);
void hw_batch_scalar(const uint8_t *pat, int32_t m, const uint8_t *targets, const int32_t *toff,
                     const int32_t *tlen, int32_t n_targets, int32_t *out);
int hw_have_neon(void);

/* ACGT and acgt collapse to the same four codes. Every other byte retains its
 * own code, so unknown symbols match only the identical byte. */
static uint8_t CODE[256];

static void init_code(void) {
    for (int i = 0; i < 256; i++) CODE[i] = (uint8_t)i;
    CODE[(unsigned char)'a'] = CODE[(unsigned char)'A'];
    CODE[(unsigned char)'c'] = CODE[(unsigned char)'C'];
    CODE[(unsigned char)'g'] = CODE[(unsigned char)'G'];
    CODE[(unsigned char)'t'] = CODE[(unsigned char)'T'];
}

typedef struct {
    uint8_t *buf;
    int32_t *off;
    int32_t *len;
    Py_ssize_t n;
    PyObject **views; /* borrowed buffers to release */
    Py_buffer *bufs;
} packed;

static void packed_free(packed *p) {
    if (p->bufs) {
        for (Py_ssize_t i = 0; i < p->n; i++)
            if (p->bufs[i].obj) PyBuffer_Release(&p->bufs[i]);
        free(p->bufs);
    }
    free(p->buf);
    free(p->off);
    free(p->len);
    free(p->views);
    p->buf = NULL;
    p->off = p->len = NULL;
    p->bufs = NULL;
    p->views = NULL;
}

/* Encode a sequence of bytes-like targets into one contiguous coded buffer. */
static int pack_targets(PyObject *seq, packed *p) {
    PyObject *fast = PySequence_Fast(seq, "targets must be a sequence of bytes-like objects");
    if (!fast) return -1;
    const Py_ssize_t n = PySequence_Fast_GET_SIZE(fast);
    if (n > INT32_MAX) {
        Py_DECREF(fast);
        PyErr_SetString(PyExc_OverflowError, "too many targets");
        return -1;
    }

    p->n = n;
    p->bufs = (Py_buffer *)calloc((size_t)(n > 0 ? n : 1), sizeof(Py_buffer));
    p->off = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    p->len = (int32_t *)malloc((size_t)(n > 0 ? n : 1) * sizeof(int32_t));
    if (!p->bufs || !p->off || !p->len) {
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }

    Py_ssize_t total = 0;
    for (Py_ssize_t i = 0; i < n; i++) {
        PyObject *item = PySequence_Fast_GET_ITEM(fast, i);
        if (PyObject_GetBuffer(item, &p->bufs[i], PyBUF_SIMPLE) < 0) {
            Py_DECREF(fast);
            return -1;
        }
        p->off[i] = (int32_t)total;
        p->len[i] = (int32_t)p->bufs[i].len;
        total += p->bufs[i].len;
        if (total > INT32_MAX) {
            Py_DECREF(fast);
            PyErr_SetString(PyExc_OverflowError, "total target length exceeds 2GB");
            return -1;
        }
    }

    p->buf = (uint8_t *)malloc((size_t)(total > 0 ? total : 1));
    if (!p->buf) {
        Py_DECREF(fast);
        PyErr_NoMemory();
        return -1;
    }
    for (Py_ssize_t i = 0; i < n; i++) {
        const uint8_t *srcb = (const uint8_t *)p->bufs[i].buf;
        uint8_t *dst = p->buf + p->off[i];
        for (int32_t j = 0; j < p->len[i]; j++) dst[j] = CODE[srcb[j]];
    }
    Py_DECREF(fast);
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
    for (Py_ssize_t i = 0; i < qb.len; i++) q[i] = CODE[qsrc[i]];
    const int32_t m = (int32_t)qb.len;
    PyBuffer_Release(&qb);

    packed p = {0};
    if (pack_targets(seq, &p) < 0) {
        packed_free(&p);
        return NULL;
    }

    int32_t *out = (int32_t *)malloc((size_t)(p.n > 0 ? p.n : 1) * sizeof(int32_t));
    if (!out) {
        packed_free(&p);
        return PyErr_NoMemory();
    }

    Py_BEGIN_ALLOW_THREADS;
    if (force_scalar)
        hw_batch_scalar(q, m, p.buf, p.off, p.len, (int32_t)p.n, out);
    else
        hw_batch(q, m, p.buf, p.off, p.len, (int32_t)p.n, out);
    Py_END_ALLOW_THREADS;

    PyObject *list = PyList_New(p.n);
    if (list) {
        for (Py_ssize_t i = 0; i < p.n; i++) {
            PyObject *v = PyLong_FromLong((long)out[i]);
            if (!v) {
                Py_CLEAR(list);
                break;
            }
            PyList_SET_ITEM(list, i, v);
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
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef moduledef = {
    PyModuleDef_HEAD_INIT, "myers_batch._kernel",
    "Batched infix (HW) bit-parallel Myers edit distance.", -1, methods, NULL, NULL, NULL, NULL,
};

PyMODINIT_FUNC PyInit__kernel(void) {
    init_code();
    PyObject *mod = PyModule_Create(&moduledef);
    if (!mod) return NULL;
    if (PyModule_AddIntConstant(mod, "MAX_QUERY", MAX_QUERY) < 0) {
        Py_DECREF(mod);
        return NULL;
    }
    return mod;
}
