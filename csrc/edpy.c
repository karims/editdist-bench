/*
 * CPython extension: edit distance straight from Python str/bytes objects,
 * with no conversion pass. Compact str objects whose characters all fit in
 * one byte (Latin-1, which includes ASCII) store exactly those bytes, so the
 * kernel reads them in place. Wider strings are rejected (ValueError) rather
 * than silently compared code-unit-wise.
 *
 * Reuses the kernels by including myers.c (shared static functions).
 */
#define PY_SSIZE_T_CLEAN
#include <Python.h>
#include "myers.c"

static int view(PyObject *o, const uint8_t **p, int *len) {
    if (PyUnicode_Check(o)) {
#if PY_VERSION_HEX < 0x030C0000
        if (PyUnicode_READY(o) < 0) return -1;
#endif
        if (PyUnicode_KIND(o) != PyUnicode_1BYTE_KIND) {
            PyErr_SetString(PyExc_ValueError, "string has characters above U+00FF; "
                                              "pass bytes (e.g. UTF-8 encoded) instead");
            return -1;
        }
        *p = PyUnicode_1BYTE_DATA(o);
        *len = (int)PyUnicode_GET_LENGTH(o);
        return 0;
    }
    if (PyBytes_Check(o)) {
        *p = (const uint8_t *)PyBytes_AS_STRING(o);
        *len = (int)PyBytes_GET_SIZE(o);
        return 0;
    }
    PyErr_SetString(PyExc_TypeError, "expected str or bytes");
    return -1;
}

typedef struct { const uint8_t **ptr; int *len; Py_ssize_t n; PyObject *fast; } seq_t;

static void seq_free(seq_t *s) {
    PyMem_Free(s->ptr); PyMem_Free(s->len); Py_XDECREF(s->fast);
}

static int seq_load(PyObject *obj, seq_t *s) {
    s->ptr = NULL; s->len = NULL;
    s->fast = PySequence_Fast(obj, "expected a sequence of str/bytes");
    if (!s->fast) return -1;
    s->n = PySequence_Fast_GET_SIZE(s->fast);
    s->ptr = PyMem_Malloc(sizeof(*s->ptr) * (s->n ? s->n : 1));
    s->len = PyMem_Malloc(sizeof(*s->len) * (s->n ? s->n : 1));
    if (!s->ptr || !s->len) { seq_free(s); PyErr_NoMemory(); return -1; }
    PyObject **items = PySequence_Fast_ITEMS(s->fast);
    for (Py_ssize_t i = 0; i < s->n; i++)
        if (view(items[i], &s->ptr[i], &s->len[i]) < 0) { seq_free(s); return -1; }
    return 0;
}

/* pairs(a_list, b_list, out: writable int32 buffer, trim: int) */
static PyObject *py_pairs(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *oa, *ob; Py_buffer out; int trim = 1;
    if (!PyArg_ParseTuple(args, "OOw*|p", &oa, &ob, &out, &trim)) return NULL;
    seq_t A, B;
    if (seq_load(oa, &A) < 0) { PyBuffer_Release(&out); return NULL; }
    if (seq_load(ob, &B) < 0) { seq_free(&A); PyBuffer_Release(&out); return NULL; }
    PyObject *ret = NULL;
    if (A.n != B.n) { PyErr_SetString(PyExc_ValueError, "length mismatch"); goto done; }
    if (out.len < (Py_ssize_t)(A.n * sizeof(int32_t))) {
        PyErr_SetString(PyExc_ValueError, "output buffer too small"); goto done;
    }
    int mx = 0;
    for (Py_ssize_t i = 0; i < A.n; i++) {
        int m = A.len[i] < B.len[i] ? A.len[i] : B.len[i];
        if (m > mx) mx = m;
    }
    ws_t ws;
    if (ws_init(&ws, (mx + 63) >> 6)) { PyErr_NoMemory(); goto done; }
    word *Pv = malloc(sizeof(word) * ws.W), *Mv = malloc(sizeof(word) * ws.W);
    if (!Pv || !Mv) { free(Pv); free(Mv); ws_free(&ws); PyErr_NoMemory(); goto done; }
    int32_t *o = (int32_t *)out.buf;
    for (Py_ssize_t i = 0; i < A.n; i++)
        o[i] = pair_distance(&ws, Pv, Mv, A.ptr[i], A.len[i], B.ptr[i], B.len[i], trim);
    free(Pv); free(Mv); ws_free(&ws);
    ret = Py_None; Py_INCREF(ret);
done:
    seq_free(&A); seq_free(&B); PyBuffer_Release(&out);
    return ret;
}

/* cross(q_list, c_list, out: writable int32 buffer of len(q)*len(c)) */
static PyObject *py_cross(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *oq, *oc; Py_buffer out;
    if (!PyArg_ParseTuple(args, "OOw*", &oq, &oc, &out)) return NULL;
    seq_t Q, C;
    if (seq_load(oq, &Q) < 0) { PyBuffer_Release(&out); return NULL; }
    if (seq_load(oc, &C) < 0) { seq_free(&Q); PyBuffer_Release(&out); return NULL; }
    PyObject *ret = NULL;
    if (out.len < (Py_ssize_t)(Q.n * C.n * sizeof(int32_t))) {
        PyErr_SetString(PyExc_ValueError, "output buffer too small"); goto done;
    }
    int mx = 0;
    for (Py_ssize_t i = 0; i < Q.n; i++) if (Q.len[i] > mx) mx = Q.len[i];
    ws_t ws;
    if (ws_init(&ws, (mx + 63) >> 6)) { PyErr_NoMemory(); goto done; }
    word *Pv = malloc(sizeof(word) * ws.W), *Mv = malloc(sizeof(word) * ws.W);
    if (!Pv || !Mv) { free(Pv); free(Mv); ws_free(&ws); PyErr_NoMemory(); goto done; }
    int32_t *o = (int32_t *)out.buf;
    for (Py_ssize_t i = 0; i < Q.n; i++) {
        int32_t *row = o + i * C.n;
        if (Q.len[i] == 0) { for (Py_ssize_t j = 0; j < C.n; j++) row[j] = C.len[j]; continue; }
        load_pattern(&ws, Q.ptr[i], Q.len[i]);
        for (Py_ssize_t j = 0; j < C.n; j++)
            row[j] = run_text(&ws, Q.len[i], C.ptr[j], C.len[j], Pv, Mv);
        clear_pattern(&ws, Q.ptr[i], Q.len[i]);
    }
    free(Pv); free(Mv); ws_free(&ws);
    ret = Py_None; Py_INCREF(ret);
done:
    seq_free(&Q); seq_free(&C); PyBuffer_Release(&out);
    return ret;
}

/* pairs_simd(a_list, b_list, out, trim=True, k=4): NEON inter-pair kernel */
static PyObject *py_pairs_simd(PyObject *self, PyObject *args) {
    (void)self;
    PyObject *oa, *ob; Py_buffer out; int trim = 1, k = 4;
    if (!PyArg_ParseTuple(args, "OOw*|pi", &oa, &ob, &out, &trim, &k)) return NULL;
    seq_t A, B;
    if (seq_load(oa, &A) < 0) { PyBuffer_Release(&out); return NULL; }
    if (seq_load(ob, &B) < 0) { seq_free(&A); PyBuffer_Release(&out); return NULL; }
    PyObject *ret = NULL;
    if (A.n != B.n) { PyErr_SetString(PyExc_ValueError, "length mismatch"); goto done; }
    if (out.len < (Py_ssize_t)(A.n * sizeof(int32_t))) {
        PyErr_SetString(PyExc_ValueError, "output buffer too small"); goto done;
    }
    int rc = pairs_simd_core(A.ptr, A.len, B.ptr, B.len, A.n, (int32_t *)out.buf, trim, k);
    if (rc == -2) { PyErr_SetString(PyExc_ValueError, "k must be 1, 2 or 4"); goto done; }
    if (rc != 0) { PyErr_NoMemory(); goto done; }
    ret = Py_None; Py_INCREF(ret);
done:
    seq_free(&A); seq_free(&B); PyBuffer_Release(&out);
    return ret;
}

static PyMethodDef methods[] = {
    {"pairs_simd", py_pairs_simd, METH_VARARGS, "pairs_simd(a, b, out, trim=True, k=4)"},
    {"pairs", py_pairs, METH_VARARGS, "pairs(a, b, out, trim=True)"},
    {"cross", py_cross, METH_VARARGS, "cross(q, c, out)"},
    {NULL, NULL, 0, NULL},
};

static struct PyModuleDef module = {PyModuleDef_HEAD_INIT, "_edpy", NULL, -1, methods,
                                    NULL, NULL, NULL, NULL};

PyMODINIT_FUNC PyInit__edpy(void) { return PyModule_Create(&module); }
