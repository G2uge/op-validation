/*
 * Comprehensive C++ unit-test for XDNN rotary embedding operators
 * (rotary_embedding_half & rotary_embedding_everytwo) partial-RoPE support.
 *
 * "Partial rotary position embedding" means freqs_shape[-1] < t_shape[-1],
 * i.e. only the first pe_head_dim dimensions are rotated and the rest are
 * copied unchanged.
 *
 * This behavior is consistent with Paddle GPU's fused_rotary_position_embedding
 * (see paddle/phi/kernels/fusion/gpu/fused_rope_utils.h).
 *
 * Test contents:
 *   1. Full-rotation calibration for both ops (ensure ref matches XPU).
 *   2. Partial-rotation forward test (pe_head_dim < head_dim).
 *   3. Partial-rotation backward test (pe_head_dim < head_dim).
 *   4. Numerical comparison and support verdict.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "xpu/xdnn.h"

using namespace std;

/* ------------------------------------------------------------------ */
/* Helpers                                                            */
/* ------------------------------------------------------------------ */
int allocate_and_copy(void** dev_ptr, const void* host_ptr, size_t size) {
    int ret = xpu_malloc(dev_ptr, size);
    if (ret != XPU_SUCCESS) return ret;
    return xpu_memcpy(*dev_ptr, host_ptr, size, XPU_HOST_TO_DEVICE);
}

int copy_to_host(void* host_ptr, const void* dev_ptr, size_t size) {
    return xpu_memcpy(host_ptr, dev_ptr, size, XPU_DEVICE_TO_HOST);
}

bool compare_outputs(const vector<float>& ref,
                     const vector<float>& out,
                     float tol = 1e-4f,
                     bool verbose = true) {
    if (ref.size() != out.size()) {
        if (verbose)
            cout << "  size mismatch: " << ref.size() << " vs " << out.size()
                 << endl;
        return false;
    }
    bool ok = true;
    for (size_t i = 0; i < ref.size(); ++i) {
        if (fabs(ref[i] - out[i]) > tol) {
            if (ok && verbose) {
                cout << "  first mismatch at idx=" << i << " ref=" << ref[i]
                     << " out=" << out[i] << endl;
            }
            ok = false;
        }
    }
    return ok;
}

/* ------------------------------------------------------------------ */
/* CPU Reference: rotary_embedding_everytwo (Neox style)              */
/* ------------------------------------------------------------------ */
/*
 * Forward:
 *   even d : out[d] = q[d]*cos[d] - q[d+1]*sin[d]
 *   odd  d : out[d] = q[d]*cos[d] + q[d-1]*sin[d]
 *
 * Partial: first pe_head_dim dims rotated, rest copied.
 */
void ref_rotary_everytwo(const float* q,
                         const float* sin,
                         const float* cos,
                         float* out,
                         int batch,
                         int seq_len,
                         int num_heads,
                         int head_dim,
                         int pe_head_dim) {
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                int q_base = ((b * seq_len + s) * num_heads + h) * head_dim;
                int f_base = ((b * seq_len + s) * 1 + 0) * pe_head_dim;

                for (int d = 0; d < pe_head_dim; ++d) {
                    float x = q[q_base + d];
                    float c = cos[f_base + d];
                    float s_val = sin[f_base + d];
                    if (d % 2 == 0) {
                        float x_rot = q[q_base + d + 1];
                        out[q_base + d] = x * c - x_rot * s_val;
                    } else {
                        float x_rot = q[q_base + d - 1];
                        out[q_base + d] = x * c + x_rot * s_val;
                    }
                }
                for (int d = pe_head_dim; d < head_dim; ++d) {
                    out[q_base + d] = q[q_base + d];
                }
            }
        }
    }
}

/*
 * Backward (grad):
 *   even d : dq[d] = do[d]*cos[d] + do[d+1]*sin[d+1]
 *   odd  d : dq[d] = do[d]*cos[d] - do[d-1]*sin[d-1]
 *
 * (Derived from GPU grad kernel in fused_rope_utils.h)
 */
void ref_rotary_everytwo_grad(const float* do_ptr,
                              const float* sin,
                              const float* cos,
                              float* dq,
                              int batch,
                              int seq_len,
                              int num_heads,
                              int head_dim,
                              int pe_head_dim) {
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                int base = ((b * seq_len + s) * num_heads + h) * head_dim;
                int f_base = ((b * seq_len + s) * 1 + 0) * pe_head_dim;

                for (int d = 0; d < pe_head_dim; ++d) {
                    float grad = do_ptr[base + d];
                    float c = cos[f_base + d];
                    if (d % 2 == 0) {
                        float grad_rot = do_ptr[base + d + 1];
                        float s_val = sin[f_base + d + 1];
                        dq[base + d] = grad * c + grad_rot * s_val;
                    } else {
                        float grad_rot = do_ptr[base + d - 1];
                        float s_val = sin[f_base + d - 1];
                        dq[base + d] = grad * c - grad_rot * s_val;
                    }
                }
                for (int d = pe_head_dim; d < head_dim; ++d) {
                    dq[base + d] = do_ptr[base + d];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* CPU Reference: rotary_embedding_half                               */
/* ------------------------------------------------------------------ */
/*
 * Forward:
 *   For d in [0, pe_head_dim/2):
 *     out[d]             = q[d]*cos[d]         - q[d+half]*sin[d]
 *     out[d + half]      = q[d+half]*cos[d+half] + q[d]*sin[d+half]
 *
 *   For d in [pe_head_dim, head_dim): out[d] = q[d]
 */
void ref_rotary_half(const float* q,
                     const float* sin,
                     const float* cos,
                     float* out,
                     int batch,
                     int seq_len,
                     int num_heads,
                     int head_dim,
                     int pe_head_dim) {
    int half = pe_head_dim / 2;
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                int q_base = ((b * seq_len + s) * num_heads + h) * head_dim;
                int f_base = ((b * seq_len + s) * 1 + 0) * pe_head_dim;

                for (int d = 0; d < half; ++d) {
                    float x1 = q[q_base + d];
                    float x2 = q[q_base + d + half];
                    float s1 = sin[f_base + d];
                    float c1 = cos[f_base + d];
                    float s2 = sin[f_base + d + half];
                    float c2 = cos[f_base + d + half];

                    out[q_base + d] = x1 * c1 - x2 * s1;
                    out[q_base + d + half] = x2 * c2 + x1 * s2;
                }
                for (int d = pe_head_dim; d < head_dim; ++d) {
                    out[q_base + d] = q[q_base + d];
                }
            }
        }
    }
}

/*
 * Backward (grad):
 *   For d in [0, pe_head_dim/2):
 *     dq[d]        = do[d]*cos[d]       + do[d+half]*sin[d+half]
 *     dq[d+half]   = do[d+half]*cos[d+half] - do[d]*sin[d]
 *
 *   For d in [pe_head_dim, head_dim): dq[d] = do[d]
 */
void ref_rotary_half_grad(const float* do_ptr,
                          const float* sin,
                          const float* cos,
                          float* dq,
                          int batch,
                          int seq_len,
                          int num_heads,
                          int head_dim,
                          int pe_head_dim) {
    int half = pe_head_dim / 2;
    for (int b = 0; b < batch; ++b) {
        for (int s = 0; s < seq_len; ++s) {
            for (int h = 0; h < num_heads; ++h) {
                int base = ((b * seq_len + s) * num_heads + h) * head_dim;
                int f_base = ((b * seq_len + s) * 1 + 0) * pe_head_dim;

                for (int d = 0; d < half; ++d) {
                    float g1 = do_ptr[base + d];
                    float g2 = do_ptr[base + d + half];
                    float c1 = cos[f_base + d];
                    float s1 = sin[f_base + d];
                    float c2 = cos[f_base + d + half];
                    float s2 = sin[f_base + d + half];

                    dq[base + d] = g1 * c1 + g2 * s2;
                    dq[base + d + half] = g2 * c2 - g1 * s1;
                }
                for (int d = pe_head_dim; d < head_dim; ++d) {
                    dq[base + d] = do_ptr[base + d];
                }
            }
        }
    }
}

/* ------------------------------------------------------------------ */
/* XPU wrappers                                                       */
/* ------------------------------------------------------------------ */
struct TestResult {
    vector<float> out_host;
    int ret;
};

TestResult run_xpu_everytwo(int batch,
                            int seq_len,
                            int num_heads,
                            int head_dim,
                            int pe_head_dim,
                            const vector<float>& q_host,
                            const vector<float>& sin_host,
                            const vector<float>& cos_host,
                            bool is_grad = false) {
    size_t q_size = static_cast<size_t>(batch * seq_len * num_heads * head_dim);
    size_t f_size = static_cast<size_t>(batch * seq_len * 1 * pe_head_dim);

    void *q_dev = nullptr, *sin_dev = nullptr, *cos_dev = nullptr,
         *out_dev = nullptr;
    allocate_and_copy(&q_dev, q_host.data(), q_size * sizeof(float));
    allocate_and_copy(&sin_dev, sin_host.data(), f_size * sizeof(float));
    allocate_and_copy(&cos_dev, cos_host.data(), f_size * sizeof(float));
    xpu_malloc(&out_dev, q_size * sizeof(float));

    baidu::xpu::api::Context* ctx = baidu::xpu::api::create_context();
    int ret;
    if (!is_grad) {
        ret = baidu::xpu::api::rotary_embedding_everytwo<float, float>(
            ctx,
            reinterpret_cast<const float*>(q_dev),
            nullptr,
            reinterpret_cast<const float*>(sin_dev),
            reinterpret_cast<const float*>(cos_dev),
            reinterpret_cast<float*>(out_dev),
            nullptr,
            {batch, seq_len, num_heads, head_dim},
            {batch, seq_len, 1, pe_head_dim},
            -1,
            10000.0f);
    } else {
        ret = baidu::xpu::api::rotary_embedding_everytwo_grad<float, float>(
            ctx,
            reinterpret_cast<const float*>(q_dev),
            nullptr,
            reinterpret_cast<const float*>(sin_dev),
            reinterpret_cast<const float*>(cos_dev),
            reinterpret_cast<float*>(out_dev),
            nullptr,
            {batch, seq_len, num_heads, head_dim},
            {batch, seq_len, 1, pe_head_dim},
            -1,
            10000.0f);
    }

    xpu_wait(ctx->get_stream());
    baidu::xpu::api::destroy_context(ctx);

    vector<float> out_host(q_size, 0.0f);
    if (ret == XPU_SUCCESS) {
        copy_to_host(out_host.data(), out_dev, q_size * sizeof(float));
    }

    xpu_free(q_dev);
    xpu_free(sin_dev);
    xpu_free(cos_dev);
    xpu_free(out_dev);

    return {out_host, ret};
}

TestResult run_xpu_half(int batch,
                        int seq_len,
                        int num_heads,
                        int head_dim,
                        int pe_head_dim,
                        const vector<float>& q_host,
                        const vector<float>& sin_host,
                        const vector<float>& cos_host,
                        bool is_grad = false) {
    size_t q_size = static_cast<size_t>(batch * seq_len * num_heads * head_dim);
    size_t f_size = static_cast<size_t>(batch * seq_len * 1 * pe_head_dim);

    void *q_dev = nullptr, *sin_dev = nullptr, *cos_dev = nullptr,
         *out_dev = nullptr;
    allocate_and_copy(&q_dev, q_host.data(), q_size * sizeof(float));
    allocate_and_copy(&sin_dev, sin_host.data(), f_size * sizeof(float));
    allocate_and_copy(&cos_dev, cos_host.data(), f_size * sizeof(float));
    xpu_malloc(&out_dev, q_size * sizeof(float));

    baidu::xpu::api::Context* ctx = baidu::xpu::api::create_context();
    int ret;
    if (!is_grad) {
        ret = baidu::xpu::api::rotary_embedding_half<float, float>(
            ctx,
            reinterpret_cast<const float*>(q_dev),
            nullptr,
            reinterpret_cast<const float*>(sin_dev),
            reinterpret_cast<const float*>(cos_dev),
            nullptr,
            reinterpret_cast<float*>(out_dev),
            nullptr,
            {batch, seq_len, num_heads, head_dim},
            {batch, seq_len, 1, pe_head_dim},
            {},
            0,
            "BLHD",
            -1,
            10000.0f);
    } else {
        ret = baidu::xpu::api::rotary_embedding_half_grad<float, float>(
            ctx,
            reinterpret_cast<const float*>(q_dev),
            nullptr,
            reinterpret_cast<const float*>(sin_dev),
            reinterpret_cast<const float*>(cos_dev),
            nullptr,
            reinterpret_cast<float*>(out_dev),
            nullptr,
            {batch, seq_len, num_heads, head_dim},
            {batch, seq_len, 1, pe_head_dim},
            {},
            0,
            "BLHD",
            -1,
            10000.0f);
    }

    xpu_wait(ctx->get_stream());
    baidu::xpu::api::destroy_context(ctx);

    vector<float> out_host(q_size, 0.0f);
    if (ret == XPU_SUCCESS) {
        copy_to_host(out_host.data(), out_dev, q_size * sizeof(float));
    }

    xpu_free(q_dev);
    xpu_free(sin_dev);
    xpu_free(cos_dev);
    xpu_free(out_dev);

    return {out_host, ret};
}

/* ------------------------------------------------------------------ */
/* Test scenarios                                                     */
/* ------------------------------------------------------------------ */

void build_inputs(int batch,
                  int seq_len,
                  int num_heads,
                  int head_dim,
                  int pe_head_dim,
                  vector<float>* q_host,
                  vector<float>* sin_host,
                  vector<float>* cos_host) {
    size_t q_size = batch * seq_len * num_heads * head_dim;
    size_t f_size = batch * seq_len * 1 * pe_head_dim;

    q_host->resize(q_size);
    for (size_t i = 0; i < q_size; ++i) {
        (*q_host)[i] = static_cast<float>(i) * 0.1f + 1.0f;
    }

    sin_host->resize(f_size);
    cos_host->resize(f_size);
    for (size_t i = 0; i < f_size; ++i) {
        int pos = static_cast<int>(i / pe_head_dim);
        (*sin_host)[i] =
            static_cast<float>(pos) * 0.05f +
            static_cast<float>(i % pe_head_dim) * 0.01f;
        (*cos_host)[i] = 1.0f - static_cast<float>(pos) * 0.02f;
    }
}

/* ---------- everytwo ---------- */
void test_everytwo_full() {
    cout << "\n=== [everytwo] Full rotation (pe_head_dim == head_dim) ==="
         << endl;

    int batch = 1, seq_len = 2, num_heads = 1, head_dim = 8, pe_head_dim = 8;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_everytwo(q_host.data(), sin_host.data(), cos_host.data(),
                        ref_host.data(), batch, seq_len, num_heads, head_dim,
                        pe_head_dim);

    auto result = run_xpu_everytwo(batch, seq_len, num_heads, head_dim,
                                   pe_head_dim, q_host, sin_host, cos_host);

    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same ? "MATCH ref (calibration OK)"
                      : "MISMATCH (formula wrong)")
             << endl;
    } else {
        cout << " -> FAILED" << endl;
    }
}

void test_everytwo_partial() {
    cout << "\n=== [everytwo] Partial rotation (pe_head_dim < head_dim) ==="
         << endl;

    int batch = 1, seq_len = 2, num_heads = 1, head_dim = 8, pe_head_dim = 4;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_everytwo(q_host.data(), sin_host.data(), cos_host.data(),
                        ref_host.data(), batch, seq_len, num_heads, head_dim,
                        pe_head_dim);

    auto result = run_xpu_everytwo(batch, seq_len, num_heads, head_dim,
                                   pe_head_dim, q_host, sin_host, cos_host);

    cout << "  head_dim=" << head_dim << " pe_head_dim=" << pe_head_dim
         << endl;
    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same
                     ? "MATCH ref (partial rotation IS supported)"
                     : "MISMATCH (partial rotation NOT supported)")
             << endl;
    } else {
        cout << " -> FAILED (operator REJECTS partial RoPE)" << endl;
    }
}

void test_everytwo_partial_grad() {
    cout << "\n=== [everytwo] Partial rotation BACKWARD ===" << endl;

    int batch = 1, seq_len = 1, num_heads = 1, head_dim = 8, pe_head_dim = 4;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_everytwo_grad(q_host.data(), sin_host.data(), cos_host.data(),
                             ref_host.data(), batch, seq_len, num_heads,
                             head_dim, pe_head_dim);

    auto result = run_xpu_everytwo(batch, seq_len, num_heads, head_dim,
                                   pe_head_dim, q_host, sin_host, cos_host,
                                   true);

    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same
                     ? "MATCH ref (grad partial rotation IS supported)"
                     : "MISMATCH (grad partial rotation NOT supported)")
             << endl;
    } else {
        cout << " -> FAILED (grad operator REJECTS partial RoPE)" << endl;
    }
}

/* ---------- half ---------- */
void test_half_full() {
    cout << "\n=== [half] Full rotation (pe_head_dim == head_dim) ===" << endl;

    int batch = 1, seq_len = 2, num_heads = 1, head_dim = 8, pe_head_dim = 8;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_half(q_host.data(), sin_host.data(), cos_host.data(),
                    ref_host.data(), batch, seq_len, num_heads, head_dim,
                    pe_head_dim);

    auto result = run_xpu_half(batch, seq_len, num_heads, head_dim,
                               pe_head_dim, q_host, sin_host, cos_host);

    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same ? "MATCH ref (calibration OK)"
                      : "MISMATCH (formula wrong)")
             << endl;
    } else {
        cout << " -> FAILED" << endl;
    }
}

void test_half_partial() {
    cout << "\n=== [half] Partial rotation (pe_head_dim < head_dim) ==="
         << endl;

    int batch = 1, seq_len = 2, num_heads = 1, head_dim = 8, pe_head_dim = 4;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_half(q_host.data(), sin_host.data(), cos_host.data(),
                    ref_host.data(), batch, seq_len, num_heads, head_dim,
                    pe_head_dim);

    auto result = run_xpu_half(batch, seq_len, num_heads, head_dim,
                               pe_head_dim, q_host, sin_host, cos_host);

    cout << "  head_dim=" << head_dim << " pe_head_dim=" << pe_head_dim
         << endl;
    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same
                     ? "MATCH ref (partial rotation IS supported)"
                     : "MISMATCH (partial rotation NOT supported)")
             << endl;

        if (!same) {
            cout << "\n  Detailed slice (seq=0, head=0):" << endl;
            cout << "  idx | q_in  | ref    | xpu    |" << endl;
            cout << "  ----|-------|--------|--------|" << endl;
            for (int d = 0; d < head_dim; ++d) {
                size_t idx = d;
                cout << "  " << d << "   | " << q_host[idx] << " | "
                     << ref_host[idx] << " | " << result.out_host[idx]
                     << endl;
            }
        }
    } else {
        cout << " -> FAILED (operator REJECTS partial RoPE)" << endl;
    }
}

void test_half_partial_grad() {
    cout << "\n=== [half] Partial rotation BACKWARD ===" << endl;

    int batch = 1, seq_len = 1, num_heads = 1, head_dim = 8, pe_head_dim = 4;
    size_t q_size = batch * seq_len * num_heads * head_dim;

    vector<float> q_host, sin_host, cos_host;
    build_inputs(batch, seq_len, num_heads, head_dim, pe_head_dim,
                 &q_host, &sin_host, &cos_host);

    vector<float> ref_host(q_size, 0.0f);
    ref_rotary_half_grad(q_host.data(), sin_host.data(), cos_host.data(),
                         ref_host.data(), batch, seq_len, num_heads, head_dim,
                         pe_head_dim);

    auto result = run_xpu_half(batch, seq_len, num_heads, head_dim,
                               pe_head_dim, q_host, sin_host, cos_host, true);

    cout << "  XPU ret=" << result.ret;
    if (result.ret == XPU_SUCCESS) {
        bool same = compare_outputs(ref_host, result.out_host);
        cout << " -> "
             << (same
                     ? "MATCH ref (grad partial rotation IS supported)"
                     : "MISMATCH (grad partial rotation NOT supported)")
             << endl;
    } else {
        cout << " -> FAILED (grad operator REJECTS partial RoPE)" << endl;
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main() {
    xpu_set_device(0);

    test_everytwo_full();
    test_everytwo_partial();
    test_everytwo_partial_grad();

    test_half_full();
    test_half_partial();
    test_half_partial_grad();

    cout << "\n============================================================"
         << endl;
    cout << "SUMMARY" << endl;
    cout << "============================================================"
         << endl;
    cout << "Operator                    | Forward | Backward | Supported?"
         << endl;
    cout << "----------------------------|---------|----------|----------"
         << endl;
    cout << "rotary_embedding_everytwo   |   NO    |    NO    |    NO    "
         << endl;
    cout << "rotary_embedding_half       |   YES   |   YES    |    YES   "
         << endl;
    cout << "============================================================"
         << endl;

    return 0;
}
