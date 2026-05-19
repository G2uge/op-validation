/*
 * C++ accuracy test for Paddle phi::fusion::FusedSwigluScaleKernel
 * (XPU backend).
 *
 * The operator performs:
 *   1. swiglu(x)  : split last dim into two halves,
 *                   out = SiLU(x1) * x2
 *   2. scale      : out = swiglu_out * scale (broadcast)
 *
 * Both forward and backward are verified against CPU reference.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;
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

float sigmoid(float x) { return 1.0f / (1.0f + std::exp(-x)); }

bool compare_outputs(const vector<float>& ref,
                     const vector<float>& out,
                     float tol = 1e-3f,
                     bool verbose = true,
                     const char* name = "output") {
    if (ref.size() != out.size()) {
        if (verbose)
            cout << "  [" << name << "] size mismatch: " << ref.size()
                 << " vs " << out.size() << endl;
        return false;
    }
    bool ok = true;
    float max_diff = 0.0f;
    size_t first_err = 0;
    for (size_t i = 0; i < ref.size(); ++i) {
        float diff = std::fabs(ref[i] - out[i]);
        if (diff > max_diff) max_diff = diff;
        if (diff > tol) {
            if (ok && verbose) first_err = i;
            ok = false;
        }
    }
    if (verbose) {
        if (ok) {
            cout << "  [" << name << "] PASS  max_diff=" << max_diff << endl;
        } else {
            cout << "  [" << name << "] FAIL  first_err_idx=" << first_err
                 << " ref=" << ref[first_err] << " out=" << out[first_err]
                 << " max_diff=" << max_diff << endl;
        }
    }
    return ok;
}

/* ------------------------------------------------------------------ */
/* CPU Reference: swiglu + scale                                      */
/* ------------------------------------------------------------------ */
/*
 * x_shape : e.g. [M, N] where N is even
 * axis    : last axis (N)
 * turn    : true  -> x1 = x[..., :N/2], x2 = x[..., N/2:]
 *           false -> x1 = x[..., N/2:], x2 = x[..., :N/2]
 *
 * scale_shape must be broadcastable to out_shape.
 */
void ref_fused_swiglu_scale(const float* x,
                            const float* scale,
                            float* out,
                            const vector<int64_t>& x_shape,
                            const vector<int64_t>& scale_shape,
                            bool turn) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;

    // compute total elements of output
    int64_t out_numel = 1;
    for (int64_t i = 0; i < ndim - 1; ++i) out_numel *= x_shape[i];
    out_numel *= half;

    // broadcasted scale shape (pad 1s)
    vector<int64_t> b_scale_shape = scale_shape;
    while ((int64_t)b_scale_shape.size() < ndim) b_scale_shape.push_back(1);

    for (int64_t idx = 0; idx < out_numel; ++idx) {
        // linear index -> multi-dim index for output
        int64_t tmp = idx;
        vector<int64_t> out_idx(ndim, 0);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (d == ndim - 1) ? half : x_shape[d];
            out_idx[d] = tmp % dim_size;
            tmp /= dim_size;
        }

        // corresponding x index
        vector<int64_t> x_idx = out_idx;
        int64_t x_offset = 0;
        int64_t stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            x_offset += x_idx[d] * stride;
            stride *= x_shape[d];
        }

        // x1 and x2 positions
        int64_t x1_pos, x2_pos;
        if (turn) {
            x1_pos = x_offset;                    // first half
            x2_pos = x_offset + half;             // second half
        } else {
            x1_pos = x_offset + half;             // second half
            x2_pos = x_offset;                    // first half
        }

        float v1 = x[x1_pos];
        float v2 = x[x2_pos];
        float silu = v1 * sigmoid(v1);
        float swiglu_val = silu * v2;

        // broadcast scale
        int64_t scale_stride = 1;
        int64_t scale_offset = 0;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (d == ndim - 1) ? half : x_shape[d];
            int64_t s_idx = (b_scale_shape[d] == 1) ? 0 : out_idx[d];
            scale_offset += s_idx * scale_stride;
            scale_stride *= b_scale_shape[d];
        }
        float s = scale[scale_offset];

        out[idx] = swiglu_val * s;
    }
}

/* ------------------------------------------------------------------ */
/* CPU Reference: swiglu_grad                                         */
/* ------------------------------------------------------------------ */
/*
 * swiglu(x) = silu(x1) * x2
 * dx1 = dout * x2 * (sigmoid(x1) + x1 * sigmoid(x1) * (1 - sigmoid(x1)))
 *     = dout * x2 * sigmoid(x1) * (1 + x1 * (1 - sigmoid(x1)))
 * dx2 = dout * silu(x1)
 */
void ref_swiglu_grad(const float* x,
                     const float* dout,
                     float* dx,
                     const vector<int64_t>& x_shape,
                     bool turn) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;
    int64_t numel = 1;
    for (auto d : x_shape) numel *= d;

    // zero dx
    memset(dx, 0, sizeof(float) * numel);

    int64_t batch = numel / last_dim;
    for (int64_t b = 0; b < batch; ++b) {
        for (int64_t h = 0; h < half; ++h) {
            int64_t x1_pos, x2_pos;
            if (turn) {
                x1_pos = b * last_dim + h;
                x2_pos = b * last_dim + h + half;
            } else {
                x1_pos = b * last_dim + h + half;
                x2_pos = b * last_dim + h;
            }
            float v1 = x[x1_pos];
            float v2 = x[x2_pos];
            float s = sigmoid(v1);
            float silu = v1 * s;
            float d = dout[b * half + h];

            float ds = s * (1.0f + v1 * (1.0f - s));
            dx[x1_pos] = d * v2 * ds;
            dx[x2_pos] = d * silu;
        }
    }
}

/* ------------------------------------------------------------------ */
/* XPU Forward Test                                                   */
/* ------------------------------------------------------------------ */
bool test_fused_swiglu_scale_forward(const vector<int64_t>& x_shape,
                                      const vector<int64_t>& scale_shape,
                                      bool turn = true) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;

    int64_t x_numel = 1;
    for (auto d : x_shape) x_numel *= d;
    int64_t out_numel = x_numel / 2;
    int64_t scale_numel = 1;
    for (auto d : scale_shape) scale_numel *= d;

    // prepare host data
    vector<float> x(x_numel);
    vector<float> scale(scale_numel);
    for (int64_t i = 0; i < x_numel; ++i) x[i] = (i % 7 - 3) * 0.3f + 0.1f;
    for (int64_t i = 0; i < scale_numel; ++i) scale[i] = (i % 5 + 1) * 0.5f;

    vector<float> ref(out_numel);
    ref_fused_swiglu_scale(x.data(), scale.data(), ref.data(), x_shape, scale_shape, turn);

    // allocate device memory
    void *d_x = nullptr, *d_scale = nullptr, *d_out = nullptr;
    allocate_and_copy(&d_x, x.data(), sizeof(float) * x_numel);
    allocate_and_copy(&d_scale, scale.data(), sizeof(float) * scale_numel);
    xpu_malloc(&d_out, sizeof(float) * out_numel);

    xpu::Context* ctx = xpu::create_context();
    int ret = 0;

    // 1. swiglu
    vector<int64_t> swiglu_shape = x_shape;
    ret = xpu::swiglu<float>(ctx,
                             static_cast<const float*>(d_x),
                             nullptr,
                             static_cast<float*>(d_out),
                             swiglu_shape,
                             ndim - 1,
                             turn);
    if (ret != 0) {
        cout << "  xpu::swiglu failed, ret=" << ret << endl;
        xpu_free(d_x); xpu_free(d_scale); xpu_free(d_out);
        xpu::destroy_context(ctx);
        return false;
    }

    // 2. broadcast_mul (inplace on out)
    vector<int64_t> out_shape = x_shape;
    out_shape[ndim - 1] = half;
    vector<int64_t> b_scale_shape = scale_shape;
    while ((int64_t)b_scale_shape.size() < ndim) b_scale_shape.push_back(1);

    // allocate temp out for mul
    void* d_mul_out = nullptr;
    xpu_malloc(&d_mul_out, sizeof(float) * out_numel);
    ret = xpu::broadcast_mul<float>(ctx,
                                    static_cast<const float*>(d_out),
                                    static_cast<const float*>(d_scale),
                                    static_cast<float*>(d_mul_out),
                                    out_shape,
                                    b_scale_shape);
    if (ret != 0) {
        cout << "  xpu::broadcast_mul failed, ret=" << ret << endl;
        xpu_free(d_x); xpu_free(d_scale); xpu_free(d_out); xpu_free(d_mul_out);
        xpu::destroy_context(ctx);
        return false;
    }

    vector<float> out(out_numel);
    copy_to_host(out.data(), d_mul_out, sizeof(float) * out_numel);

    xpu_free(d_x); xpu_free(d_scale); xpu_free(d_out); xpu_free(d_mul_out);
    xpu::destroy_context(ctx);

    return compare_outputs(ref, out, 1e-3f, true, "forward");
}

/* ------------------------------------------------------------------ */
/* XPU Backward Test                                                  */
/* ------------------------------------------------------------------ */
bool test_fused_swiglu_scale_grad(const vector<int64_t>& x_shape,
                                   const vector<int64_t>& scale_shape,
                                   bool turn = true) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;

    int64_t x_numel = 1;
    for (auto d : x_shape) x_numel *= d;
    int64_t out_numel = x_numel / 2;
    int64_t scale_numel = 1;
    for (auto d : scale_shape) scale_numel *= d;

    // prepare host data
    vector<float> x(x_numel);
    vector<float> scale(scale_numel);
    vector<float> dout(out_numel);
    for (int64_t i = 0; i < x_numel; ++i) x[i] = (i % 7 - 3) * 0.3f + 0.1f;
    for (int64_t i = 0; i < scale_numel; ++i) scale[i] = (i % 5 + 1) * 0.5f;
    for (int64_t i = 0; i < out_numel; ++i) dout[i] = (i % 3 + 1) * 0.2f;

    // CPU ref: swiglu(x) without scale
    vector<float> ones_scale(scale_numel, 1.0f);
    vector<float> ref_swiglu_raw(out_numel);
    ref_fused_swiglu_scale(x.data(), ones_scale.data(), ref_swiglu_raw.data(), x_shape, scale_shape, turn);

    // d_u = dout * scale
    vector<int64_t> out_shape = x_shape;
    out_shape[ndim - 1] = half;
    vector<int64_t> b_scale_shape = scale_shape;
    while ((int64_t)b_scale_shape.size() < ndim) b_scale_shape.push_back(1);

    vector<float> d_u(out_numel);
    for (int64_t idx = 0; idx < out_numel; ++idx) {
        int64_t tmp = idx;
        vector<int64_t> out_idx(ndim, 0);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (d == ndim - 1) ? half : x_shape[d];
            out_idx[d] = tmp % dim_size;
            tmp /= dim_size;
        }
        int64_t scale_offset = 0;
        int64_t scale_stride = 1;
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (d == ndim - 1) ? half : x_shape[d];
            int64_t s_idx = (b_scale_shape[d] == 1) ? 0 : out_idx[d];
            scale_offset += s_idx * scale_stride;
            scale_stride *= b_scale_shape[d];
        }
        d_u[idx] = dout[idx] * scale[scale_offset];
    }

    vector<float> ref_dx(x_numel);
    ref_swiglu_grad(x.data(), d_u.data(), ref_dx.data(), x_shape, turn);

    // dscale = sum(dout * swiglu_out, axis=-1)
    vector<int64_t> dscale_shape;
    for (int64_t i = 0; i < ndim - 1; ++i) dscale_shape.push_back(x_shape[i]);
    if (dscale_shape.empty()) dscale_shape.push_back(1);
    int64_t dscale_numel = 1;
    for (auto d : dscale_shape) dscale_numel *= d;

    vector<float> ref_dscale(dscale_numel, 0.0f);
    for (int64_t idx = 0; idx < out_numel; ++idx) {
        int64_t tmp = idx;
        vector<int64_t> out_idx(ndim, 0);
        for (int64_t d = ndim - 1; d >= 0; --d) {
            int64_t dim_size = (d == ndim - 1) ? half : x_shape[d];
            out_idx[d] = tmp % dim_size;
            tmp /= dim_size;
        }
        int64_t dscale_idx = 0;
        int64_t stride = 1;
        for (int64_t d = ndim - 2; d >= 0; --d) {
            dscale_idx += out_idx[d] * stride;
            stride *= x_shape[d];
        }
        ref_dscale[dscale_idx] += dout[idx] * ref_swiglu_raw[idx];
    }

    // XPU
    void *d_x = nullptr, *d_scale = nullptr, *d_dout = nullptr;
    void *d_swiglu_out = nullptr, *d_d_u = nullptr, *d_dx = nullptr;
    void *d_dscale = nullptr, *d_mul_tmp = nullptr;

    allocate_and_copy(&d_x, x.data(), sizeof(float) * x_numel);
    allocate_and_copy(&d_scale, scale.data(), sizeof(float) * scale_numel);
    allocate_and_copy(&d_dout, dout.data(), sizeof(float) * out_numel);
    xpu_malloc(&d_swiglu_out, sizeof(float) * out_numel);
    xpu_malloc(&d_d_u, sizeof(float) * out_numel);
    xpu_malloc(&d_dx, sizeof(float) * x_numel);
    xpu_malloc(&d_dscale, sizeof(float) * dscale_numel);
    xpu_malloc(&d_mul_tmp, sizeof(float) * out_numel);

    xpu::Context* ctx = xpu::create_context();
    int ret = 0;

    // 1. recompute swiglu_out
    ret = xpu::swiglu<float>(ctx,
                             static_cast<const float*>(d_x),
                             nullptr,
                             static_cast<float*>(d_swiglu_out),
                             x_shape,
                             ndim - 1,
                             turn);
    if (ret != 0) {
        cout << "  xpu::swiglu (grad recompute) failed, ret=" << ret << endl;
        goto grad_cleanup;
    }

    // 2. d_u = dout * scale
    ret = xpu::broadcast_mul<float>(ctx,
                                    static_cast<const float*>(d_dout),
                                    static_cast<const float*>(d_scale),
                                    static_cast<float*>(d_d_u),
                                    out_shape,
                                    b_scale_shape);
    if (ret != 0) {
        cout << "  xpu::broadcast_mul (grad) failed, ret=" << ret << endl;
        goto grad_cleanup;
    }

    // 3. swiglu_grad
    ret = xpu::swiglu_grad<float>(ctx,
                                  static_cast<const float*>(d_x),
                                  nullptr,
                                  static_cast<const float*>(d_d_u),
                                  static_cast<float*>(d_dx),
                                  nullptr,
                                  x_shape,
                                  ndim - 1,
                                  turn);
    if (ret != 0) {
        cout << "  xpu::swiglu_grad failed, ret=" << ret << endl;
        goto grad_cleanup;
    }

    // 4. dscale = reduce_sum(dout * swiglu_out, axis=-1)
    ret = xpu::broadcast_mul<float>(ctx,
                                    static_cast<const float*>(d_dout),
                                    static_cast<const float*>(d_swiglu_out),
                                    static_cast<float*>(d_mul_tmp),
                                    out_shape,
                                    out_shape);
    if (ret != 0) {
        cout << "  xpu::broadcast_mul (mul_tmp) failed, ret=" << ret << endl;
        goto grad_cleanup;
    }

    {
        vector<int64_t> rdims = {ndim - 1};
        ret = xpu::reduce_sum<float>(ctx,
                                     static_cast<const float*>(d_mul_tmp),
                                     static_cast<float*>(d_dscale),
                                     out_shape,
                                     rdims);
        if (ret != 0) {
            cout << "  xpu::reduce_sum failed, ret=" << ret << endl;
            goto grad_cleanup;
        }
    }

grad_cleanup:
    {
        vector<float> dx(x_numel);
        vector<float> dscale(dscale_numel);
        copy_to_host(dx.data(), d_dx, sizeof(float) * x_numel);
        copy_to_host(dscale.data(), d_dscale, sizeof(float) * dscale_numel);

        xpu_free(d_x); xpu_free(d_scale); xpu_free(d_dout);
        xpu_free(d_swiglu_out); xpu_free(d_d_u); xpu_free(d_dx);
        xpu_free(d_dscale); xpu_free(d_mul_tmp);
        xpu::destroy_context(ctx);

        if (ret != 0) return false;

        bool ok_dx = compare_outputs(ref_dx, dx, 1e-3f, true, "dx");
        bool ok_dscale = compare_outputs(ref_dscale, dscale, 1e-3f, true, "dscale");
        return ok_dx && ok_dscale;
    }
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main() {
    xpu_set_device(0);
    int total = 0;
    int passed = 0;

    auto run_forward = [&](const vector<int64_t>& x_shape,
                           const vector<int64_t>& scale_shape,
                           const char* desc) {
        total++;
        cout << "[Forward] " << desc << endl;
        if (test_fused_swiglu_scale_forward(x_shape, scale_shape, true))
            passed++;
    };

    auto run_grad = [&](const vector<int64_t>& x_shape,
                        const vector<int64_t>& scale_shape,
                        const char* desc) {
        total++;
        cout << "[Grad] " << desc << endl;
        if (test_fused_swiglu_scale_grad(x_shape, scale_shape, true))
            passed++;
    };

    // Forward tests
    run_forward({4, 8}, {4, 1}, "2D x=[4,8] scale=[4,1]");
    run_forward({4, 8}, {1, 4}, "2D x=[4,8] scale=[1,4]");
    run_forward({4, 8}, {4},    "2D x=[4,8] scale=[4]");
    run_forward({2, 3, 16}, {2, 3, 8}, "3D x=[2,3,16] scale=[2,3,8]");
    run_forward({2, 3, 16}, {1, 1, 8}, "3D x=[2,3,16] scale=[1,1,8]");
    run_forward({2, 3, 4, 16}, {2, 1, 4, 8}, "4D x=[2,3,4,16] scale=[2,1,4,8]");
    run_forward({1, 128}, {1, 64}, "2D x=[1,128] scale=[1,64]");
    run_forward({1024, 256}, {1024, 1}, "2D x=[1024,256] scale=[1024,1]");

    // Grad tests
    run_grad({4, 8}, {4, 1}, "2D x=[4,8] scale=[4,1]");
    run_grad({4, 8}, {1, 4}, "2D x=[4,8] scale=[1,4]");
    run_grad({2, 3, 16}, {2, 3, 8}, "3D x=[2,3,16] scale=[2,3,8]");
    run_grad({2, 3, 4, 16}, {2, 1, 4, 8}, "4D x=[2,3,4,16] scale=[2,1,4,8]");
    run_grad({1024, 256}, {1024, 1}, "2D x=[1024,256] scale=[1024,1]");

    cout << "\n========================================" << endl;
    cout << "Total tests: " << total << endl;
    cout << "Passed:      " << passed << endl;
    cout << "Failed:      " << (total - passed) << endl;
    cout << "========================================" << endl;

    return (passed == total) ? 0 : 1;
}
