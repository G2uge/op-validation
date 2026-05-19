/*
 * C++ performance benchmark for XDNN fused swiglu + scale.
 * Measures forward and backward latency, averaging over multiple warm-up
 * and benchmark iterations.
 */

#include <cmath>
#include <cstdlib>
#include <cstring>
#include <iostream>
#include <vector>
#include <chrono>
#include "xpu/xdnn.h"

namespace xpu = baidu::xpu::api;
using namespace std;

static double elapsed_ms(const chrono::high_resolution_clock::time_point& start,
                         const chrono::high_resolution_clock::time_point& end) {
    return chrono::duration<double, milli>(end - start).count();
}

int allocate_and_copy(void** dev_ptr, const void* host_ptr, size_t size) {
    int ret = xpu_malloc(dev_ptr, size);
    if (ret != XPU_SUCCESS) return ret;
    return xpu_memcpy(*dev_ptr, host_ptr, size, XPU_HOST_TO_DEVICE);
}

/* ------------------------------------------------------------------ */
/* Forward benchmark                                                  */
/* ------------------------------------------------------------------ */
double benchmark_forward(const vector<int64_t>& x_shape,
                         const vector<int64_t>& scale_shape,
                         int warmup,
                         int iters,
                         bool turn = true) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;

    int64_t x_numel = 1;
    for (auto d : x_shape) x_numel *= d;
    int64_t out_numel = x_numel / 2;
    int64_t scale_numel = 1;
    for (auto d : scale_shape) scale_numel *= d;

    vector<float> x(x_numel);
    vector<float> scale(scale_numel);
    for (int64_t i = 0; i < x_numel; ++i) x[i] = (i % 7 - 3) * 0.3f + 0.1f;
    for (int64_t i = 0; i < scale_numel; ++i) scale[i] = (i % 5 + 1) * 0.5f;

    void *d_x = nullptr, *d_scale = nullptr, *d_out = nullptr, *d_mul_out = nullptr;
    allocate_and_copy(&d_x, x.data(), sizeof(float) * x_numel);
    allocate_and_copy(&d_scale, scale.data(), sizeof(float) * scale_numel);
    xpu_malloc(&d_out, sizeof(float) * out_numel);
    xpu_malloc(&d_mul_out, sizeof(float) * out_numel);

    xpu::Context* ctx = xpu::create_context();
    vector<int64_t> out_shape = x_shape;
    out_shape[ndim - 1] = half;
    vector<int64_t> b_scale_shape = scale_shape;
    while ((int64_t)b_scale_shape.size() < ndim) b_scale_shape.push_back(1);

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        xpu::swiglu<float>(ctx,
                          static_cast<const float*>(d_x),
                          nullptr,
                          static_cast<float*>(d_out),
                          x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_out),
                                 static_cast<const float*>(d_scale),
                                 static_cast<float*>(d_mul_out),
                                 out_shape, b_scale_shape);
    }
    xpu_wait(ctx->get_stream());

    // benchmark
    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        xpu::swiglu<float>(ctx,
                          static_cast<const float*>(d_x),
                          nullptr,
                          static_cast<float*>(d_out),
                          x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_out),
                                 static_cast<const float*>(d_scale),
                                 static_cast<float*>(d_mul_out),
                                 out_shape, b_scale_shape);
    }
    xpu_wait(ctx->get_stream());
    auto t1 = chrono::high_resolution_clock::now();

    xpu_free(d_x); xpu_free(d_scale); xpu_free(d_out); xpu_free(d_mul_out);
    xpu::destroy_context(ctx);

    return elapsed_ms(t0, t1) / iters;
}

/* ------------------------------------------------------------------ */
/* Backward benchmark                                                 */
/* ------------------------------------------------------------------ */
double benchmark_grad(const vector<int64_t>& x_shape,
                       const vector<int64_t>& scale_shape,
                       int warmup,
                       int iters,
                       bool turn = true) {
    int64_t ndim = x_shape.size();
    int64_t last_dim = x_shape[ndim - 1];
    int64_t half = last_dim / 2;

    int64_t x_numel = 1;
    for (auto d : x_shape) x_numel *= d;
    int64_t out_numel = x_numel / 2;
    int64_t scale_numel = 1;
    for (auto d : scale_shape) scale_numel *= d;

    vector<int64_t> dscale_shape;
    for (int64_t i = 0; i < ndim - 1; ++i) dscale_shape.push_back(x_shape[i]);
    if (dscale_shape.empty()) dscale_shape.push_back(1);
    int64_t dscale_numel = 1;
    for (auto d : dscale_shape) dscale_numel *= d;

    vector<float> x(x_numel), scale(scale_numel), dout(out_numel);
    for (int64_t i = 0; i < x_numel; ++i) x[i] = (i % 7 - 3) * 0.3f + 0.1f;
    for (int64_t i = 0; i < scale_numel; ++i) scale[i] = (i % 5 + 1) * 0.5f;
    for (int64_t i = 0; i < out_numel; ++i) dout[i] = (i % 3 + 1) * 0.2f;

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
    vector<int64_t> out_shape = x_shape;
    out_shape[ndim - 1] = half;
    vector<int64_t> b_scale_shape = scale_shape;
    while ((int64_t)b_scale_shape.size() < ndim) b_scale_shape.push_back(1);
    vector<int64_t> rdims = {ndim - 1};

    // warm-up
    for (int i = 0; i < warmup; ++i) {
        xpu::swiglu<float>(ctx, static_cast<const float*>(d_x), nullptr,
                          static_cast<float*>(d_swiglu_out), x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_dout),
                                 static_cast<const float*>(d_scale),
                                 static_cast<float*>(d_d_u),
                                 out_shape, b_scale_shape);
        xpu::swiglu_grad<float>(ctx,
                               static_cast<const float*>(d_x), nullptr,
                               static_cast<const float*>(d_d_u),
                               static_cast<float*>(d_dx), nullptr,
                               x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_dout),
                                 static_cast<const float*>(d_swiglu_out),
                                 static_cast<float*>(d_mul_tmp),
                                 out_shape, out_shape);
        xpu::reduce_sum<float>(ctx,
                              static_cast<const float*>(d_mul_tmp),
                              static_cast<float*>(d_dscale),
                              out_shape, rdims);
    }
    xpu_wait(ctx->get_stream());

    auto t0 = chrono::high_resolution_clock::now();
    for (int i = 0; i < iters; ++i) {
        xpu::swiglu<float>(ctx, static_cast<const float*>(d_x), nullptr,
                          static_cast<float*>(d_swiglu_out), x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_dout),
                                 static_cast<const float*>(d_scale),
                                 static_cast<float*>(d_d_u),
                                 out_shape, b_scale_shape);
        xpu::swiglu_grad<float>(ctx,
                               static_cast<const float*>(d_x), nullptr,
                               static_cast<const float*>(d_d_u),
                               static_cast<float*>(d_dx), nullptr,
                               x_shape, ndim - 1, turn);
        xpu::broadcast_mul<float>(ctx,
                                 static_cast<const float*>(d_dout),
                                 static_cast<const float*>(d_swiglu_out),
                                 static_cast<float*>(d_mul_tmp),
                                 out_shape, out_shape);
        xpu::reduce_sum<float>(ctx,
                              static_cast<const float*>(d_mul_tmp),
                              static_cast<float*>(d_dscale),
                              out_shape, rdims);
    }
    xpu_wait(ctx->get_stream());
    auto t1 = chrono::high_resolution_clock::now();

    xpu_free(d_x); xpu_free(d_scale); xpu_free(d_dout);
    xpu_free(d_swiglu_out); xpu_free(d_d_u); xpu_free(d_dx);
    xpu_free(d_dscale); xpu_free(d_mul_tmp);
    xpu::destroy_context(ctx);

    return elapsed_ms(t0, t1) / iters;
}

/* ------------------------------------------------------------------ */
/* Main                                                               */
/* ------------------------------------------------------------------ */
int main() {
    xpu_set_device(0);
    const int warmup = 10;
    const int iters = 100;

    cout << "{\"benchmarks\":[\n";
    bool first = true;

    auto run = [&](const char* name,
                   const vector<int64_t>& x_shape,
                   const vector<int64_t>& scale_shape) {
        if (!first) cout << ",\n";
        first = false;

        double fwd_ms = benchmark_forward(x_shape, scale_shape, warmup, iters, true);
        double bwd_ms = benchmark_grad(x_shape, scale_shape, warmup, iters, true);

        cout << "  {\"name\":\"" << name << "\","
             << "\"x_shape\":\"";
        for (size_t i = 0; i < x_shape.size(); ++i) {
            if (i) cout << "x";
            cout << x_shape[i];
        }
        cout << "\","
             << "\"scale_shape\":\"";
        for (size_t i = 0; i < scale_shape.size(); ++i) {
            if (i) cout << "x";
            cout << scale_shape[i];
        }
        cout << "\","
             << "\"forward_ms\":" << fwd_ms << ","
             << "\"backward_ms\":" << bwd_ms << "}";
    };

    run("small_2D", {4, 8}, {4, 1});
    run("medium_2D", {1024, 256}, {1024, 1});
    run("large_2D", {4096, 512}, {4096, 1});
    run("small_3D", {2, 3, 16}, {1, 1, 8});
    run("medium_3D", {8, 128, 512}, {8, 128, 256});
    run("large_3D", {32, 512, 1024}, {32, 512, 512});

    cout << "\n]}\n";
    return 0;
}
