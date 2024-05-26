#include <cuda_runtime.h>
#include "solver_cuda.h"

__device__ __host__ inline int coord(int w, int h, int width) {
    return ((h * width) + w);
}

Solver_cuda::Solver_cuda(int w, int h, int iters, float relax_factor, float dt) {
    this->width = w + 2; // +2 for border
    this->height = h + 2;
    this->iters = iters;
    this->relax_factor = relax_factor;
    this->dt = dt;
    // allocate memory on GPU (device)
    cudaMalloc(&grid_s, width * height * sizeof(uint8_t));
    cudaMalloc(&grid_u, width * height * sizeof(float));
    cudaMalloc(&tmp_u, width * height * sizeof(float));
    cudaMalloc(&grid_v, width * height * sizeof(float));
    cudaMalloc(&tmp_v, width * height * sizeof(float));
    cudaMalloc(&grid_m, width * height * sizeof(float));
    cudaMalloc(&tmp_m, width * height * sizeof(float));
    cudaMalloc(&frame_data, w * h * sizeof(float));
    // allocate memory on CPU (host)
    this->frame_data_host = new uint8_t[w * h];
}

Solver_cuda::~Solver_cuda() {
    cudaFree(grid_s);
    cudaFree(grid_u);
    cudaFree(tmp_u);
    cudaFree(grid_v);
    cudaFree(tmp_v);
    cudaFree(grid_m);
    cudaFree(tmp_m);
    delete[] frame_data_host;
}

__global__ void iterate_compression_kernel(uint8_t *grid_s, float *grid_u, float *grid_v, int width, int height, float relax_factor, int iteration) {
    int w = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int h = blockIdx.y * blockDim.y + threadIdx.y + 1;
    // check bounds
    if (w >= width - 1 || h >= height - 1) return;
    // check if we are on correct cell
    if ((w + h) % 2 == iteration % 2) return;

    if (grid_s[h * width + w] == 0) return;

    int s_up = grid_s[(h-1) * width + w];
    int s_down = grid_s[(h+1) * width + w];
    int s_left = grid_s[h * width + w - 1];
    int s_right = grid_s[h * width + w + 1];
    int s = s_up + s_down + s_left + s_right;

    float div = - grid_v[h * width + w] + grid_v[(h+1) * width + w] - grid_u[h * width + w] + grid_u[h * width + w + 1];
    float p = div / s;
    p *= relax_factor;

    grid_v[h * width + w] += s_up * p;
    grid_v[(h+1) * width + w] -= s_down * p;
    grid_u[h * width + w] += s_left * p;
    grid_u[h * width + w + 1] -= s_right * p;
}

void Solver_cuda::iterate_compression() {
    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    // Run the iterations with CUDA
    for (int i = 0; i < iters * 2; ++i) {
        iterate_compression_kernel<<<gridDim, blockDim>>>(grid_s, grid_u, grid_v, width, height, relax_factor, i);
        cudaDeviceSynchronize(); // Ensure all threads complete before the next iteration
    }
}

__global__ void advect_velocities_kernel(uint8_t *grid_s, float *grid_u, float *grid_v, float *tmp_u, float *tmp_v, int width, int height, float dt) {
    int w = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int h = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (w >= width - 1 || h >= height - 1) return;
    // if there is collision object
    if (grid_s[coord(w, h, width)] == 0) return;
    tmp_v[coord(w, h, width)] = grid_v[coord(w, h, width)];
    tmp_u[coord(w, h, width)] = grid_u[coord(w, h, width)];
    if (grid_s[coord(w, h-1, width)] != 0) {
        float v = grid_v[coord(w, h, width)];
        // get u part of the vector by avareging 4 surrounding 'u's
        float u = (grid_u[coord(w, h-1, width)] + grid_u[coord(w+1, h-1, width)] + grid_u[coord(w, h, width)] + grid_u[coord(w+1, h, width)]) / 4;
        // backtrack based on timestep
        float x = (float) w - (u * dt);
        float y = (float) h - (v * dt);
        // correct if backtracked coordinates are out of bounds
        x = max(min(x, (float) (width-1)), (float) 1.0);
        y = max(min(y, (float) (height-1)), (float) 1.0);
        if (grid_s[coord((int) x, (int) y, width)] != 0) {
            int idx = (int) x;
            int idy = (int) y;
            float ratio_x = x - idx;
            float ratio_y = y - idy;
            float tmp1 = (grid_v[coord(idx, idy, width)] * (1.0 - ratio_x)) + (grid_v[coord(idx+1, idy, width)] * ratio_x);
            float tmp2 = (grid_v[coord(idx, idy+1, width)] * (1.0 - ratio_x)) + (grid_v[coord(idx+1, idy+1, width)] * ratio_x);
            float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
            tmp_v[coord(w, h, width)] = tmp;
        }
    }
    // if there is not obstacle leftwards
    if (grid_s[coord(w-1, h, width)] != 0) {
        float u = grid_u[coord(w, h, width)];
        // get v part of the vector by avareging 4 surrounding 'v's
        float v = (grid_v[coord(w-1, h, width)] + grid_v[coord(w, h, width)] + grid_v[coord(w-1, h+1, width)] + grid_v[coord(w, h+1, width)]) / 4;
        // backtrack based on timestep
        float x = (float) w - (u * dt);
        float y = (float) h - (v * dt);
        // correct if backtracked coordinates are out of bounds
        x = max(min(x, (float) (width-1)), (float) 1.0);
        y = max(min(y, (float) (height-1)), (float) 1.0);
        if (grid_s[coord((int) x, (int) y, width)] != 0) {
            int idx = (int) x;
            int idy = (int) y;
            float ratio_x = x - idx;
            float ratio_y = y - idy;
            float tmp1 = (grid_u[coord(idx, idy, width)] * (1.0 - ratio_x)) + (grid_u[coord(idx+1, idy, width)] * ratio_x);
            float tmp2 = (grid_u[coord(idx, idy+1, width)] * (1.0 - ratio_x)) + (grid_u[coord(idx+1, idy+1, width)] * ratio_x);
            float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
            tmp_u[coord(w, h, width)] = tmp;
        }
    }
}

void Solver_cuda::advect_velocities() {
    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    advect_velocities_kernel<<<gridDim, blockDim>>>(grid_s, grid_u, grid_v, tmp_u, tmp_v, width, height, dt);
    cudaDeviceSynchronize(); // Ensure all threads complete
    // swap the tmp and main grids
    float *tmp_ptr = grid_v;
    grid_v = tmp_v;
    tmp_v = tmp_ptr;
    tmp_ptr = grid_u;
    grid_u = tmp_u;
    tmp_u = tmp_ptr;
}

__global__ void advect_smoke_kernel(uint8_t *grid_s, float *grid_u, float *grid_v, float *grid_m, float *tmp_m, int width, int height, float dt) {
    int w = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int h = blockIdx.y * blockDim.y + threadIdx.y + 1;
    if (w >= width - 1 || h >= height - 1) return;
    // if there is collision object
    if (grid_s[coord(w, h, width)] == 0) return;
    tmp_m[coord(w, h, width)] = grid_m[coord(w, h, width)];
    float v = (grid_v[coord(w, h, width)] + grid_v[coord(w, h+1, width)]) / 2;  
    float u = (grid_u[coord(w, h, width)] + grid_u[coord(w+1, h, width)]) / 2;
    float x = (float) w - (u * dt);
    float y = (float) h - (v * dt);
    // correct if backtracked coordinates are out of bounds
    x = max(min(x, (float) (width-1)), (float) 0.0);
    y = max(min(y, (float) (height-1)), (float) 0.0);
    if (grid_s[coord((int) x, (int) y, width)] != 0 || ((int) x) == 0) {
        int idx = (int) x;
        int idy = (int) y;
        float ratio_x = x - idx;
        float ratio_y = y - idy;
        float tmp1 = (grid_m[coord(idx, idy, width)] * (1.0 - ratio_x)) + (grid_m[coord(idx+1, idy, width)] * ratio_x);
        float tmp2 = (grid_m[coord(idx, idy+1, width)] * (1.0 - ratio_x)) + (grid_m[coord(idx+1, idy+1, width)] * ratio_x);
        float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
        tmp_m[coord(w, h, width)] = tmp;
    }
}

void Solver_cuda::advect_smoke() {
    // Define block and grid sizes
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    advect_smoke_kernel<<<gridDim, blockDim>>>(grid_s, grid_u, grid_v, grid_m, tmp_m, width, height, dt);
    cudaDeviceSynchronize(); // Ensure all threads complete
    // swap the tmp and main grids
    float *tmp_ptr = grid_m;
    grid_m = tmp_m;
    tmp_m = tmp_ptr;
}

void Solver_cuda::wind_tunnel() {
    float *grid_vC = new float[width*height];
    float *tmp_vC = new float[width*height];
    float *grid_uC = new float[width*height];
    float *tmp_uC = new float[width*height];
    float *grid_mC = new float[width*height];
    float *tmp_mC = new float[width*height];
    uint8_t *grid_sC = new uint8_t[width*height];
    
    for (int i = 0; i < width * height; ++i) {
        grid_vC[i] = 0;
        tmp_vC[i] = 0;
        grid_uC[i] = 16.0;
        tmp_uC[i] = 16.0;
        grid_mC[i] = 0;
        tmp_mC[i] = 0;
    }

    for (int i = (height / 2) - (height / 16); i < (height / 2) + (height / 16); ++i) {
        grid_mC[coord(0, i, width)] = 1.0;
        grid_mC[coord(1, i, width)] = 1.0;
        tmp_mC[coord(0, i, width)] = 1.0;
        tmp_mC[coord(1, i, width)] = 1.0;
    }

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            // if outside boundary
            if (w == 0 || w == width - 1 || h == 0 || h == height - 1) {
                grid_sC[coord(w, h, width)] = 0;
            }
            // if object inside simulation space
            else if (sqrt((w + 1 - ((height / 2) + (height / 8)))*(w + 1 - ((height / 2) + (height / 8))) + (h + 1 - (width / 4))*(h + 1 - (width / 4))) <= (height / 8) + (height / 16)) {
                grid_sC[coord(w, h, width)] = 0;
                grid_uC[coord(w, h, width)] = 0;
                grid_uC[coord(w+1, h, width)] = 0;
                tmp_uC[coord(w, h, width)] = 0;
                tmp_uC[coord(w+1, h, width)] = 0;
            }
            // if space where fluid can flow
            else {
                grid_sC[coord(w, h, width)] = 1;
            }
        }
    }

    cudaMemcpy(grid_s, grid_sC, width * height * sizeof(uint8_t), cudaMemcpyHostToDevice);
    cudaMemcpy(grid_v, grid_vC, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_v, tmp_vC, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grid_u, grid_uC, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_u, tmp_uC, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(grid_m, grid_mC, width * height * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(tmp_m, tmp_mC, width * height * sizeof(float), cudaMemcpyHostToDevice);
}

__global__ void get_frame_kernel(uint8_t *grid_s, uint8_t *frame_data, float *grid_m, int width, int height) {
    int w = blockIdx.x * blockDim.x + threadIdx.x + 1;
    int h = blockIdx.y * blockDim.y + threadIdx.y + 1;

    if (w >= width - 1 || w == 0 || h == 0 || h >= height - 1) return;

    int idx = (h-1)*(width-2) + (w-1);
    frame_data[idx] = 255 - (128 * grid_m[coord(w, h, width)]);
    frame_data[idx] *= grid_s[coord(w, h, width)];
}

uint8_t *Solver_cuda::get_frame() {
    dim3 blockDim(16, 16);
    dim3 gridDim((width + blockDim.x - 1) / blockDim.x, (height + blockDim.y - 1) / blockDim.y);
    get_frame_kernel<<<gridDim, blockDim>>>(grid_s, frame_data, grid_m, width, height);
    cudaDeviceSynchronize();
    cudaMemcpy(frame_data_host, frame_data, (width-2) * (height-2) * sizeof(uint8_t), cudaMemcpyDeviceToHost);
    return frame_data_host;
}
