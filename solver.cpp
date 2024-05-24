#include <math.h>
#include <algorithm>
#include <iostream>

#include "solver.h"

inline int Solver::coord(int w, int h) {
    return ((h * width) + w);
}

Solver::Solver(int w, int h, int iters, float relax_factor, float dt) {
    this->width = w + 2; // +2 for border
    this->height = h + 2;
    this->iters = iters;
    this->relax_factor = relax_factor;
    this->dt = dt;
    this->grid_v = new float[this->width * this->height];
    this->grid_u = new float[this->width * this->height];
    this->tmp_v = new float[this->width * this->height];
    this->tmp_u = new float[this->width * this->height];
    this->grid_s = new uint8_t[this->width * this->height];
    this->grid_m = new float[this->width * this->height];
    this->tmp_m = new float[this->width * this->height];
    this->frame_data = new uint8_t[w * h];
}

Solver::~Solver() {
    delete[] grid_v;
    delete[] grid_u;
    delete[] tmp_v;
    delete[] tmp_u;
    delete[] grid_s;
    delete[] grid_m;
    delete[] tmp_m;
    delete[] frame_data;
}

void Solver::iterate_compression() {
    for (int i = 0; i < iters * 2; ++i) {
        // do not process border cells
        for (int h = 1; h < height - 1; ++h) {
            for (int w = 1; w < width - 1; ++w) {
                if ((w+h) % 2 == i % 2) {
                    continue;
                }
                // if there is object in the cell
                if (grid_s[coord(w, h)] == 0) {
                    continue;
                }
                // count how many surrounding cells are border cells
                int s_up = grid_s[coord(w, h-1)];
                int s_down = grid_s[coord(w, h+1)];
                int s_left = grid_s[coord(w-1, h)];
                int s_right = grid_s[coord(w+1, h)];
                int s = s_up + s_down + s_left + s_right;
                // calculate divergence and divergence correction (with relax factor for faster convergence)
                float div = - grid_v[coord(w, h)] + grid_v[coord(w, h+1)] - grid_u[coord(w, h)] + grid_u[coord(w+1, h)];
                float p = div / s;
                p *= relax_factor;
                // update cell velocities
                grid_v[coord(w, h)] += s_up * p;
                grid_v[coord(w, h+1)] -= s_down * p;
                grid_u[coord(w, h)] += s_left * p;
                grid_u[coord(w+1, h)] -= s_right * p;
            }
        }
    }
}

void Solver::advect_velocities() {
    // ignore border cells
    for (int h = 1; h < height - 1; ++h) {
        for (int w = 1; w < width - 1; ++w) {
            // if object cell
            if (grid_s[coord(w, h)] == 0) {
                continue;    
            }
            tmp_v[coord(w, h)] = grid_v[coord(w, h)];
            tmp_u[coord(w, h)] = grid_u[coord(w, h)];
            // if there is not obstacle upwards
            if (grid_s[coord(w, h-1)] != 0) {
                float v = grid_v[coord(w, h)];
                // get u part of the vector by avareging 4 surrounding 'u's
                float u = (grid_u[coord(w, h-1)] + grid_u[coord(w+1, h-1)] + grid_u[coord(w, h)] + grid_u[coord(w+1, h)]) / 4;
                // backtrack based on timestep
                float x = (float) w - (u * dt);
                float y = (float) h - (v * dt);
                x = std::max(std::min(x, (float) (width-1)), (float) 1.0);
                y = std::max(std::min(y, (float) (height-1)), (float) 1.0);
                if (grid_s[coord((int) x, (int) y)] != 0) {
                    int idx = (int) x;
                    int idy = (int) y;
                    float ratio_x = x - idx;
                    float ratio_y = y - idy;
                    float tmp1 = (grid_v[coord(idx, idy)] * (1.0 - ratio_x)) + (grid_v[coord(idx+1, idy)] * ratio_x);
                    float tmp2 = (grid_v[coord(idx, idy+1)] * (1.0 - ratio_x)) + (grid_v[coord(idx+1, idy+1)] * ratio_x);
                    float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
                    tmp_v[coord(w, h)] = tmp;
                }
            }
            // if there is not obstacle leftwards
            if (grid_s[coord(w-1, h)] != 0) {
                float u = grid_u[coord(w, h)];
                // get v part of the vector by avareging 4 surrounding 'v's
                float v = (grid_v[coord(w-1, h)] + grid_v[coord(w, h)] + grid_v[coord(w-1, h+1)] + grid_v[coord(w, h+1)]) / 4;
                // backtrack based on timestep
                float x = (float) w - (u * dt);
                float y = (float) h - (v * dt);
                x = std::max(std::min(x, (float) (width-1)), (float) 1.0);
                y = std::max(std::min(y, (float) (height-1)), (float) 1.0);
                if (grid_s[coord((int) x, (int) y)] != 0) {
                    int idx = (int) x;
                    int idy = (int) y;
                    float ratio_x = x - idx;
                    float ratio_y = y - idy;
                    float tmp1 = (grid_u[coord(idx, idy)] * (1.0 - ratio_x)) + (grid_u[coord(idx+1, idy)] * ratio_x);
                    float tmp2 = (grid_u[coord(idx, idy+1)] * (1.0 - ratio_x)) + (grid_u[coord(idx+1, idy+1)] * ratio_x);
                    float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
                    tmp_u[coord(w, h)] = tmp;
                }
            }
        }
    }
    float *tmp_ptr = grid_v;
    grid_v = tmp_v;
    tmp_v = tmp_ptr;

    tmp_ptr = grid_u;
    grid_u = tmp_u;
    tmp_u = tmp_ptr;
}

void Solver::advect_smoke() {
    // ignore border cells
    for (int h = 1; h < height - 1; ++h) {
        for (int w = 1; w < width - 1; ++w) {
            // if not object cell
            if (grid_s[coord(w, h)] != 0) {
                //tmp_m[coord(w, h)] = grid_m[coord(w, h)];
                float v = (grid_v[coord(w, h)] + grid_v[coord(w, h+1)]) / 2;  
                float u = (grid_u[coord(w, h)] + grid_u[coord(w+1, h)]) / 2;
                float x = (float) w - (u * dt);
                float y = (float) h - (v * dt);
                x = std::max(std::min(x, (float) (width-1)), (float) 0.0);
                y = std::max(std::min(y, (float) (height-1)), (float) 0.0);
                if (grid_s[coord((int) x, (int) y)] != 0 || ((int) x) == 0) {
                    int idx = (int) x;
                    int idy = (int) y;
                    float ratio_x = x - idx;
                    float ratio_y = y - idy;
                    float tmp1 = (grid_m[coord(idx, idy)] * (1.0 - ratio_x)) + (grid_m[coord(idx+1, idy)] * ratio_x);
                    float tmp2 = (grid_m[coord(idx, idy+1)] * (1.0 - ratio_x)) + (grid_m[coord(idx+1, idy+1)] * ratio_x);
                    float tmp = (tmp1 * (1.0 - ratio_y)) + (tmp2 * ratio_y);
                    tmp_m[coord(w, h)] = tmp;
                }
            }
        }
    }
    float *tmp_ptr = grid_m;
    grid_m = tmp_m;
    tmp_m = tmp_ptr;
}

void Solver::wind_tunnel() {
    for (int i = 0; i < width * height; ++i) {
        grid_v[i] = 0;
        tmp_v[i] = 0;
        grid_u[i] = 8.0;
        tmp_u[i] = 8.0;
        grid_m[i] = 0;
        tmp_m[i] = 0;
    }

    for (int i = (height / 2) - (height / 16); i < (height / 2) + (height / 16); ++i) {
        grid_m[coord(0, i)] = 1.0;
        grid_m[coord(1, i)] = 1.0;
        tmp_m[coord(0, i)] = 1.0;
        tmp_m[coord(1, i)] = 1.0;
    }

    for (int h = 0; h < height; ++h) {
        for (int w = 0; w < width; ++w) {
            // if outside boundary
            if (w == 0 || w == width - 1 || h == 0 || h == height - 1) {
                grid_s[coord(w, h)] = 0;
            }
            // if object inside simulation space
            else if (sqrt((w + 1 - ((height / 2) + (height / 8)))*(w + 1 - ((height / 2) + (height / 8))) + (h + 1 - (width / 4))*(h + 1 - (width / 4))) <= (height / 8) + (height / 16)) {
                grid_s[coord(w, h)] = 0;
                grid_u[coord(w, h)] = 0;
                grid_u[coord(w+1, h)] = 0;
                tmp_u[coord(w, h)] = 0;
                tmp_u[coord(w+1, h)] = 0;
            }
            // if space where fluid can flow
            else {
                grid_s[coord(w, h)] = 1;
            }
        }
    }
}

uint8_t *Solver::get_frame() {
    for (int h = 1; h < height - 1; ++h) {
        for (int w = 1; w < width - 1; ++w) {
            int idx = (h-1)*(width-2) + (w-1);
            frame_data[idx] = 255 - (128 * grid_m[coord(w, h)]);
            if (grid_s[coord(w, h)] == 0) {
                frame_data[idx] = 0;
            }
        }
    }
    return frame_data;
}
