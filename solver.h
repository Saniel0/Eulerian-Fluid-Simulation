#include <cstdint>

class Solver {
    private:
        int width;
        int height;
        int iters;
        float relax_factor;
        float dt;
        float *grid_v; // horizontal velocity - staggered grid
        float *grid_u; // vertical velocity - staggered grid
        float *tmp_v;
        float *tmp_u;
        uint8_t *grid_s; // number of directions where velocity can change, 0-4 - collocated grid
        float *grid_m;
        float *tmp_m;
        uint8_t *frame_data;
        inline int coord(int w, int h);

    public:
        Solver(int w, int h, int iters, float relax_factor, float dt);
        ~Solver();
        void iterate_compression();
        void advect_velocities();
        void advect_smoke();
        uint8_t *get_frame();
        void wind_tunnel();
};
