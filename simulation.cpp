#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <iostream>

#include "view.h"
#include "solver.h"
#include "solver_cuda.h"

const int WIDTH = 1024;
const int HEIGHT = 512;
const float SCALING = 2.0;
const int FPS = 60;

int main(int argc, char* argv[]) {
    View *view = new View(WIDTH * SCALING, HEIGHT * SCALING, WIDTH, HEIGHT, FPS);
    view->open_window();

    //Solver *sim = new Solver(WIDTH, HEIGHT, 120, 1.95, 0.4);
    //sim->wind_tunnel();

    Solver_cuda *sim = new Solver_cuda(WIDTH, HEIGHT, 120, 1.95, 0.4);
    sim->wind_tunnel();

    bool running = true;
    SDL_Event event;
    int iter = 1;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        Uint32 time = SDL_GetTicks();
        sim->iterate_compression();
        std::cout << SDL_GetTicks() - time << ' ';
        sim->advect_velocities();
        std::cout << SDL_GetTicks() - time << ' ';
        sim->advect_smoke();
        std::cout << SDL_GetTicks() - time << '\t';

        view->update_frame(sim->get_frame());
    }

    //view->close_window();
    return 0;
}
