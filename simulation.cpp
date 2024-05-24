#include <SDL2/SDL.h>
#include <GL/glew.h>
#include <iostream>

#include "view.h"
#include "solver.h"

const int WIDTH = 256;
const int HEIGHT = 128;
const float SCALING = 4.0;
const int FPS = 60;

int main(int argc, char* argv[]) {
    View *view = new View(WIDTH * SCALING, HEIGHT * SCALING, WIDTH, HEIGHT, FPS);
    view->open_window();

    Solver *sim = new Solver(WIDTH, HEIGHT, 60, 1.95, 0.4);
    sim->wind_tunnel();

    bool running = true;
    SDL_Event event;
    while (running) {
        while (SDL_PollEvent(&event)) {
            if (event.type == SDL_QUIT) {
                running = false;
            }
        }

        sim->iterate_compression();
        sim->advect_velocities();
        sim->advect_smoke();

        view->update_frame(sim->get_frame());
    }

    view->close_window();
    return 0;
}
