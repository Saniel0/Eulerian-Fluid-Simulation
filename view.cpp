#include <iostream>

#include "view.h"

View::View(int view_width, int view_height, int data_width, int data_height, int fps) {
    // save basic parameter
    this->view_width = view_width;
    this->view_height = view_height;
    this->data_width = data_width;
    this->data_height = data_height;
    this->frame_time = 1000 / fps;
    // we have not rendered any frame yet
    this->last_frame_time = 0;
}

int View::open_window() {
    if (SDL_Init(SDL_INIT_VIDEO) < 0) {
        std::cerr << "Failed to initialize SDL: " << SDL_GetError() << std::endl;
        return 1;
    }

    window = SDL_CreateWindow("Eulerian Fluid Simulation", SDL_WINDOWPOS_CENTERED, SDL_WINDOWPOS_CENTERED, view_width, view_height, SDL_WINDOW_OPENGL);
    if (!window) {
        std::cerr << "Failed to create window: " << SDL_GetError() << std::endl;
        SDL_Quit();
        return 1;
    }

    context = SDL_GL_CreateContext(window);
    if (!context) {
        std::cerr << "Failed to create OpenGL context: " << SDL_GetError() << std::endl;
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    glewExperimental = GL_TRUE;
    if (glewInit() != GLEW_OK) {
        std::cerr << "Failed to initialize GLEW" << std::endl;
        SDL_GL_DeleteContext(context);
        SDL_DestroyWindow(window);
        SDL_Quit();
        return 1;
    }

    glViewport(0, 0, view_width, view_height);

    // create texture
    glGenTextures(1, &texture);
    glBindTexture(GL_TEXTURE_2D, texture);
    // set texture parameters
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_S, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_WRAP_T, GL_CLAMP_TO_EDGE);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MIN_FILTER, GL_NEAREST);
    glTexParameteri(GL_TEXTURE_2D, GL_TEXTURE_MAG_FILTER, GL_NEAREST);
    // initialize texture
    glTexImage2D(GL_TEXTURE_2D, 0, GL_LUMINANCE, data_width, data_height, 0, GL_LUMINANCE, GL_UNSIGNED_BYTE, nullptr);

    return 0;
}

void View::close_window() {
    glDeleteTextures(1, &texture);
    SDL_GL_DeleteContext(context);
    SDL_DestroyWindow(window);
    SDL_Quit();
}

void View::update_frame(uint8_t *frame_data) {
    // Textures are fast and allow for scaling

    // Upload the simulation data to the texture
    glTexSubImage2D(GL_TEXTURE_2D, 0, 0, 0, data_width, data_height, GL_LUMINANCE, GL_UNSIGNED_BYTE, frame_data);

    // Enable texturing
    glEnable(GL_TEXTURE_2D);

    // Clear the screen
    glClear(GL_COLOR_BUFFER_BIT);

    // Draw a textured quad
    glBegin(GL_QUADS);
    glTexCoord2f(0.0f, 0.0f);
    glVertex2f(-1.0f, -1.0f);
    glTexCoord2f(1.0f, 0.0f);
    glVertex2f(1.0f, -1.0f);
    glTexCoord2f(1.0f, 1.0f);
    glVertex2f(1.0f, 1.0f);
    glTexCoord2f(0.0f, 1.0f);
    glVertex2f(-1.0f, 1.0f);
    glEnd();

    // Disable texturing
    glDisable(GL_TEXTURE_2D);

    Uint32 time = SDL_GetTicks() - last_frame_time;
    if (time < frame_time) {
        SDL_Delay(frame_time - time);
    }
    last_frame_time = SDL_GetTicks();

    SDL_GL_SwapWindow(SDL_GL_GetCurrentWindow());
}
