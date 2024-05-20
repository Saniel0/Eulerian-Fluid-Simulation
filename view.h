#include <SDL2/SDL.h>
#include <GL/glew.h>

class View {
    private:
        int view_width;
        int view_height;
        int data_width;
        int data_height;
        int frame_time;
        Uint32 last_frame_time;
        SDL_Window* window;
        SDL_GLContext context;
        GLuint texture;
    
    public:
        View(int view_width, int view_height, int data_width, int data_height, int fps);
        int open_window();
        void close_window();
        void update_frame(uint8_t *frame_data);
};
