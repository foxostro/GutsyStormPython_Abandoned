import pyglet
from pyglet.gl import *
from ctypes import pointer, sizeof


vbo = GLuint()

glGenBuffers(1, pointer(vbo))

cube_tris = [ -1, +1, -1,   +1, +1, -1,   -1, +1, +1, # Top Face
              +1, +1, -1,   +1, +1, +1,   -1, +1, +1,
              -1, -1, -1,   +1, -1, -1,   -1, -1, +1, # Bottom Face
              +1, -1, -1,   +1, -1, +1,   -1, -1, +1,
              -1, +1, +1,   +1, +1, +1,   -1, -1, +1, # Front Face
              +1, +1, +1,   +1, -1, +1,   -1, -1, +1,
              -1, +1, -1,   +1, +1, -1,   -1, -1, -1, # Back Face
              +1, +1, -1,   +1, -1, -1,   -1, -1, -1,
              +1, +1, +1,   +1, +1, -1,   +1, -1, +1, # Right Face
              +1, +1, -1,   +1, -1, -1,   +1, -1, +1
              -1, +1, +1,   -1, +1, -1,   -1, -1, +1, # Left Face
              -1, +1, -1,   -1, -1, -1,   -1, -1, +1
            ]
data = (GLfloat*len(cube_tris))(*cube_tris)

glBindBuffer(GL_ARRAY_BUFFER, vbo)
glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)

window = pyglet.window.Window(width=640, height=400)

glClearColor(0.2, 0.4, 0.5, 1.0)

gluLookAt(5.0, 5.0, -5.0, # Eye
          0.0, 0.0, 0.0,  # Center
          0.0, 1.0, 0.0)  # Up

glEnableClientState(GL_VERTEX_ARRAY)


@window.event
def on_resize(width, height):
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(65, width / float(height), .1, 1000)
    glMatrixMode(GL_MODELVIEW)
    return pyglet.event.EVENT_HANDLED


@window.event
def on_draw():
    glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
    glColor3f(0.8, 0.8, 0.8)
    glBindBuffer(GL_ARRAY_BUFFER, vbo)
    glVertexPointer(3, GL_FLOAT, 0, 0)
    glDrawArrays(GL_TRIANGLES, 0, len(cube_tris)/3)


pyglet.app.run()
