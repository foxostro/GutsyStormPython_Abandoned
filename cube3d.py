import pyglet
from pyglet.gl import *
from pyglet.window import key
from ctypes import pointer, sizeof

rot = 0.0
useWireframe = False


def vec(*args):
    "Create a ctypes array of floats"
    return (GLfloat * len(args))(*args)


vbo_cube_verts = GLuint()
vbo_cube_norms = GLuint()

glGenBuffers(1, pointer(vbo_cube_verts))
glGenBuffers(1, pointer(vbo_cube_norms))

cube_verts = [
-1, +1, +1,   +1, +1, -1,   -1, +1, -1, # Top Face
-1, +1, +1,   +1, +1, +1,   +1, +1, -1,
-1, -1, -1,   +1, -1, -1,   -1, -1, +1, # Bottom Face
+1, -1, -1,   +1, -1, +1,   -1, -1, +1,
-1, -1, +1,   +1, +1, +1,   -1, +1, +1, # Front Face
-1, -1, +1,   +1, -1, +1,   +1, +1, +1,
-1, +1, -1,   +1, +1, -1,   -1, -1, -1, # Back Face
+1, +1, -1,   +1, -1, -1,   -1, -1, -1,
+1, +1, -1,   +1, +1, +1,   +1, -1, +1, # Right Face
+1, -1, -1,   +1, +1, -1,   +1, -1, +1,
-1, -1, +1,   -1, +1, +1,   -1, +1, -1, # Left Face
-1, -1, +1,   -1, +1, -1,   -1, -1, -1,
]

cube_norms = [
 0, +1,  0,    0, +1,  0,   0, +1,  0, # Top Face
 0, +1,  0,    0, +1,  0,   0, +1,  0,
 0, -1,  0,    0, -1,  0,   0, -1,  0, # Bottom Face
 0, -1,  0,    0, -1,  0,   0, -1,  0,
 0,  0, +1,    0,  0, +1,   0,  0, +1, # Front Face
 0,  0, +1,    0,  0, +1,   0,  0, +1,
 0,  0, -1,    0,  0, -1,   0,  0, -1, # Back Face
 0,  0, -1,    0,  0, -1,   0,  0, -1,
+1,  0,  0,   +1,  0,  0,  +1,  0,  0, # Right Face
+1,  0,  0,   +1,  0,  0,  +1,  0,  0,
-1,  0,  0,   -1,  0,  0,  -1,  0,  0, # Left Face
-1,  0,  0,   -1,  0,  0,  -1,  0,  0
]

data = vec(*cube_verts)
glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_verts)
glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
del data

data = vec(*cube_norms)
glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_norms)
glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
del data

# Try to create a window that support antialiasing.
try:
    config = Config(sample_buffers=1,
                    samples=4,
                    depth_size=24,
                    double_buffer=True)
    window = pyglet.window.Window(width=640,
                                  height=400,
                                  resizable=True,
                                  config=config,
                                  vsync=True)
except pyglet.window.NoSuchConfigException:
    # Well, if it's not supported then get whatever we can get.
    window = pyglet.window.Window(width=640,
                                  height=400,
                                  resizable=True)


glClearColor(0.2, 0.4, 0.5, 1.0)

gluLookAt(3.0, 3.0, 3.0,  # Eye
          0.0, 0.0, 0.0,  # Center
          0.0, 1.0, 0.0)  # Up

glEnableClientState(GL_VERTEX_ARRAY)


light0pos = [20.0, 20.0, 20.0, 0.0] # directional light

glEnable(GL_DEPTH_TEST)

glEnable(GL_CULL_FACE)
glFrontFace(GL_CCW)

# Simple light setup.  On Windows GL_LIGHT0 is enabled by default, but this is
# not the case on Linux or Mac, so remember to always include it.
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)

glLightfv(GL_LIGHT0, GL_POSITION, vec(*light0pos))
glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.8, 0.5, 0.5, 1.0))
glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 50)


def update(dt):
    global rot
    rot += 20 * dt
    rot %= 360
pyglet.clock.schedule(update)


@window.event
def on_key_press(symbol, modifiers):
    global useWireframe

    if symbol == key.W:
        useWireframe = not useWireframe
        return pyglet.event.EVENT_HANDLED
    elif symbol == key.ESCAPE:
        pyglet.app.exit()
        return pyglet.event.EVENT_HANDLED


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

    if useWireframe:
        glPolygonMode(GL_FRONT, GL_LINE)
    else:
        glPolygonMode(GL_FRONT, GL_FILL)

    glPushMatrix()
    glRotatef(rot, 0, 1, 0)

    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_norms)
    glNormalPointer(GL_FLOAT, 0, 0)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_cube_verts)
    glVertexPointer(3, GL_FLOAT, 0, 0)
    glColor3f(0.8, 0.8, 0.8)
    glDrawArrays(GL_TRIANGLES, 0, len(cube_verts)/3)

    glPopMatrix()


pyglet.app.run()
