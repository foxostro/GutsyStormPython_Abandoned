import pyglet
from pyglet.gl import *
from pyglet.window import key
from ctypes import pointer, sizeof
from Shader import Shader
import itertools
import time
import random
import pnoise
import numpy


RES_X = 128
RES_Y = 32
RES_Z = 128
L = 0.5 # Half-length of the cube along one side.
useWireframe = False
rot = 0.0

# Generate a heightmap for the terrain.
random.seed(time.time())
pnoise.shuffle()
voxelData = numpy.arange(RES_X*RES_Z).reshape(RES_X, RES_Z)
freq = 0.8
numOctaves = 1
for x,z in itertools.product(range(0,RES_X), range(0,RES_Z)):
    n = pnoise.perlinNoiseWithMultipleOctaves(float(x)/RES_X/freq,
                                              float(z)/RES_Z/freq,
                                              0.0,
                                              numOctaves)
    c = int(n*(RES_Y/2)+(RES_Y/2-1))
    voxelData[x,z] = c


def isGround(x,y,z):
    return y < voxelData[x,z]


def vec(*args):
    "Create a ctypes array of floats"
    return (GLfloat * len(args))(*args)


# Try to create a window that support antialiasing.
try:
    config = Config(sample_buffers=1,
                    samples=4,
                    depth_size=16,
                    double_buffer=True)
    window = pyglet.window.Window(width=640,
                                  height=480,
                                  resizable=True,
                                  config=config,
                                  vsync=True)
except pyglet.window.NoSuchConfigException:
    # Well, if it's not supported then get whatever we can get.
    window = pyglet.window.Window(width=640,
                                  height=480,
                                  resizable=True)

def getCubeVerts(x,y,z):
    return [ x-L, y+L, z+L,   x+L, y+L, z-L,   x-L, y+L, z-L, # Top Face
             x-L, y+L, z+L,   x+L, y+L, z+L,   x+L, y+L, z-L,
             x-L, y-L, z-L,   x+L, y-L, z-L,   x-L, y-L, z+L, # Bottom Face
             x+L, y-L, z-L,   x+L, y-L, z+L,   x-L, y-L, z+L,
             x-L, y-L, z+L,   x+L, y+L, z+L,   x-L, y+L, z+L, # Front Face
             x-L, y-L, z+L,   x+L, y-L, z+L,   x+L, y+L, z+L,
             x-L, y+L, z-L,   x+L, y+L, z-L,   x-L, y-L, z-L, # Back Face
             x+L, y+L, z-L,   x+L, y-L, z-L,   x-L, y-L, z-L,
             x+L, y+L, z-L,   x+L, y+L, z+L,   x+L, y-L, z+L, # Right Face
             x+L, y-L, z-L,   x+L, y+L, z-L,   x+L, y-L, z+L,
             x-L, y-L, z+L,   x-L, y+L, z+L,   x-L, y+L, z-L, # Left Face
             x-L, y-L, z+L,   x-L, y+L, z-L,   x-L, y-L, z-L,
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

# Generate one gigantic batch containing all polygon data.
# Many of the faces are hidden, so there is room for improvement here.
verts = []
norms = []
for x,y,z in itertools.product(range(0,RES_X),
                               range(0,RES_Y),
                               range(0,RES_Z)):
    if isGround(x,y,z):
        verts.extend(getCubeVerts(x - RES_X/2,
                                  y - RES_Y/2,
                                  z - RES_Z/2))
        norms.extend(cube_norms)
numTrianglesInBatch = len(verts)/3
print "Generated Geometry"

vbo_verts = GLuint()
vbo_norms = GLuint()

glGenBuffers(1, pointer(vbo_verts))
glGenBuffers(1, pointer(vbo_norms))

data = vec(*verts)
glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
del data

data = vec(*norms)
glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
del data

glEnableClientState(GL_VERTEX_ARRAY)
glEnableClientState(GL_NORMAL_ARRAY)
glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
glNormalPointer(GL_FLOAT, 0, 0)
glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
glVertexPointer(3, GL_FLOAT, 0, 0)

glClearColor(0.2, 0.4, 0.5, 1.0)

gluLookAt(0.0, 80.0, 120.0,  # Eye
          0.0, 0.0, 0.0,  # Center
          0.0, 1.0, 0.0)  # Up

glEnable(GL_CULL_FACE)
glFrontFace(GL_CCW)

# Simple light setup.  On Windows GL_LIGHT0 is enabled by default, but this is
# not the case on Linux or Mac, so remember to always include it.
glEnable(GL_LIGHTING)
glEnable(GL_LIGHT0)

glLightfv(GL_LIGHT0, GL_POSITION, vec(20.0, 40.0, 30.0, 0.0))
glLightfv(GL_LIGHT0, GL_AMBIENT, vec(0.3, 0.3, 0.3, 1.0))
glLightfv(GL_LIGHT0, GL_DIFFUSE, vec(0.9, 0.9, 0.9, 1.0))
glLightfv(GL_LIGHT0, GL_SPECULAR, vec(1.0, 1.0, 1.0, 1.0))

glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, vec(0.8, 0.5, 0.5, 1.0))
glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, vec(1, 1, 1, 1))
glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 10)

# create the Phong Shader by Jerome GUINOT aka 'JeGX' - jegx [at] ozone3d [dot] net
# see http://www.ozone3d.net/tutorials/glsl_lighting_phong.php
shader = Shader(['''
varying vec3 normal, lightDir0, lightDir1, eyeVec;

void main()
{
    normal = gl_NormalMatrix * gl_Normal;

    vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

    lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
    lightDir1 = vec3(gl_LightSource[1].position.xyz - vVertex);
    eyeVec = -vVertex;

    gl_Position = ftransform();
}
'''], ['''
varying vec3 normal, lightDir0, lightDir1, eyeVec;

void main (void)
{
    vec4 final_color =
    (gl_FrontLightModelProduct.sceneColor * gl_FrontMaterial.ambient) +
    (gl_LightSource[0].ambient * gl_FrontMaterial.ambient) +
    (gl_LightSource[1].ambient * gl_FrontMaterial.ambient);

    vec3 N = normalize(normal);
    vec3 L0 = normalize(lightDir0);
    vec3 L1 = normalize(lightDir1);

    float lambertTerm0 = dot(N,L0);
    float lambertTerm1 = dot(N,L1);

    if(lambertTerm0 > 0.0)
    {
        final_color += gl_LightSource[0].diffuse *
                       gl_FrontMaterial.diffuse *
                       lambertTerm0;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L0, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[0].specular *
                       gl_FrontMaterial.specular *
                       specular;
    }
    if(lambertTerm1 > 0.0)
    {
        final_color += gl_LightSource[1].diffuse *
                       gl_FrontMaterial.diffuse *
                       lambertTerm1;

        vec3 E = normalize(eyeVec);
        vec3 R = reflect(-L1, N);
        float specular = pow( max(dot(R, E), 0.0),
                         gl_FrontMaterial.shininess );
        final_color += gl_LightSource[1].specular *
                       gl_FrontMaterial.specular *
                       specular;
    }
    gl_FragColor = final_color;
}
'''])

fps_display = pyglet.clock.ClockDisplay()


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

    #############################################

    glEnable(GL_DEPTH_TEST)
    glEnable(GL_LIGHTING)

    glPushMatrix()
    glRotatef(rot, 0, 1, 0)
    shader.bind()
    glDrawArrays(GL_TRIANGLES, 0, numTrianglesInBatch)
    shader.unbind()
    glPopMatrix()

    #############################################

    glDisable(GL_DEPTH_TEST)
    glDisable(GL_LIGHTING)

    glMatrixMode(GL_PROJECTION)
    glPushMatrix()
    glLoadIdentity()
    gluOrtho2D(-400, 400, -400, 400)

    glMatrixMode(GL_MODELVIEW)
    glPushMatrix()
    glLoadIdentity()
    glTranslatef(-400, 350, 0)
    fps_display.draw()
    glPopMatrix()

    glMatrixMode(GL_PROJECTION)
    glPopMatrix()
    glMatrixMode(GL_MODELVIEW)


pyglet.app.run()
