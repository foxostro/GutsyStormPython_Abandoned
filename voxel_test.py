#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
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


RES_X = 64
RES_Y = 32
RES_Z = 64
L = 0.5 # Half-length of the cube along one side.
useWireframe = False
rot = 0.0
numTrianglesInBatch = None
shader = None
fps_display = None


def groundGradient(p):
    """Return a value between -1 and +1 so that a line through the y-axis maps
    to a smooth gradient of values from -1 to +1.
    """
    y = p[1]
    if y < 0.0:
        return -1
    elif y > 1.0:
        return +1
    else:
        return 2.0*y - 1.0


def isGround(noiseSource0, noiseSource1, p):
    """"Returns True if the point is ground, False otherwise.
    """
    freq0 = 1.0
    freq1 = 1.0
    numOctaves0 = 4
    numOctaves1 = 4
    turbScaleX = 1.3
    turbScaleY = 1.0
    n = noiseSource0.getValueWithMultipleOctaves(float(p[0])*freq0,
                                                 float(p[1])*freq0,
                                                 float(p[2])*freq0,
                                                 numOctaves0)
    yFreq = turbScaleX * ((n+1) / 2.0)
    t = turbScaleY * noiseSource1.getValueWithMultipleOctaves(float(p[0])*freq1,
                                                              float(p[1])*yFreq,
                                                              float(p[2])*freq1,
                                                              numOctaves1)
    pPrime = (p[0], p[1] + t, p[1])
    return groundGradient(pPrime) <= 0


def computeTerrainData():
    noiseSource0 = pnoise.PerlinNoise()
    noiseSource1 = pnoise.PerlinNoise()
    voxelData = numpy.arange(RES_X*RES_Y*RES_Z).reshape(RES_X, RES_Y, RES_Z)

    # Sample a unit cube of the isGround function and save to voxelData.
    # Sampling resolution is specified by RES_X, RES_Y, and RES_Z.
    for x,y,z in itertools.product(range(0,RES_X),
                                   range(0,RES_Y),
                                   range(0,RES_Z)):
        # p should range from 0 to +1 on each axis
        p = (float(x) / RES_X,
             float(y) / RES_Y,
             float(z) / RES_Z)
        voxelData[x, y, z] = isGround(noiseSource0, noiseSource1, p)

    return voxelData


def vec(*args):
    "Create a ctypes array of floats"
    return (GLfloat * len(args))(*args)


def createWindow():
    "Creates a Pyglet window."
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

    return window
window = createWindow()


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


def generateGeometry(voxelData):
    """Generate one gigantic batch containing all polygon data.
    Many of the faces are hidden, so there is room for improvement here.
    """
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
    verts = []
    norms = []

    for x,y,z in itertools.product(range(0,RES_X),
                                   range(0,RES_Y),
                                   range(0,RES_Z)):
        if voxelData[x,y,z]:
            verts.extend(getCubeVerts(x - RES_X/2,
                                      y - RES_Y/2,
                                      z - RES_Z/2))
            norms.extend(cube_norms)

    return verts, norms


def createVertexBufferObject(verts):
    vbo_verts = GLuint()
    glGenBuffers(1, pointer(vbo_verts))
    data = vec(*verts)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
    glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
    return vbo_verts


def bindVerts(vbo_verts):
    glEnableClientState(GL_VERTEX_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
    glVertexPointer(3, GL_FLOAT, 0, 0)


def bindNorms(vbo_norms):
    glEnableClientState(GL_NORMAL_ARRAY)
    glBindBuffer(GL_ARRAY_BUFFER, vbo_norms)
    glNormalPointer(GL_FLOAT, 0, 0)


def setupGLState():
    glClearColor(0.2, 0.4, 0.5, 1.0)

    gluLookAt(10.0, 50.0, 64.0,  # Eye
               0.0,  0.0,  0.0,  # Center
               0.0,  1.0,  0.0)  # Up

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


def createShaderObject():
    "Create a shader to apply to the terrain."
    # create the Phong Shader by Jerome GUINOT aka 'JeGX' - jegx [at] ozone3d [dot] net
    # see http://www.ozone3d.net/tutorials/glsl_lighting_phong.php
    return Shader(['''
    varying vec3 normal, lightDir0, eyeVec;

    void main()
    {
        normal = gl_NormalMatrix * gl_Normal;

        vec3 vVertex = vec3(gl_ModelViewMatrix * gl_Vertex);

        lightDir0 = vec3(gl_LightSource[0].position.xyz - vVertex);
        eyeVec = -vVertex;

        gl_Position = ftransform();
    }
    '''], ['''
    varying vec3 normal, lightDir0, eyeVec;

    void main (void)
    {
        vec4 final_color =
        (gl_FrontLightModelProduct.sceneColor * gl_FrontMaterial.ambient) +
        (gl_LightSource[0].ambient * gl_FrontMaterial.ambient);

        vec3 N = normalize(normal);
        vec3 L0 = normalize(lightDir0);

        float lambertTerm0 = dot(N,L0);

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
        gl_FragColor = final_color;
    }
    '''])


def main():
    global window
    global numTrianglesInBatch
    global shader
    global fps_display

    random.seed(time.time())
    print "Seeded random number generator"

    setupGLState()
    shader = createShaderObject()
    fps_display = pyglet.clock.ClockDisplay()
    print "Setup initial OpenGL state."

    voxelData = computeTerrainData()
    print "Computed terrain"

    verts, norms = generateGeometry(voxelData)
    numTrianglesInBatch = len(verts)/3
    print "Generated Geometry"

    vbo_verts = createVertexBufferObject(verts)
    del verts
    vbo_norms = createVertexBufferObject(norms)
    del norms
    print "Created vertex buffer objects"

    bindVerts(vbo_verts)
    bindNorms(vbo_norms)
    print "Uploaded geometry to the GPU"

    pyglet.app.run()


def update(dt):
    global rot
    rot += 5 * dt
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


main()
