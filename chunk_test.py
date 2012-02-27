#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"""Generates a bit of terrain and renders it in a window.
Prototype test-bed for the terrain generator and chunk geometry generation.
"""

import pyglet
from pyglet.gl import *
from pyglet.window import key
import random
import time
import tempfile

from Shader import Shader
from Chunk import Chunk
from TerrainGenerator import TerrainGenerator


#RES_X = 160
#RES_Y = 48
#RES_Z = 160
RES_X = 64
RES_Y = 32
RES_Z = 64

useWireframe = False
rot = 0.0
shader = None
fps_display = None
chunks = None


def vec(*args):
    "Convenience function to create a ctypes array of floats"
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
    global shader
    global fps_display
    global chunks

    setupGLState()
    shader = createShaderObject()
    fps_display = pyglet.clock.ClockDisplay()
    print "Setup initial OpenGL state."

    terrainGenerator = TerrainGenerator(RES_X, RES_Y, RES_Z, 1330332183.734619)
    terrainData = terrainGenerator.generate()
    chunks = map(lambda d: Chunk(d[0], d[1], d[2]), terrainData)
    print "Generated chunks."

    # To test chunk serialization, do a round-trip to disk.
    fileNames = [tempfile.mktemp() for i in range(0, len(chunks))]
    for fileName, chunk in zip(fileNames, chunks):
        chunk.saveToDisk(fileName)
        chunk.destroy() # frees VBOs
    del chunks

    chunks = []
    for fileName in fileNames:
        chunks.append(Chunk.loadFromDisk(fileName))
    print "Completed chunk round-trip to disk."

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
    glTranslatef(-RES_X/2, -RES_Y/2, -RES_Z/2) # terrain rotates at center, not the corner
    shader.bind()
    glEnableClientState(GL_VERTEX_ARRAY)
    glEnableClientState(GL_NORMAL_ARRAY)
    for chunk in chunks:
        chunk.bind()
        chunk.draw()
    glDisableClientState(GL_VERTEX_ARRAY)
    glDisableClientState(GL_NORMAL_ARRAY)
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
