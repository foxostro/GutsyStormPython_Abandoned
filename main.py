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
import math
import tempfile
from collections import defaultdict

from Shader import Shader
from Chunk import Chunk
import TerrainGenerator
import vec
import quat


RES_X = 128
RES_Y = 64
RES_Z = 128

useWireframe = False
shader = None
fps_display = None
chunks = None
keysDown = defaultdict(bool)
cameraPos = vec.vec(0.0, 0.0, 100.0)
cameraRot = quat.quatFromAxisAngle(vec.vec(0,1,0), 0)
cameraSpeed = 10.0
cameraRotSpeed = 1.0


def arrayOfGLfloat(*args):
    "Convenience function to create a ctypes array of GLfloats"
    return (GLfloat * len(args))(*args)


def createWindow():
    "Creates a Pyglet window."
    # Try to create a window that support antialiasing.
    try:
        config = Config(sample_buffers=1,
                        samples=4,
                        depth_size=32,
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

    glEnable(GL_CULL_FACE)
    glFrontFace(GL_CCW)

    # Simple light setup. On Windows GL_LIGHT0 is enabled by default, but this
    # is not the case on Linux or Mac.
    glEnable(GL_LIGHTING)
    glEnable(GL_LIGHT0)

    glLightfv(GL_LIGHT0, GL_POSITION, arrayOfGLfloat(20.0, 40.0, 30.0, 0.0))
    glLightfv(GL_LIGHT0, GL_AMBIENT, arrayOfGLfloat(0.3, 0.3, 0.3, 1.0))
    glLightfv(GL_LIGHT0, GL_DIFFUSE, arrayOfGLfloat(0.9, 0.9, 0.9, 1.0))
    glLightfv(GL_LIGHT0, GL_SPECULAR, arrayOfGLfloat(1.0, 1.0, 1.0, 1.0))

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arrayOfGLfloat(0.8, 0.5, 0.5, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, arrayOfGLfloat(1, 1, 1, 1))
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

    # Generate terrain data and geometry.
    a = time.time()
    seed = 1330765820.45 # time.time()
    print "seed:", seed
    terrainData = TerrainGenerator.generate(RES_X, RES_Y, RES_Z, seed)
    b = time.time()
    print "Generated terrain. It took %.1fs." % (b-a)

    # Save list of terrain/geometry chunks.
    # Initialization must be done on the main thread as VBOs must be generated
    # on the main thread.
    a = time.time()
    chunks = map(lambda d: Chunk(d[0], d[1], d[2], d[3], d[4]), terrainData)
    b = time.time()
    print "Generated chunks. It took %.1fs." % (b-a)

    ## To test chunk serialization, do a round-trip to disk.
    ## Save to disk...
    #a = time.time()
    #fileNames = [tempfile.mktemp() for i in range(0, len(chunks))]
    #for fileName, chunk in zip(fileNames, chunks):
    #    chunk.saveToDisk(fileName)
    #    chunk.destroy() # frees VBOs
    #del chunks
    #b = time.time()
    #print "Saved chunks to disk. It took %.1fs." % (b-a)
    #
    ## Load from disk...
    #a = time.time()
    #chunks = []
    #for fileName in fileNames:
    #    chunks.append(Chunk.loadFromDisk(fileName))
    #b = time.time()
    #print "Loaded chunks from disk. It took %.1fs." % (b-a)
    #print "Completed chunk round-trip to disk."

    pyglet.app.run()


def update(dt):
    global cameraPos, cameraRot

    if keysDown[key.W]:
        acceleration = quat.mulByVec(cameraRot, vec.vec(0, 0, -cameraSpeed*dt))
        cameraPos = vec.add(cameraPos, acceleration)
    elif keysDown[key.S]:
        acceleration = quat.mulByVec(cameraRot, vec.vec(0, 0, cameraSpeed*dt))
        cameraPos = vec.add(cameraPos, acceleration)

    if keysDown[key.A]:
        acceleration = quat.mulByVec(cameraRot, vec.vec(-cameraSpeed*dt, 0, 0))
        cameraPos = vec.add(cameraPos, acceleration)
    elif keysDown[key.D]:
        acceleration = quat.mulByVec(cameraRot, vec.vec(cameraSpeed*dt, 0, 0))
        cameraPos = vec.add(cameraPos, acceleration)

    if keysDown[key.LEFT]:
        deltaRot = quat.quatFromAxisAngle(vec.vec(0,1,0), -cameraRotSpeed*dt)
        cameraRot = quat.mulByQuat(cameraRot, deltaRot)
    elif keysDown[key.RIGHT]:
        deltaRot = quat.quatFromAxisAngle(vec.vec(0,1,0), cameraRotSpeed*dt)
        cameraRot = quat.mulByQuat(cameraRot, deltaRot)

    localAxisX = vec.normalize(quat.mulByVec(cameraRot, vec.vec(1,0,0)))

    if keysDown[key.UP]:
        deltaRot = quat.quatFromAxisAngle(localAxisX, -cameraRotSpeed*dt)
        print localAxisX
        cameraRot = quat.mulByQuat(cameraRot, deltaRot)
    elif keysDown[key.DOWN]:
        deltaRot = quat.quatFromAxisAngle(localAxisX, cameraRotSpeed*dt)
        cameraRot = quat.mulByQuat(cameraRot, deltaRot)

pyglet.clock.schedule(update)


@window.event
def on_key_press(symbol, modifiers):
    global useWireframe, keysDown

    keysDown[symbol] = True

    if symbol == key.R:
        useWireframe = not useWireframe
    elif symbol == key.P:
        print "Camera Position:", cameraPos
        print "Camera Rotation:", cameraRot
    elif symbol == key.ESCAPE:
        pyglet.app.exit()

    return pyglet.event.EVENT_HANDLED


@window.event
def on_key_release(symbol, modifiers):
    global keysDown
    keysDown[symbol] = False
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

    # Apply camera transformation.
    axis, angle = quat.quatToAxisAngle(cameraRot)
    glRotatef(angle * 180.0/math.pi, axis[0], axis[1], axis[2])
    del axis, angle
    glTranslatef(-cameraPos[0], -cameraPos[1], -cameraPos[2])

    glTranslatef(-RES_X/2, -RES_Y/2, -RES_Z/2) # terrain position is at its center
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
