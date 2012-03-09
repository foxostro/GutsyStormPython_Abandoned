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
import Terrain
from math3D import Quaternion, Vector3, Frustum
import math3D


useWireframe = False
shader = None
fps_display = None
chunkStore = None
seed = 1330765820.45
keysDown = defaultdict(bool)
cameraPos = Vector3(92.317, 33.122, 122.606)
cameraRot = Quaternion.fromAxisAngle(Vector3(0,1,0), 0)
cameraSpeed = 5.0
cameraRotSpeed = 1.0
cameraFrustum = Frustum()
cameraEye = Vector3(0,0,0)
cameraCenter = Vector3(0,0,0)
cameraUp = Vector3(0,0,0)


def updateCameraFrustum():
    "Update the cached camera frustum and look vectors"
    global cameraFrustum, cameraEye, cameraCenter, cameraUp
    cameraEye, cameraCenter, cameraUp = \
        math3D.getCameraEyeCenterUp(cameraPos, cameraRot)
    cameraFrustum.setCamDef(cameraEye, cameraCenter, cameraUp)


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

    glMaterialfv(GL_FRONT_AND_BACK, GL_AMBIENT_AND_DIFFUSE, arrayOfGLfloat(0.5, 0.5, 0.5, 1.0))
    glMaterialfv(GL_FRONT_AND_BACK, GL_SPECULAR, arrayOfGLfloat(1, 1, 1, 1))
    glMaterialf(GL_FRONT_AND_BACK, GL_SHININESS, 1)


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
    global chunkStore
    global seed

    setupGLState()
    shader = createShaderObject()
    fps_display = pyglet.clock.ClockDisplay(format='%(fps).0f FPS')

    updateCameraFrustum()
    #seed = time.time()
    chunkStore = Terrain.ChunkStore(seed)
    chunkStore.setCamera(cameraPos, cameraRot, cameraFrustum)

    pyglet.app.run()


def update(dt):
    global cameraPos, cameraRot

    wasCameraModified = False

    if keysDown[key.W]:
        acceleration = cameraRot.mulByVec(Vector3(0, 0, -cameraSpeed*dt))
        cameraPos = cameraPos.add(acceleration)
        wasCameraModified = True
    elif keysDown[key.S]:
        acceleration = cameraRot.mulByVec(Vector3(0, 0, cameraSpeed*dt))
        cameraPos = cameraPos.add(acceleration)
        wasCameraModified = True

    if keysDown[key.A]:
        acceleration = cameraRot.mulByVec(Vector3(-cameraSpeed*dt, 0, 0))
        cameraPos = cameraPos.add(acceleration)
        wasCameraModified = True
    elif keysDown[key.D]:
        acceleration = cameraRot.mulByVec(Vector3(cameraSpeed*dt, 0, 0))
        cameraPos = cameraPos.add(acceleration)
        wasCameraModified = True

    if keysDown[key.LEFT]:
        deltaRot = Quaternion.fromAxisAngle(Vector3(0,1,0), cameraRotSpeed*dt)
        cameraRot = deltaRot.mulByQuat(cameraRot)
        wasCameraModified = True
    elif keysDown[key.RIGHT]:
        deltaRot = Quaternion.fromAxisAngle(Vector3(0,1,0), -cameraRotSpeed*dt)
        cameraRot = deltaRot.mulByQuat(cameraRot)
        wasCameraModified = True

    if keysDown[key.UP]:
        deltaRot = Quaternion.fromAxisAngle(Vector3(1,0,0), -cameraRotSpeed*dt)
        cameraRot = cameraRot.mulByQuat(deltaRot)
        wasCameraModified = True
    elif keysDown[key.DOWN]:
        deltaRot = Quaternion.fromAxisAngle(Vector3(1,0,0), cameraRotSpeed*dt)
        cameraRot = cameraRot.mulByQuat(deltaRot)
        wasCameraModified = True

    if wasCameraModified:
        updateCameraFrustum()
        chunkStore.setCamera(cameraPos, cameraRot, cameraFrustum)

    chunkStore.update(dt)

pyglet.clock.schedule(update)


@window.event
def on_key_press(symbol, modifiers):
    global useWireframe, keysDown, cameraRot

    keysDown[symbol] = True

    if symbol == key.R:
        useWireframe = not useWireframe
    elif symbol == key.I:
        cameraRot = Quaternion.fromAxisAngle(Vector3(0,1,0), 0)
    elif symbol == key.P:
        print "Camera Position:", cameraPos
        print "Camera Rotation:", Quaternion.toAxisAngle(cameraRot)
    elif symbol == key.ESCAPE:
        chunkStore.sync()
        pyglet.app.exit()

    return pyglet.event.EVENT_HANDLED


@window.event
def on_key_release(symbol, modifiers):
    global keysDown
    keysDown[symbol] = False
    return pyglet.event.EVENT_HANDLED


@window.event
def on_resize(width, height):
    global cameraFrustum
    glViewport(0, 0, width, height)
    glMatrixMode(GL_PROJECTION)
    glLoadIdentity()
    gluPerspective(65, width / float(height), .1, 1000)
    cameraFrustum.setCamInternals(65, width / float(height), .1, 1000)
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

    # Set the camera.
    gluLookAt(cameraEye.x,    cameraEye.y,    cameraEye.z,
              cameraCenter.x, cameraCenter.y, cameraCenter.z,
              cameraUp.x,     cameraUp.y,     cameraUp.z)

    shader.bind()
    chunkStore.drawVisibleChunks()
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


if __name__ == "__main__":
    main()
