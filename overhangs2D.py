#!/usr/bin/env python
# vim: et ts=4

from PIL import Image
from pnoise import PerlinNoise
import random
import time
import itertools


def groundGradient(p):
    """Return a value between -1 and +1 so that a line through the y-axis maps
    to a smooth gradient of values from -1 to +1.
    """
    y = p[1]

    if y < 0.05:
        return -1
    elif y > 1.0:
        return +1
    else:
        return 2.0*y - 1.0


def isGround(noiseSource0, noiseSource1, p):
    """"Returns True if the point is ground, False otherwise.
    """
    freq0 = 1.0
    freq1 = 2.0
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


def generateImage(fn, w, h):
    """Generates an image file containing perlin noise.
    fn   - File name for the image.
    w    - Width of the image, in pixels.
    h    - Height of the image, in pixels.
    """
    noiseSource0 = PerlinNoise()
    noiseSource1 = PerlinNoise()
    img = Image.new('RGB', (w,h))
    pix = img.load()

    for x, y in itertools.product(range(0,w), range(0,h)):
        p = (float(x)/w, float(y)/h, 0.5)
        if isGround(noiseSource0, noiseSource1, p):
            c = 0
        else:
            c = 255
        pix[x,y] = (c,c,c)

    img.save(fn)

random.seed(time.time())
generateImage("test.png", 64, 32)
