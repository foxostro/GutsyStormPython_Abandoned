#!/usr/bin/env python
# vim: et ts=4

from PIL import Image
import pnoise
import random
import time
import itertools


def generateImage(fn, w, h, freq, numOctaves):
    """Generates an image file containing perlin noise.
    fn   - File name for the image.
    w    - Width of the image, in pixels.
    h    - Height of the image, in pixels.
    freq - The frequency of the noise.
    numOctaves = The number of octaves of noise to use.
    """
    img = Image.new('RGB', (w,h))
    pix = img.load()

    for x, y in itertools.product(range(0,w), range(0,h)):
        n = pnoise.perlinNoiseWithMultipleOctaves(float(x)/w/freq,
                                                  float(y)/h/freq,
                                                  0.0,
                                                  numOctaves)
        c = int(n*128+127)
        pix[x,y] = (c,c,c)

    img.save(fn)

random.seed(time.time())
pnoise.shuffle()
generateImage("test.png", 128, 128, 0.4, 4)
