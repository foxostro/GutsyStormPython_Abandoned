#!/usr/bin/env python
# vim: et ts=4

from PIL import Image
from pnoise import PerlinNoise
import random
import time
import itertools


def generatePerlin(fn, w, h, freq, numOctaves):
    """Generates an image file containing perlin noise.
    fn   - File name for the image.
    w    - Width of the image, in pixels.
    h    - Height of the image, in pixels.
    freq - The frequency of the noise.
    numOctaves = The number of octaves of noise to use.
    """
    noiseSource = PerlinNoise()
    img = Image.new('RGB', (w,h))
    pix = img.load()

    for x, y in itertools.product(range(0,w), range(0,h)):
        n = noiseSource.getValueWithMultipleOctaves(float(x)/w/freq,
                                                    float(y)/h/freq,
                                                    0.0,
                                                    numOctaves)
        c = int(n*128+127)
        pix[x,y] = (c,c,c)

    img.save(fn)


def generateRidgedMulitFractal(fn, w, h, freq):
    """Generates an image file of ridged multifractal noise.
    fn   - File name for the image.
    w    - Width of the image, in pixels.
    h    - Height of the image, in pixels.
    freq - The frequency of the noise.
    numOctaves = The number of octaves of noise to use.
    """
    noiseSource = PerlinNoise()
    img = Image.new('RGB', (w,h))
    pix = img.load()

    for x, y in itertools.product(range(0,w), range(0,h)):
        n = abs(noiseSource.getValue(float(x)/w/freq,
                                     float(y)/h/freq,
                                     0.0))
        c = int(255 - n*255)
        pix[x,y] = (c,c,c)

    img.save(fn)

random.seed(time.time())
generatePerlin("test.png", 128, 128, 0.4, 4)
generateRidgedMulitFractal("test2.png", 128, 128, 0.4)
