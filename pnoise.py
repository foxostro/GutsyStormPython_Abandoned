#!/usr/bin/env python
# vim: et ts=4
import math
import random
from Foundation import *
import objc


objc.loadBundle("CZGPerlinGenerator", globals(),
                bundle_path=objc.pathForFramework("./Frameworks/pnoise.framework"))


class PerlinNoise:
    def __init__(self):
        self.generator = CZGPerlinGenerator.alloc().init()


    def getValue(self, x, y, z):
        return self.generator.perlinNoiseX_y_z_t_(x, y, z, 0.0)


    def getValueWithMultipleOctaves(self, x, y, z, numOctaves):
        """Computes a perlin noise value at the specified position with multiple
        octaves of noise layered over top of it.
        """
        c = 0
        for i in range(1, numOctaves+1):
            a = pow(2, i)
            c += self.getValue(x * a, y * a, z * a) / a
        return c
