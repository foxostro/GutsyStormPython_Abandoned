#!/usr/bin/env python
# vim: et ts=4
import math
import time
from Foundation import *
import objc


objc.loadBundle("CZGPerlinGenerator", globals(),
                bundle_path=objc.pathForFramework("./Frameworks/pnoise.framework"))


class PerlinNoise:
    def __init__(self, **kwargs):
        self.generator = CZGPerlinGenerator.alloc().init()
        self.generator.setOctaves_(kwargs.get('numOctaves', 4))
        self.generator.setPersistence_(kwargs.get('persistence', 0.5))
        self.generator.setZoom_(kwargs.get('zoom', 100))

        randomseed = kwargs.get('randomseed', 0)
        self.generator.regeneratePermutationTableWithSeed_(randomseed)


    def getValue(self, x, y, z, t=0.0):
        return self.generator.perlinNoiseX_y_z_t_(x, y, z, t)
