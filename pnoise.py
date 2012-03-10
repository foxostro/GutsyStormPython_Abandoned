#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
#
# This file is part of GutsyStorm.
# GutsyStorm  Copyright (C) 2012  Andrew Fox <foxostro@gmail.com>.
#
# GutsyStorm is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# GutsyStorm is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
#
# You should have received a copy of the GNU General Public License
# along with GutsyStorm.  If not, see <http://www.gnu.org/licenses/>.
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
