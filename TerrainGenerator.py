#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"""Terrain generator module.
"""

import numpy
import itertools
import time
from pnoise import PerlinNoise


def terrainWorker(args):
    """Run by worker processes for the terrain generator to compute terrain for
    a single chunk.
    gen - TerrainGenerator object
    extent - A list containing two triples which describe the min and max
             corners of the terrain bounding cube, in world-space.
    """
    assert len(args)==2
    gen = args[0]
    extent = args[1]
    voxelData = gen.computeTerrainData(extent[0], extent[1])
    return [voxelData, extent[0], extent[1]]


class TerrainGenerator:
    "Generates procedural voxel terrain from given parameters."

    def groundGradient(self, p):
        """Return a value between -1 and +1 so that a line through the y-axis
        maps to a smooth gradient of values from -1 to +1.
        """
        y = p[1]
        if y < 0.0:
            return -1
        elif y > self.terrainHeight:
            return +1
        else:
            return 2.0*(y/self.terrainHeight) - 1.0


    def isGround(self, p):
        "Returns True if the point is ground, False otherwise."
        turbScaleX = 1.5
        turbScaleY = self.terrainHeight / 2.0
        n = self.noiseSource0.getValue(float(p[0]),
                                       float(p[1]),
                                       float(p[2]))
        yFreq = turbScaleX * ((n+1) / 2.0)
        t = turbScaleY * \
            self.noiseSource1.getValue(float(p[0]),
                                       float(p[1])*yFreq,
                                       float(p[2]))
        pPrime = (p[0], p[1] + t, p[1])
        return self.groundGradient(pPrime) <= 0


    def computeTerrainData(self, minP, maxP):
        """
        Returns voxelData which represents the voxel terrain values for the
        points between minP and maxP. The chunk is translated so that
        voxelData[0,0,0] corresponds to (minX, minY, minZ).
        The size of the chunk is unscaled so that, for example, the width of
        the chunk is equal to maxP-minP. Ditto for the other major axii.
        """
        print "Generating terrain for chunk; minP=%r, maxP=%r" % (minP, maxP)
        minX, minY, minZ = minP
        maxX, maxY, maxZ = maxP

        voxelData = numpy.arange((maxX-minX)*(maxY-minY)*(maxZ-minZ))
        voxelData = voxelData.reshape(maxX-minX, maxY-minY, maxZ-minZ)

        for x,y,z in itertools.product(range(minX, maxX),
                                       range(minY, maxY),
                                       range(minZ, maxZ)):
            # p is in world-space, not chunk-space
            p = (float(x), float(y), float(z))
            voxelData[x - minX, y - minY, z - minZ] = self.isGround(p)

        return voxelData


    def breakWorldIntoChunkExtents(self):
        # Define the regions to use for the world's chunks.
        numChunksAlongX = 8
        numChunksAlongZ = 8
        assert self.terrainWidth % numChunksAlongX == 0
        assert self.terrainDepth % numChunksAlongZ == 0
        stepX = self.terrainWidth / numChunksAlongX
        stepZ = self.terrainDepth / numChunksAlongZ
        extents = []
        for x in xrange(0, self.terrainWidth, stepX):
            for z in xrange(0, self.terrainDepth, stepZ):
                # The point here is to pack together arguments
                # to the terrainWorker processes.
                extents.append([self,
                                [(x, 0, z),
                                 (x+stepX, self.terrainHeight, z+stepZ)]
                               ])
        return extents


    def generate(self):
        "Generates and returns terrain data."
        a = time.time()
        extents = self.breakWorldIntoChunkExtents()
        terrainData = map(terrainWorker, extents)
        b = time.time()
        print "Generated terrain. It took %.1fs." % (b-a)
        return terrainData


    def __init__(self, terrainWidth, terrainHeight, terrainDepth, randomseed):
        print "Using random seed: %r" % randomseed

        self.noiseSource0 = PerlinNoise(randomseed=randomseed)
        self.noiseSource1 = PerlinNoise(randomseed=randomseed)

        self.terrainWidth = terrainWidth
        self.terrainHeight = terrainHeight
        self.terrainDepth = terrainDepth
