#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"""Terrain generation
"""

import numpy
import itertools
import time
import multiprocessing
from pnoise import PerlinNoise
from math3D import Vector3


def groundGradient(terrainHeight, p):
    """Return a value between -1 and +1 so that a line through the y-axis
    maps to a smooth gradient of values from -1 to +1.
    """
    y = p.y
    if y < 0.0:
        return -1
    elif y > terrainHeight:
        return +1
    else:
        return 2.0*(y/terrainHeight) - 1.0


def isGround(terrainHeight, noiseSource0, noiseSource1, p):
    "Returns True if the point is ground, False otherwise."
    turbScaleX = 1.5
    turbScaleY = terrainHeight / 2.0
    n = noiseSource0.getValue(float(p.x),
                              float(p.y),
                              float(p.z))
    yFreq = turbScaleX * ((n+1) / 2.0)
    t = turbScaleY * \
        noiseSource1.getValue(float(p.x),
                              float(p.y)*yFreq,
                              float(p.z))
    pPrime = Vector3(p.x, p.y + t, p.y)
    return groundGradient(terrainHeight, pPrime) <= 0


def computeTerrainData(noiseSourceSeed0, noiseSourceSeed1, \
                       terrainHeight, \
                       minP, maxP):
    """
    Returns voxelData which represents the voxel terrain values for the
    points between minP and maxP. The chunk is translated so that
    voxelData[0,0,0] corresponds to (minX, minY, minZ).
    The size of the chunk is unscaled so that, for example, the width of
    the chunk is equal to maxP-minP. Ditto for the other major axii.
    """
    print "Generating terrain for chunk: minP=%r ; maxP=%r" % (minP, maxP)
    minX, minY, minZ = minP.x, minP.y, minP.z
    maxX, maxY, maxZ = maxP.x, maxP.y, maxP.z

    # Must create PerlinNoise here as Objective-C objects cannot be pickled and
    # send to other processes with IPC for use in multiprocessing.
    noiseSource0 = PerlinNoise(randomseed=noiseSourceSeed0)
    noiseSource1 = PerlinNoise(randomseed=noiseSourceSeed1)

    voxelData = numpy.arange((maxX-minX)*(maxY-minY)*(maxZ-minZ))
    voxelData = voxelData.reshape(maxX-minX, maxY-minY, maxZ-minZ)

    for x,y,z in itertools.product(range(minX, maxX),
                                   range(minY, maxY),
                                   range(minZ, maxZ)):
        # p is in world-space, not chunk-space
        p = Vector3(float(x), float(y), float(z))
        g = isGround(terrainHeight, noiseSource0, noiseSource1, p)
        voxelData[x - minX, y - minY, z - minZ] = g

    return voxelData


def generateGeometry(voxelData, minP, maxP):
    """Generate one gigantic batch containing all polygon data.
    Many of the faces are hidden, so there is room for improvement here.
    voxelData - Represents the voxel terrain values for the points between
                minP and maxP. The chunk is translated so that
                voxelData[0,0,0] corresponds to (minX, minY, minZ).
                The size of the chunk is unscaled so that, for example, the
                width of the chunk is equal to maxP-minP. Ditto for the
                other major axii.
    """
    L = 0.5 # Half-length of each block along each of its sides.
    minX, minY, minZ = minP.x, minP.y, minP.z
    maxX, maxY, maxZ = maxP.x, maxP.y, maxP.z

    verts = []
    norms = []

    for x,y,z in itertools.product(range(minX, maxX),
                                   range(minY, maxY),
                                   range(minZ, maxZ)):
        if not voxelData[x-minX, y-minY, z-minZ]:
            continue

        # Top Face
        if not (y+1<maxY and voxelData[x-minX, y-minY+1, z-minZ]):
            verts.extend([x-L, y+L, z+L,  x+L, y+L, z-L,  x-L, y+L, z-L,
                          x-L, y+L, z+L,  x+L, y+L, z+L,  x+L, y+L, z-L])
            norms.extend([  0,  +1,   0,    0,  +1,   0,    0,  +1,   0,
                            0,  +1,   0,    0,  +1,   0,    0,  +1,   0])

        # Bottom Face
        if not (y-1>=minY and voxelData[x-minX, y-minY-1, z-minZ]):
            verts.extend([x-L, y-L, z-L,  x+L, y-L, z-L,  x-L, y-L, z+L,
                          x+L, y-L, z-L,  x+L, y-L, z+L,  x-L, y-L, z+L])
            norms.extend([  0,  -1,   0,      0, -1,  0,    0,  -1,   0,
                            0,  -1,   0,      0, -1,  0,    0,  -1,   0])

        # Front Face
        if not (z+1<maxZ and voxelData[x-minX, y-minY, z-minZ+1]):
            verts.extend([x-L, y-L, z+L,  x+L, y+L, z+L,  x-L, y+L, z+L,
                          x-L, y-L, z+L,  x+L, y-L, z+L,  x+L, y+L, z+L])
            norms.extend([  0,   0,  +1,    0,   0,  +1,    0,   0,  +1,
                            0,   0,  +1,    0,   0,  +1,    0,   0,  +1])

        # Back Face
        if not (z-1>=minZ and voxelData[x-minX, y-minY, z-minZ-1]):
            verts.extend([x-L, y+L, z-L,  x+L, y+L, z-L,  x-L, y-L, z-L,
                          x+L, y+L, z-L,  x+L, y-L, z-L,  x-L, y-L, z-L])
            norms.extend([  0,   0,  -1,    0,   0,  -1,    0,   0,  -1,
                            0,   0,  -1,    0,   0,  -1,    0,   0,  -1])

        # Right Face
        if not (x+1<maxX and voxelData[x-minX+1, y-minY, z-minZ]):
            verts.extend([x+L, y+L, z-L,  x+L, y+L, z+L,  x+L, y-L, z+L,
                          x+L, y-L, z-L,  x+L, y+L, z-L,  x+L, y-L, z+L])
            norms.extend([ +1,   0,   0,   +1,   0,   0,   +1,   0,   0,
                           +1,   0,   0,   +1,   0,   0,   +1,   0,   0])

        # Left Face
        if not (x-1>=minX and voxelData[x-minX-1, y-minY, z-minZ]):
            verts.extend([x-L, y-L, z+L,  x-L, y+L, z+L,  x-L, y+L, z-L,
                          x-L, y-L, z+L,  x-L, y+L, z-L,  x-L, y-L, z-L])
            norms.extend([ -1,   0,   0,   -1,   0,   0,   -1,   0,   0,
                           -1,   0,   0,   -1,   0,   0,   -1,   0,   0])

    return verts, norms


def workerProcess(args):
    """Run by worker processes for the terrain generator to compute terrain for
    a single chunk.
    """
    # Unpack arguments
    assert len(args)==5
    noiseSourceSeed0 = args[0]
    noiseSourceSeed1 = args[1]
    terrainHeight = args[2]
    minP = args[3]
    maxP = args[4]

    # Generate terrain and geometry.
    voxelData = computeTerrainData(noiseSourceSeed0, noiseSourceSeed1,
                                   terrainHeight,
                                   minP, maxP)
    verts, norms = generateGeometry(voxelData, minP, maxP)

    # Return value will be used as parameters to Chunk.__init__
    return [voxelData, verts, norms, minP, maxP]


def generateWorkBatches(stepX, stepZ, \
                        terrainWidth, terrainHeight, terrainDepth, \
                        noiseSourceSeed0, noiseSourceSeed1):
    assert terrainWidth % stepX == 0
    assert terrainDepth % stepZ == 0
    args = []
    for x in xrange(0, terrainWidth, stepX):
        for z in xrange(0, terrainDepth, stepZ):
            # The point here is to pack together arguments
            # to the terrainWorker processes.
            args.append([noiseSourceSeed0,
                         noiseSourceSeed1,
                         terrainHeight,
                         Vector3(x, 0, z),
                         Vector3(x+stepX, terrainHeight, z+stepZ)])
    return args


def generate(w, h, d, randomseed):
    "Generates and returns terrain data."
    noiseSourceSeed0 = randomseed
    noiseSourceSeed1 = randomseed + 1
    pool = multiprocessing.Pool()
    args = generateWorkBatches(16, 16,
                               w, h, d,
                               noiseSourceSeed0, noiseSourceSeed1)
    terrainData = pool.map(workerProcess, args)
    return terrainData
