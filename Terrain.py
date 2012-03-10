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

from pyglet.gl import *
from ctypes import pointer, sizeof
import os
import itertools
import pickle
import numpy
import time
import math
import functools
import multiprocessing, logging
from pnoise import PerlinNoise
from math3D import Vector3, Quaternion, Frustum
import math3D


logger = multiprocessing.log_to_stderr(logging.INFO)


class memoized(object):
   """Decorator that caches a function's return value each time it is called.
   If called later with the same arguments, the cached value is returned, and
   not re-evaluated.
   <http://wiki.python.org/moin/PythonDecoratorLibrary>
   """
   def __init__(self, func):
      self.func = func
      self.cache = {}
   def __call__(self, *args):
      try:
         return self.cache[args]
      except KeyError:
         value = self.func(*args)
         self.cache[args] = value
         return value
      except TypeError:
         # uncachable -- for instance, passing a list as an argument.
         # Better to not cache than to blow up entirely.
         return self.func(*args)
   def __repr__(self):
      """Return the function's docstring."""
      return self.func.__doc__
   def __get__(self, obj, objtype):
      """Support instance methods."""
      return functools.partial(self.__call__, obj)


def asyncGenerateTerrain(seed, terrainHeight, minP, maxP):
    logger.debug("Generating new chunk data at %s" % str(minP))
    voxelData = Chunk.computeTerrainData(seed, terrainHeight, minP, maxP)
    verts, norms = Chunk.generateGeometry(voxelData, minP, maxP)
    return voxelData, verts, norms


def asyncLoadTerrain(folder, chunkID):
    fn = computeChunkFileName(folder, chunkID)
    if not os.path.exists(fn):
        raise Exception("File does not exist: %s" % fn)

    logger.debug("Loading chunk from disk: %s" % fn)
    onDiskFormat = pickle.load(open(fn, "rb"))

    if not len(onDiskFormat) == 4:
        raise Exception("On disk chunk format is totally unrecognized.")

    if not onDiskFormat[0] == "magic":
        raise Exception("Chunk uses unsupported format version \"%r\"." % onDiskFormat[0])

    voxelData = onDiskFormat[1]
    minP = onDiskFormat[2]
    maxP = onDiskFormat[3]

    verts, norms = Chunk.generateGeometry(voxelData, minP, maxP)

    return voxelData, verts, norms


def saveChunkToDiskWorker(folder, voxelData, minP, maxP):
    assert voxelData is not None
    assert minP is not None
    assert maxP is not None

    fn = computeChunkFileName(folder, computeChunkIDFromMinP(minP))
    logger.debug("Saving chunk to disk: %s" % fn)
    onDiskFormat = ["magic", voxelData, minP, maxP]
    pickle.dump(onDiskFormat, open(fn, "wb"))

    return True


class Chunk:
    "Chunk of terrain and associated geometry."
    sizeX = 32
    sizeY = 64
    sizeZ = 32

    def __init__(self):
        self.minP = Vector3(0,0,0)
        self.maxP = Vector3(0,0,0)
        self.boxVertices = math3D.getBoxVertices(self.minP, self.maxP)
        self.voxelData = None
        self.numTrianglesInBatch = 0
        self.vbo_verts = GLuint(0)
        self.vbo_norms = GLuint(0)
        self.pool = None
        self.dirty = True
        self.terrainTaskResult = None
        self.saveTaskResult = None

        # Lock to protect terrain data (voxelData, verts, and norms)
        self.terrainDataLock = multiprocessing.Lock()


    def __repr__(self):
        return "<Chunk %s>" % computeChunkID(self.minP)


    def __str__(self):
        return "<Chunk %s>" % computeChunkID(self.minP)


    def isVisible(self, frustum):
        return frustum.boxInFrustum(self.boxVertices) != Frustum.OUTSIDE


    @classmethod
    def fromProceduralGeneration(cls, minP, maxP, terrainHeight, seed,
                                 pool, folder):
        chunk = Chunk()
        chunk.minP = minP # extents of the chunk in world-space
        chunk.maxP = maxP #   "
        chunk.boxVertices = math3D.getBoxVertices(chunk.minP, chunk.maxP)
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)
        chunk.voxelData = None
        chunk.numTrianglesInBatch = 0
        chunk.verts = None
        chunk.norms = None
        chunk.pool = pool
        chunk.folder = folder

        # Spin off a task to generate terrain and geometry.
        # Chunk will have no terrain or geometry until this has finished.
        chunk.terrainTaskResult = pool.apply_async(asyncGenerateTerrain,
            [seed, terrainHeight, minP, maxP],
            callback=lambda r: chunk._callbackTerrainTaskHasFinished(r))

        return chunk


    def setNotDirty(self):
        """Called by the async save operation when complete to mark the chunk
        as not being dirty.
        """
        self.dirty = False


    def _callbackTerrainTaskHasFinished(self, results):
        "Callback for when the terrain generating/loading task is finished."
        with self.terrainDataLock:
            self.terrainTaskResult = None
            self.voxelData, self.verts, self.norms = results
            assert self.voxelData is not None
            assert self.verts is not None
            assert self.norms is not None
            assert len(self.verts)%3==0

            self.numTrianglesInBatch = len(self.verts)/3

            # Chunk is now dirty as it has never been saved. Spin off a task
            # to save it asynchronously.
            self.saveTaskResult = self.pool.apply_async(saveChunkToDiskWorker,
                [self.folder, self.voxelData, self.minP, self.maxP],
                callback = lambda r: self.setNotDirty())


    def maybeGenerateVBOs(self):
        "If terrain geometry is available then generate VBOs"
        with self.terrainDataLock:
            if self.verts and not self.vbo_verts:
                self.vbo_verts = self.createVertexBufferObject(self.verts)

            if self.norms and not self.vbo_norms:
                self.vbo_norms = self.createVertexBufferObject(self.norms)


    def draw(self):
        # If geometry is not available then there is nothing to do now.
        if (not self.vbo_verts) or (not self.vbo_norms):
            return

        glEnableClientState(GL_VERTEX_ARRAY)
        glEnableClientState(GL_NORMAL_ARRAY)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glVertexPointer(3, GL_FLOAT, 0, 0)

        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_norms)
        glNormalPointer(GL_FLOAT, 0, 0)

        glDrawArrays(GL_TRIANGLES, 0, self.numTrianglesInBatch)

        glDisableClientState(GL_VERTEX_ARRAY)
        glDisableClientState(GL_NORMAL_ARRAY)


    def destroy(self):
        """Destroy the chunk and free all resources consumed by it, including
        GPU memory for its vertex buffer objects.
        """
        doomed_buffers = [self.vbo_verts, self.vbo_norms]
        buffers = (GLuint * len(doomed_buffers))(*doomed_buffers)
        glDeleteBuffers(len(buffers), buffers)

        self.vbo_verts = GLuint(0) # 0 is an invalid handle
        self.vbo_norms = GLuint(0) # 0 is an invalid handle

        self.voxelData = None


    def saveToDisk(self, folder, block=False):
        """Saves the chunk if possible. Returns False is the chunk cannot be
        saved for some reason such as terrain generation being in progress at
        the moment. Always saves the chunk synchronously.
        block - If True then wait for terrain generation to complete and
                save the terrain to disk. May take longer.
        """
        if not self.dirty:
            return True # nothing to do

        # First, make sure we can save right now. Maybe block to wait for
        # terrain results to come back.
        if self.terrainTaskResult and not self.terrainTaskResult.ready():
            if block:
                self.terrainTaskResult.wait()
            else:
                return False # would block, so bail out

        with self.terrainDataLock:
            if self.saveTaskResult and not self.saveTaskResult.ready():
                self.saveTaskResult.wait() # wait for save task to finish
            else:
                # Save synchronously.
                saveChunkToDiskWorker(self.folder, self.voxelData,
                                      self.minP, self.maxP)

            self.dirty = False
            self.saveTaskResult = None

        return True


    @staticmethod
    def computeChunkMinP(p):
        return Vector3(int(p.x / Chunk.sizeX) * Chunk.sizeX,
                       int(p.y / Chunk.sizeY) * Chunk.sizeY,
                       int(p.z / Chunk.sizeZ) * Chunk.sizeZ)


    @classmethod
    def loadFromDisk(cls, folder, chunkID, minP, maxP, pool):
        chunk = Chunk()
        chunk.minP = minP # extents of the chunk in world-space
        chunk.maxP = maxP #   "
        chunk.boxVertices = math3D.getBoxVertices(chunk.minP, chunk.maxP)
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)
        chunk.voxelData = None
        chunk.numTrianglesInBatch = 0
        chunk.verts = None
        chunk.norms = None
        chunk.dirty = False
        chunk.pool = pool
        chunk.folder = folder

        # Spin off a task to load the terrain.
        chunk.terrainTaskResult = pool.apply_async(asyncLoadTerrain,
            [folder, chunkID],
            callback=lambda r: chunk._callbackTerrainTaskHasFinished(r))

        return chunk


    @staticmethod
    def createVertexBufferObject(verts):
        assert verts
        vbo_verts = GLuint()
        glGenBuffers(1, pointer(vbo_verts))
        data = (GLfloat * len(verts))(*verts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        return vbo_verts


    @staticmethod
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


    @staticmethod
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
        return Chunk.groundGradient(terrainHeight, pPrime) <= 0


    @staticmethod
    def computeTerrainData(seed, terrainHeight, minP, maxP):
        """
        Returns voxelData which represents the voxel terrain values for the
        points between minP and maxP. The chunk is translated so that
        voxelData[0,0,0] corresponds to (minX, minY, minZ).
        The size of the chunk is unscaled so that, for example, the width of
        the chunk is equal to maxP-minP. Ditto for the other major axii.
        """
        logger.debug("Generating terrain for chunk: %r" % minP)
        minX, minY, minZ = int(minP.x), int(minP.y), int(minP.z)
        maxX, maxY, maxZ = int(maxP.x), int(maxP.y), int(maxP.z)

        # Must create PerlinNoise here as Objective-C objects cannot be pickled and
        # send to other processes with IPC for use in multiprocessing.
        noiseSource0 = PerlinNoise(randomseed=seed)
        noiseSource1 = PerlinNoise(randomseed=seed+1)

        voxelData = numpy.arange((maxX-minX)*(maxY-minY)*(maxZ-minZ))
        voxelData = voxelData.reshape(maxX-minX, maxY-minY, maxZ-minZ)

        for x,y,z in itertools.product(range(minX, maxX),
                                       range(minY, maxY),
                                       range(minZ, maxZ)):
            # p is in world-space, not chunk-space
            p = Vector3(float(x), float(y), float(z))
            g = Chunk.isGround(terrainHeight, noiseSource0, noiseSource1, p)
            voxelData[x - minX, y - minY, z - minZ] = g

        return voxelData


    @staticmethod
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
        minX, minY, minZ = int(minP.x), int(minP.y), int(minP.z)
        maxX, maxY, maxZ = int(maxP.x), int(maxP.y), int(maxP.z)

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


@memoized
def computeChunkFileName(folder, chunkID):
    return os.path.join(folder, str(chunkID))


@memoized
def computeChunkIDFromMinP(minP):
    t = (minP.x, minP.y, minP.z)
    return hash(t)


def computeChunkID(p):
    """Given an arbitrary point in space, retrieve the ID of the chunk
    which resides there.
    """
    return computeChunkIDFromMinP(Chunk.computeChunkMinP(p))


class ChunkStore:
    activeRegionSizeX = 256 # These are the dimensions of the active region.
    activeRegionSizeY = 64
    activeRegionSizeZ = 256

    # The number of chunks which will be in the active region.
    numActiveChunks = activeRegionSizeX/Chunk.sizeX * \
                      activeRegionSizeY/Chunk.sizeY * \
                      activeRegionSizeZ/Chunk.sizeZ

    # The maximum number of chunks to keep in the cache at once.
    cacheSizeLimit = max(numActiveChunks, 512) # TODO: What size is best?


    def __init__(self, seed):
        logger.info("Initializing ChunkStore")
        self.chunks = {}
        self.pool = multiprocessing.Pool(processes=8)
        self.seed = seed
        self.cameraPos = Vector3(0,0,0)
        self.cameraRot = Quaternion(0,0,0,1)
        self.cameraFrustum = Frustum()
        self.activeChunks = [None] * ChunkStore.numActiveChunks
        self.visibleChunks = []
        self.saveFolder = "world_%s" % str(seed)
        os.system("/bin/mkdir -p \'%s\'" % self.saveFolder)


    def _evictChunk(self, chunkID):
        "Evict a chunk from the cache."
        if chunkID not in self.chunks:
            raise Exception("Chunk is not in the cache.")

        # Sync chunks in case we evict a dirty chunk or a chunk with tasks
        # in flight. Must do this synchronously.
        self.chunks[chunkID].saveToDisk(self.saveFolder, True)

        # Deallocate VBOs. (VRAM)
        self.chunks[chunkID].destroy()

        del self.chunks[chunkID]


    def _performOneEviction(self):
        "Evict one chunk from the cache (not an active chunk)."
        for chunk in self.chunks.values():
            if chunk not in self.activeChunks:
                toEvictID = computeChunkIDFromMinP(chunk.minP)
                self._evictChunk(toEvictID)
                logger.info("Evicted a chunk. Cache now contains " \
                            "%d/%d chunks" % (len(self.chunks),
                                              self.cacheSizeLimit))
                assert len(self.chunks) < self.cacheSizeLimit
                return


    def _performEvictionIfNecessary(self):
        "Evict chunks until the cache is no longer larger than the max size."
        while len(self.chunks) >= ChunkStore.cacheSizeLimit:
            self._performOneEviction()


    def _generateOrLoadChunk(self, chunkID, minP):
        "Load the chunk from disk or generate it here and now."
        chunk = None
        maxP = minP.add(Vector3(Chunk.sizeX, Chunk.sizeY, Chunk.sizeZ))

        if os.path.exists(computeChunkFileName(self.saveFolder,chunkID)):
            chunk = Chunk.loadFromDisk(self.saveFolder,
                                       chunkID,
                                       minP, maxP,
                                       self.pool)
        else:
            chunk =  Chunk.fromProceduralGeneration(minP, maxP,
                                                    self.activeRegionSizeY,
                                                    self.seed,
                                                    self.pool,
                                                    self.saveFolder)

        return chunk


    def getChunk(self, p):
        """Retrieves a chunk of the game world at an arbritrary point in space.
        """
        chunkID = computeChunkID(p)
        chunk = None
        try:
            chunk = self.chunks[chunkID]
        except KeyError:
            self._performEvictionIfNecessary()
            minP = Chunk.computeChunkMinP(p)
            chunk = self._generateOrLoadChunk(chunkID, minP)
            self.chunks[chunkID] = chunk
            logger.info("Added chunk to the cache. It now contains " \
                        "%d/%d chunks" % (len(self.chunks),
                                          self.cacheSizeLimit))

        return chunk


    def sync(self, bl=False):
        "Ensure dirty chunks are saved to disk."
        # By default, don't block. If a chunk hasn't generated data yet then
        # we'll regenerate the chunk in the next session. Nothing is lost.
        map(lambda c: c.saveToDisk(self.saveFolder, bl), self.chunks.values())
        logger.info("sync'd chunks to disk.")


    def update(self, dt):
        map(Chunk.maybeGenerateVBOs, self.visibleChunks)


    def _updateInternalActiveAndVisibleChunksCache(self):
        """Update the internal cache of active chunks. Call when the camera
        changes to update internal book keeping about which chunks are active.
        """
        activeChunks = self.activeChunks
        visibleChunks = []
        cx = self.cameraPos.x
        cy = self.cameraPos.y
        cz = self.cameraPos.z
        W = self.activeRegionSizeX/2
        H = self.activeRegionSizeY/2
        D = self.activeRegionSizeZ/2
        xs = numpy.arange(cx - W, cx + W, Chunk.sizeX)
        ys = numpy.arange(cy - H, cy + H, Chunk.sizeY)
        zs = numpy.arange(cz - D, cz + D, Chunk.sizeZ)
        i = 0
        for x,y,z in itertools.product(xs, ys, zs):
            chunk = self.getChunk(Vector3(x, y, z))
            activeChunks[i] = chunk

            if chunk.isVisible(self.cameraFrustum):
                visibleChunks.append(chunk)

            i += 1
            if i >= len(activeChunks):
                break

        self.visibleChunks = visibleChunks


    def drawVisibleChunks(self):
        "Draw all chunks which are currently visible."
        map(Chunk.draw, self.visibleChunks)


    def setCamera(self, p, r, fr):
        self.cameraPos = p
        self.cameraRot = r
        self.cameraFrustum = fr
        self._updateInternalActiveAndVisibleChunksCache()


if __name__ == "__main__":
    cameraPos = Vector3(0,0,0)
    cameraRot = Quaternion.fromAxisAngle(Vector3(0,1,0), 0)
    cameraSpeed = 5.0
    cameraRotSpeed = 1.0
    cameraFrustum = Frustum()
    cameraEye = Vector3(0,0,0)
    cameraCenter = Vector3(0,0,0)
    cameraUp = Vector3(0,0,0)

    cameraEye, cameraCenter, cameraUp = \
        math3D.getCameraEyeCenterUp(cameraPos, cameraRot)
    cameraFrustum.setCamInternals(65, 640.0/480.0, .1, 1000)
    cameraFrustum.setCamDef(cameraEye, cameraCenter, cameraUp)

    chunkStore = ChunkStore(0)
    chunkStore.setCamera(cameraPos, cameraRot, cameraFrustum)

    print "A chunk at the origin:", chunkStore.getChunk(cameraPos)
    chunkStore.prefetchChunk(Vector3(0.0, 0.0, 0.0))
    print "Active Chunks:", chunkStore.activeChunks
    print "Visible Chunks:", chunkStore.visibleChunks

    chunkStore.sync()
