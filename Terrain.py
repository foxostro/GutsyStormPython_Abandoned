#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
from pyglet.gl import *
from ctypes import pointer, sizeof
import os
import itertools
import pickle
import numpy
import time
import math
import multiprocessing, logging
from pnoise import PerlinNoise
from math3D import Vector3, Quaternion


WOULD_BLOCK = 0
NOW_AVAILABLE = 1
ERROR = 2
logger=multiprocessing.log_to_stderr(logging.INFO)


def asyncGenerateTerrain(seed, terrainHeight, minP, maxP):
    voxelData = Chunk.computeTerrainData(seed, terrainHeight, minP, maxP)
    verts, norms = Chunk.generateGeometry(voxelData, minP, maxP)
    return voxelData, verts, norms


def asyncLoadTerrain(folder, chunkID):
    fn = Chunk.computeChunkFileName(folder, chunkID)
    if not os.path.exists(fn):
        raise Exception("File does not exist: %s" % fn)

    logger.info("Loading chunk from disk: %r" % chunkID)
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


class Chunk:
    "Chunk of terrain and associated geometry."
    sizeX = 16
    sizeY = 16
    sizeZ = 16

    def __init__(self):
        self.minP = Vector3(0,0,0)
        self.maxP = Vector3(0,0,0)
        self.voxelData = None
        self.numTrianglesInBatch = 0
        self.vbo_verts = GLuint(0)
        self.vbo_norms = GLuint(0)
        self.asyncTerrainResult = None
        self.dirty = True


    def __repr__(self):
        return "<Chunk %s>" % Chunk.computeChunkID(self.minP)


    def __str__(self):
        return "<Chunk %s>" % Chunk.computeChunkID(self.minP)


    @classmethod
    def fromProceduralGeneration(cls, minP, maxP, terrainHeight, seed, pool):
        chunk = Chunk()
        chunk.minP = minP # extents of the chunk in world-space
        chunk.maxP = maxP #   "
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)
        chunk.voxelData = None
        chunk.numTrianglesInBatch = 0
        chunk.verts = None
        chunk.norms = None

        # Spin off a task to generate terrain and geometry.
        # Chunk will have no terrain or geometry until this has finished.
        chunk.asyncTerrainResult = \
            pool.apply_async(asyncGenerateTerrain,
                             [seed, terrainHeight, minP, maxP])

        return chunk


    def updateTerrainFromAsyncGenResults(self, block=False):
        if not self.asyncTerrainResult:
            return NOW_AVAILABLE

        # We may want to block and wait for results now.
        if block and not self.asyncTerrainResult.ready():
            logger.info("blocking to wait for terrain data")
            self.asyncTerrainResult.wait(30)
            if not self.asyncTerrainResult.ready():
                raise Exception("Blocked for 30s waiting for terrain data " \
                                "and never got it. It's probably not coming." \
                                " Bailing out.")

        # If results are not ready then bail out. 
        if not self.asyncTerrainResult.ready():
            return WOULD_BLOCK

        if not self.asyncTerrainResult.successful():
            logger.error("Terrain generation failed for chunk %r" % self.minP)
            self.asyncTerrainResult = None
            return ERROR

        self.voxelData, self.verts, self.norms = self.asyncTerrainResult.get()

        self.asyncTerrainResult = None

        assert self.voxelData is not None
        assert self.verts is not None
        assert self.norms is not None

        assert len(self.verts)%3==0
        self.numTrianglesInBatch = len(self.verts)/3

        return NOW_AVAILABLE


    def maybeGenerateVBOs(self):
        "Check up on the async terrain generation task and maybe generate VBOs"
        didSomething = False

        if NOW_AVAILABLE == self.updateTerrainFromAsyncGenResults(False):
            if self.verts and not self.vbo_verts:
                self.vbo_verts = self.createVertexBufferObject(self.verts)
                didSomething = True

            if self.norms and not self.vbo_norms:
                self.vbo_norms = self.createVertexBufferObject(self.norms)
                didSomething = True

        return didSomething


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


    @staticmethod
    def computeChunkFileName(folder, chunkID):
        return os.path.join(folder, chunkID)


    def saveToDisk(self, folder, block=False):
        """Saves the chunk if possible. Returns False is the chunk cannot be
        saved for some reason such as terrain generation being in progress at
        the moment.
        block - If True then wait for terrain generation to complete and
                save the terrain to disk. May take longer.
        """
        # First, make sure we can save right now. Maybe block to wait for
        # terrain results to come back.
        if self.updateTerrainFromAsyncGenResults(block) != NOW_AVAILABLE:
            return False

        if self.dirty:
            assert self.voxelData is not None
            assert self.minP is not None
            assert self.maxP is not None

            chunkID = Chunk.computeChunkID(self.minP)
            logger.info("Saving chunk to disk: %r" % chunkID)
            fn = Chunk.computeChunkFileName(folder, chunkID)
            onDiskFormat = ["magic", self.voxelData, self.minP, self.maxP]
            pickle.dump(onDiskFormat, open(fn, "wb"))
            self.dirty = False
            return True


    @classmethod
    def loadFromDisk(cls, folder, chunkID, minP, maxP, pool):
        chunk = Chunk()
        chunk.minP = minP # extents of the chunk in world-space
        chunk.maxP = maxP #   "
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)
        chunk.voxelData = None
        chunk.numTrianglesInBatch = 0
        chunk.verts = None
        chunk.norms = None
        chunk.dirty = False

        # Spin off a task to load the terrain.
        chunk.asyncTerrainResult = \
            pool.apply_async(asyncLoadTerrain, [folder, chunkID])

        # Chunk will have no terrain or geometry until this has finished.
        #chunk.voxelData, chunk.verts, chunk.norms = \
        #    asyncLoadTerrain(folder, chunkID)
        #assert len(chunk.verts)%3==0
        #chunk.numTrianglesInBatch = len(chunk.verts)/3
        #chunk.asyncTerrainResult = None

        return chunk


    @staticmethod
    def computeChunkMinP(p):
        return Vector3(math.floor(p.x / Chunk.sizeX) * Chunk.sizeX,
                       math.floor(p.y / Chunk.sizeY) * Chunk.sizeY,
                       math.floor(p.z / Chunk.sizeZ) * Chunk.sizeZ)


    @staticmethod
    def computeChunkID(p):
        """Given an arbitrary point in space, retrieve the ID of the chunk
        which resides there.
        """
        q = Chunk.computeChunkMinP(p)
        return "%d_%d_%d" % (q.x, q.y, q.z)


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
        logger.info("Generating terrain for chunk: %r" % minP)
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


class ChunkStore:
    RES_X = 128 # These are the dimensions of the active region.
    RES_Y = 64
    RES_Z = 128
    chunksToSavePerFrame = 10
    chunkBufferGeneratingTimeBudget = 5.0 / 30.0
    chunkSavingTimeBudget = 1.0 / 30.0


    def __init__(self, seed):
        logger.info("Initializing ChunkStore")
        self.chunks = {}
        self.pool = multiprocessing.Pool(processes=8)
        self.seed = seed
        self.cameraPos = Vector3(0,0,0)
        self.cameraRot = Quaternion(0,0,0,1)
        self.saveFolder = "world"
        os.system("/bin/mkdir -p \'%s\'" % self.saveFolder)


    def generateOrLoadChunk(self, chunkID, minP):
        "Load the chunk from disk or generate it here and now."
        chunk = None
        maxP = minP.add(Vector3(Chunk.sizeX, Chunk.sizeY, Chunk.sizeZ))

        if os.path.exists(Chunk.computeChunkFileName(self.saveFolder,chunkID)):
            logger.info("Chunk seems to exist on disk; loading: %r" % chunkID)
            chunk = Chunk.loadFromDisk(self.saveFolder,
                                       chunkID,
                                       minP, maxP,
                                       self.pool)
        else:
            logger.info("Chunk does not exist on disk; generating: %r" % chunkID)
            chunk =  Chunk.fromProceduralGeneration(minP, maxP,
                                                    self.RES_Y,
                                                    self.seed,
                                                    self.pool)
        return chunk


    def getChunk(self, p):
        "Retrieves a chunk of the game world at an arbritrary point in space."
        chunkID = Chunk.computeChunkID(p)
        chunk = None
        try:
            chunk = self.chunks[chunkID]
        except KeyError:
            minP = Chunk.computeChunkMinP(p)
            chunk = self.generateOrLoadChunk(chunkID, minP)
            self.chunks[chunkID] = chunk

        return chunk


    def sync(self):
        "Ensure all chunks are saved to disk."
        for chunk in self.chunks.values():
            chunk.saveToDisk(self.saveFolder)
        logger.info("sync'd chunks to disk")


    def perFramePartialSync(self):
        "Save a few dirty chunks every frame to keep the game interactive."
        startTime = time.time()
        for chunk in self.chunks.values():
            if chunk.dirty:
                chunk.saveToDisk(self.saveFolder)

            if time.time() - startTime > self.chunkSavingTimeBudget:
                break


    def perfFramePartialVBOGeneration(self):
        "Generate a few VBOs per frame to keep the game interactive."
        startTime = time.time()
        for chunk in self.getVisibleChunks():
            chunk.maybeGenerateVBOs()

            if time.time() - startTime > self.chunkBufferGeneratingTimeBudget:
                break


    def update(self, dt):
        self.perFramePartialSync()
        self.perfFramePartialVBOGeneration()


    def getActiveChunks(self):
        activeChunks = []

        # Return all chunks near the camera.
        x = self.cameraPos.x - self.RES_X/2
        while x < (self.cameraPos.x + self.RES_X/2):
            y = self.cameraPos.y - self.RES_Y/2
            while y < (self.cameraPos.y + self.RES_Y/2):
                z = self.cameraPos.z - self.RES_Z/2
                while z < (self.cameraPos.z + self.RES_Z/2):
                    chunk = self.getChunk(Vector3(x, y, z))
                    activeChunks.append(chunk)
                    z += Chunk.sizeZ
                y += Chunk.sizeY
            x += Chunk.sizeX

        return activeChunks


    def getVisibleChunks(self):
        return self.getActiveChunks() # TODO: only return chunks in the camera frustum


    def setCamera(self, p, r):
        self.cameraPos = p
        self.cameraRot = r


if __name__ == "__main__":
    chunkStore = ChunkStore(0)
    assert Chunk.computeChunkID(Vector3(0.0, 0.0, 0.0)) == "0_0_0"
    print chunkStore.getChunk(Vector3(0.0, 0.0, 0.0))
    print chunkStore.getActiveChunks()
