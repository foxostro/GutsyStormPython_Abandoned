#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
from pyglet.gl import *
from ctypes import pointer, sizeof
import itertools
import pickle
import numpy
import time
import math
import multiprocessing
from pnoise import PerlinNoise
from math3D import Vector3, Quaternion


class Chunk:
	"Chunk of terrain and associated geometry."
    sizeX = 16
    sizeY = 16
    sizeZ = 16

    @staticmethod
    def createVertexBufferObject(verts):
        vbo_verts = GLuint()
        glGenBuffers(1, pointer(vbo_verts))
        data = (GLfloat * len(verts))(*verts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        return vbo_verts


    def __init__(self):
		self.minP = Vector3(0,0,0)
		self.maxP = Vector3(0,0,0)
		self.voxelData = None
        self.numTrianglesInBatch = 0
        self.vbo_verts = GLuint(0)
        self.vbo_norms = GLuint(0)


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


    def __repr__(self):
        return "<Chunk %s>" % Chunk.computeChunkID(self.minP)


    def __str__(self):
        return "<Chunk %s>" % Chunk.computeChunkID(self.minP)


	@classmethod
    def fromProceduralGeneration(cls, minP, maxP, terrainHeight, seed):
        chunk = Chunk()
		chunk.minP = minP # extents of the chunk in world-space
		chunk.maxP = maxP #   "
		chunk.voxelData = chunk.computeTerrainData(seed, terrainHeight,
                                                   minP, maxP)
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)

        # Generate terrain and geometry immediately.
        # TODO: Generate asynchronously.
        verts, norms = Chunk.generateGeometry(chunk.voxelData, minP, maxP)
        chunk.numTrianglesInBatch = len(verts)/3
        chunk.verts = verts
        chunk.norms = norms

        return chunk


    def update(self, dt):
        if not self.vbo_verts:
            self.vbo_verts = self.createVertexBufferObject(self.verts)

        if not self.vbo_norms:
            self.vbo_norms = self.createVertexBufferObject(self.norms)


    def draw(self):
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


	def saveToDisk(self, fn):
		onDiskFormat = ["magic", self.voxelData, self.minP, self.maxP]
		pickle.dump(onDiskFormat, open(fn, "wb"))


	@classmethod
	def loadFromDisk(cls, fn):
		onDiskFormat = pickle.load(open(fn, "rb"))
		if not len(onDiskFormat) == 4:
			raise Exception("On disk chunk format is totally unrecognized.")
		if not onDiskFormat[0] == "magic":
			raise Exception("Chunk uses unsupported format version \"%r\"." % onDiskFormat[0])
		voxelData = onDiskFormat[1]
		minP = onDiskFormat[2]
		maxP = onDiskFormat[3]
        verts, norms = TerrainGenerator.generateGeometry(voxelData, minP, maxP)

        # Build the chunk from the data we just loaded. 
        chunk = Chunk()
        chunk.minP = minP
        chunk.maxP = maxP
        chunk.voxelData = voxelData
        chunk.numTrianglesInBatch = len(verts)/3
        chunk.verts = verts
        chunk.norms = norms
        chunk.vbo_verts = GLuint(0)
        chunk.vbo_norms = GLuint(0)
        return chunk


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
        print "Generating terrain for chunk: %r" % minP
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
    RES_X = 256 # These are the dimensions of the active region.
    RES_Y = 64
    RES_Z = 256


    def __init__(self, seed):
        self.chunks = {}
        self.seed = seed
        self.cameraPos = Vector3(0,0,0)
        self.cameraRot = Quaternion(0,0,0,1)


    def getChunk(self, p):
        "Retrieves a chunk of the game world at an arbritrary point in space."
        chunkID = Chunk.computeChunkID(p)
        try:
            chunk = self.chunks[chunkID]
        except KeyError:
            # Chunk has not been created yet, so create it now.
            minP = Chunk.computeChunkMinP(p)
            maxP = minP.add(Vector3(Chunk.sizeX, Chunk.sizeY, Chunk.sizeZ))
            chunk =  Chunk.fromProceduralGeneration(minP, maxP,
                                                    self.RES_Y, self.seed)
            self.chunks[chunkID] = chunk

        return chunk


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
        return self.chunks.values() # TODO: only return chunks in the camera frustum


    def setCamera(self, p, r):
        self.cameraPos = p
        self.cameraRot = r


if __name__ == "__main__":
    chunkStore = ChunkStore(0)
    assert Chunk.computeChunkID(Vector3(0.0, 0.0, 0.0)) == "0_0_0"
    print chunkStore.getChunk(Vector3(0.0, 0.0, 0.0))
    print chunkStore.getActiveChunks()
