#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
from pyglet.gl import *
from ctypes import pointer, sizeof
import itertools
import pickle


class Chunk:
	"Chunk of terrain and associated geometry."

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
        minX, minY, minZ = minP
        maxX, maxY, maxZ = maxP

        def getVoxelData(x, y, z):
            if x>=minX and x<maxX and \
               y>=minY and y<maxY and \
               z>=minZ and z<maxZ:
                return voxelData[x-minX, y-minY, z-minZ]
            else:
                return False

        verts = []
        norms = []

        for x,y,z in itertools.product(range(minX, maxX),
                                       range(minY, maxY),
                                       range(minZ, maxZ)):
            if not getVoxelData(x,y,z):
                continue

            # Top Face
            if not getVoxelData(x,y+1,z):
                verts.extend([x-L, y+L, z+L,  x+L, y+L, z-L,  x-L, y+L, z-L,
                              x-L, y+L, z+L,  x+L, y+L, z+L,  x+L, y+L, z-L])
                norms.extend([  0,  +1,   0,    0,  +1,   0,    0,  +1,   0,
                                0,  +1,   0,    0,  +1,   0,    0,  +1,   0])

            # Bottom Face
            if not getVoxelData(x,y-1,z):
                verts.extend([x-L, y-L, z-L,  x+L, y-L, z-L,  x-L, y-L, z+L,
                              x+L, y-L, z-L,  x+L, y-L, z+L,  x-L, y-L, z+L])
                norms.extend([  0,  -1,   0,      0, -1,  0,    0,  -1,   0,
                                0,  -1,   0,      0, -1,  0,    0,  -1,   0])

            # Front Face
            if not getVoxelData(x,y,z+1):
                verts.extend([x-L, y-L, z+L,  x+L, y+L, z+L,  x-L, y+L, z+L,
                              x-L, y-L, z+L,  x+L, y-L, z+L,  x+L, y+L, z+L])
                norms.extend([  0,   0,  +1,    0,   0,  +1,    0,   0,  +1,
                                0,   0,  +1,    0,   0,  +1,    0,   0,  +1])

            # Back Face
            if not getVoxelData(x,y,z-1):
                verts.extend([x-L, y+L, z-L,  x+L, y+L, z-L,  x-L, y-L, z-L,
                              x+L, y+L, z-L,  x+L, y-L, z-L,  x-L, y-L, z-L])
                norms.extend([  0,   0,  -1,    0,   0,  -1,    0,   0,  -1,
                                0,   0,  -1,    0,   0,  -1,    0,   0,  -1])

            # Right Face
            if not getVoxelData(x+1,y,z):
                verts.extend([x+L, y+L, z-L,  x+L, y+L, z+L,  x+L, y-L, z+L,
                              x+L, y-L, z-L,  x+L, y+L, z-L,  x+L, y-L, z+L])
                norms.extend([ +1,   0,   0,   +1,   0,   0,   +1,   0,   0,
                               +1,   0,   0,   +1,   0,   0,   +1,   0,   0])

            # Left Face
            if not getVoxelData(x-1,y,z):
                verts.extend([x-L, y-L, z+L,  x-L, y+L, z+L,  x-L, y+L, z-L,
                              x-L, y-L, z+L,  x-L, y+L, z-L,  x-L, y-L, z-L])
                norms.extend([ -1,   0,   0,   -1,   0,   0,   -1,   0,   0,
                               -1,   0,   0,   -1,   0,   0,   -1,   0,   0])

        return verts, norms


    @staticmethod
    def createVertexBufferObject(verts):
        vbo_verts = GLuint()
        glGenBuffers(1, pointer(vbo_verts))
        data = (GLfloat * len(verts))(*verts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        return vbo_verts


    def __init__(self, voxelData, minP, maxP):
		self.minP = minP # extents of the chunk in world-space
		self.maxP = maxP #   "
		self.voxelData = voxelData # the actual terrain data for the chunk

		# Generate geometry for the chunk.
        verts, norms = self.generateGeometry(voxelData, minP, maxP)
        self.numTrianglesInBatch = len(verts)/3
        self.vbo_verts = self.createVertexBufferObject(verts)
        del verts
        self.vbo_norms = self.createVertexBufferObject(norms)
        del norms


    def bind(self):
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_verts)
        glVertexPointer(3, GL_FLOAT, 0, 0)
        glBindBuffer(GL_ARRAY_BUFFER, self.vbo_norms)
        glNormalPointer(GL_FLOAT, 0, 0)


    def draw(self):
        glDrawArrays(GL_TRIANGLES, 0, self.numTrianglesInBatch)


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
		return Chunk(voxelData, minP, maxP)
