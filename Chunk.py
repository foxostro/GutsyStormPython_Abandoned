#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
from pyglet.gl import *
from ctypes import pointer, sizeof
import itertools
import pickle
import TerrainGenerator


class Chunk:
	"Chunk of terrain and associated geometry."

    @staticmethod
    def createVertexBufferObject(verts):
        vbo_verts = GLuint()
        glGenBuffers(1, pointer(vbo_verts))
        data = (GLfloat * len(verts))(*verts)
        glBindBuffer(GL_ARRAY_BUFFER, vbo_verts)
        glBufferData(GL_ARRAY_BUFFER, sizeof(data), data, GL_STATIC_DRAW)
        return vbo_verts


    def __init__(self, voxelData, verts, norms, minP, maxP):
		self.minP = minP # extents of the chunk in world-space
		self.maxP = maxP #   "
		self.voxelData = voxelData # the actual terrain data for the chunk

        # Send geometry to the GPU.
        self.numTrianglesInBatch = len(verts)/3
        self.vbo_verts = self.createVertexBufferObject(verts)
        self.vbo_norms = self.createVertexBufferObject(norms)


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
        verts, norms = TerrainGenerator.generateGeometry(voxelData, minP, maxP)
		return Chunk(voxelData, verts, norms, minP, maxP)
