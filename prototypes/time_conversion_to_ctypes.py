#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python

import time
import array
import ctypes

numElements = 500000
t = ctypes.c_float * numElements
dataAsList = [float(1)] * numElements
dataAsArray = array.array('f', dataAsList)

# Measure time to convert from list to c_float array.
a = time.time()
dataAsCtypesArray = t(*dataAsList)
b = time.time()
print "Time to convert list to c_float array: %.3f" % float(b-a)

# Measure time to convert from an array to c_float array.
a = time.time()
dataAsCtypesArray = t(*dataAsArray)
b = time.time()
print "Time to convert buffer to c_float array: %.3f" % float(b-a)

# Measure time to convert from a buffer to c_float array.
a = time.time()
dataAsCtypesArray = t.from_buffer(dataAsArray)
b = time.time()
print "Time to convert buffer to c_float array: %.3f" % float(b-a)
print len(dataAsCtypesArray)
print numElements
