#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"Routines which operate on 3D vectors."

from collections import namedtuple
import math


vec = namedtuple('vec', 'x y z')
eps = 1e-8


def length(v):
    return math.sqrt(v.x*v.x + v.y*v.y + v.z*v.z)


def normalize(v):
    "Returns a normalized vector (unit length)"
    mag2 = v.x*v.x + v.y*v.y + v.z*v.z
    if abs(mag2) > eps and abs(mag2 - 1.0) > eps:
        mag = math.sqrt(mag2)
        return vec(v.x / mag, v.y / mag, v.z / mag)
    else:
        return v


def add(v1, v2):
    return vec(v1.x+v2.x, v1.y+v2.y, v1.z+v2.z)


def sub(v1, v2):
    return vec(v1.x-v2.x, v1.y-v2.y, v1.z-v2.z)


def isEqual(v1, v2):
    return abs(v1.x-v2.x)<eps and \
           abs(v1.y-v2.y)<eps and \
           abs(v1.z-v2.z)<eps
