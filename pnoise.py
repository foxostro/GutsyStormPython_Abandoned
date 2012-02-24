#!/usr/bin/env python
# vim: et ts=4
import math
#import time
import random

# Generate the permutation table the Perlin noise generator.
random.seed(0) # time.time())
p = range(0,256) * 2
random.shuffle(p)


def lerp(t, a, b):
    return a + t * (b - a)


def fade(t):
    return t * t * t * (t * (t * 6 - 15) + 10)


def grad(hash, x, y, z):
    h = hash & 15
    if h < 8:
        u = x
    else:
        u = y
    if h < 4:
        v = y
    elif h == 12 or h == 14:
        v = x
    else:
        v = z
    if h & 1 != 0:
        u = -u
    if h & 2 != 0:
        v = -v
    return u + v


def perlinNoise(x, y, z):
    "Computes a perlin noise value at the specified position"
    global p

    X = int(math.floor(x)) & 255
    Y = int(math.floor(y)) & 255
    Z = int(math.floor(z)) & 255
    x -= math.floor(x)
    y -= math.floor(y)
    z -= math.floor(z)

    u = fade(x)
    v = fade(y)
    w = fade(z)

    A =  p[X] + Y
    AA = p[A] + Z
    AB = p[A + 1] + Z
    B =  p[X + 1] + Y
    BA = p[B] + Z
    BB = p[B + 1] + Z

    pAA = p[AA]
    pAB = p[AB]
    pBA = p[BA]
    pBB = p[BB]
    pAA1 = p[AA + 1]
    pBA1 = p[BA + 1]
    pAB1 = p[AB + 1]
    pBB1 = p[BB + 1]

    gradAA =  grad(pAA, x,   y,   z)
    gradBA =  grad(pBA, x-1, y,   z)
    gradAB =  grad(pAB, x,   y-1, z)
    gradBB =  grad(pBB, x-1, y-1, z)
    gradAA1 = grad(pAA1,x,   y,   z-1)
    gradBA1 = grad(pBA1,x-1, y,   z-1)
    gradAB1 = grad(pAB1,x,   y-1, z-1)
    gradBB1 = grad(pBB1,x-1, y-1, z-1)

    return lerp(w,
                lerp(v, lerp(u, gradAA, gradBA), lerp(u, gradAB, gradBB)),
                lerp(v, lerp(u, gradAA1,gradBA1),lerp(u, gradAB1,gradBB1)))


def perlinNoiseWithMultipleOctaves(x, y, z, numOctaves):
    """Computes a perlin noise value at the specified position with multiple
    octaves of noise layered over top of it.
    """
    c = 0
    for i in range(1, numOctaves+1):
        p = pow(2, i)
        c += perlinNoise(x * p, y * p, z * p) / p
    return c
