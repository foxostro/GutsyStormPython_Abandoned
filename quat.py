#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"Routines which operate on quaternions."

import math
import vec


eps = 1e-8
x = 0
y = 1
z = 2
w = 3


def quat(x, y, z, w):
    return (x, y, z, w)


def normalize(q):
    "Returns a normalized quaternion (unit length)"
    mag2 = q[x]*q[x] + q[y]*q[y] + q[z]*q[z] + q[w]*q[w]
    if abs(mag2) > eps and abs(mag2 - 1.0) > eps:
        mag = math.sqrt(mag2)
        return quat(q[x] / mag, q[y] / mag, q[z] / mag, q[w] / mag)
    else:
        return q


def conjugate(q):
    return quat(-q[x], -q[y], -q[z], q[w])


def mulByQuat(q, r):
    "Multiplying q1 with q2 applies the rotation q2 to q1"
    x1, y1, z1, w1 = q
    x2, y2, z2, w2 = r

    w3 = w1*w2 - x1*x2 - y1*y2 - z1*z2
    x3 = w1*x2 + x1*w2 + y1*z2 - z1*y2
    y3 = w1*y2 - x1*z2 + y1*w2 + z1*x2
    z3 = w1*z2 + x1*y2 - y1*x2 + z1*w2

    return quat(x3, y3, z3, w3)


def mulByVec(q, v):
    """Multiplying a quaternion q with a vector v applies the q-rotation
    to v"""
    q2 = mulByQuat(mulByQuat(q, quat(v[x], v[y], v[z], 0.0)), conjugate(q))
    return vec.vec(q2[x], q2[y], q2[z])


def quatFromAxisAngle(v, angle):
    "Create a quaternion from an axis vector and an angle."
    vn = vec.normalize(v)
    sinAngle = math.sin(angle / 2.0)
    return (vn[x] * sinAngle,
            vn[y] * sinAngle,
            vn[z] * sinAngle,
            math.cos(angle / 2.0))


def quatToAxisAngle(q):
    "Accepts a quaternion and returns the axis vector and angle."
    angle = 2.0 * math.acos(q[w])
    scale = math.sqrt(1-q[w]*q[w])
    if scale < eps:
        axis = vec.vec(0,1,0)
    else:
        axis = vec.vec(q[x] / scale,
                       q[y] / scale,
                       q[z] / scale)
    return axis, angle


if __name__ == "__main__":
    print "Running tests."

    v1 = vec.vec(0.0, 0.0, 0.0)
    v2 = vec.vec(100.0, 100.0, 100.0)
    r1 = quatFromAxisAngle(vec.vec(0,1,0), 0.0)
    r2 = quatFromAxisAngle(vec.vec(0,1,0), math.pi/2.0)
    r3 = quatFromAxisAngle(vec.vec(1,0,0), -math.pi/2.0)
    r4 = quatFromAxisAngle(vec.vec(1,0,0), -math.pi/4.0)
    i = quat(0.0, 0.0, 0.0, 1.0)

    assert mulByQuat(i, conjugate(i)) == i
    assert vec.isEqual(mulByQuat(i, r1), r1)
    assert vec.isEqual(mulByQuat(r1, i), r1)

    assert vec.isEqual(mulByVec(i, v1), v1)
    assert vec.isEqual(mulByVec(i, v2), v2)

    assert vec.isEqual(mulByVec(r2, vec.vec(0,0,2)), vec.vec(2,0,0))
    assert vec.isEqual(mulByVec(r1, vec.vec(0,0,2)), vec.vec(0,0,2))

    assert vec.isEqual(mulByVec(r3, vec.vec(0,0,2)), vec.vec(0,2,0))
    assert vec.isEqual(mulByVec(r4, vec.vec(0,0,2)), vec.vec(0,math.sqrt(2),math.sqrt(2)))

    print "Passed."
