#!/usr/bin/env python
# vim:ts=4:sw=4:et:filetype=python
"Routines which operate on quaternions."

import math


eps = 1e-8


class Vector3:
    def __init__(self, x, y, z):
        self.x = x
        self.y = y
        self.z = z


    def __eq__(a, b):
        return abs(a.x-b.x)<eps and \
               abs(a.y-b.y)<eps and \
               abs(a.z-b.z)<eps


    def __repr__(self):
        return "(%.3f, %.3f, %.3f)" % (self.x, self.y, self.z)


    def __str__(self):
        return "(%.3f, %.3f, %.3f)" % (self.x, self.y, self.z)


    def length(self):
        return math.sqrt(self.x*self.x + self.y*self.y + self.z*self.z)


    def normalize(self):
        "Returns a normalized vector (unit length)"
        mag2 = self.x*self.x + self.y*self.y + self.z*self.z
        if abs(mag2) > eps and abs(mag2 - 1.0) > eps:
            mag = math.sqrt(mag2)
            return Vector3(self.x / mag, self.y / mag, self.z / mag)
        else:
            return self


    def add(self, v):
        return Vector3(self.x+v.x, self.y+v.y, self.z+v.z)


    def sub(self, v):
        return Vector3(self.x-v.x, self.y0v.y, self.z0v.z)


    def __eq__(a, b):
        return abs(a.x-b.x)<eps and \
               abs(a.y-b.y)<eps and \
               abs(a.z-b.z)<eps


class Quaternion:
    def __init__(self, x, y, z, w):
        self.x = x
        self.y = y
        self.z = z
        self.w = w


    def __eq__(a, b):
        return abs(a.x-b.x)<eps and \
               abs(a.y-b.y)<eps and \
               abs(a.z-b.z)<eps and \
               abs(a.w-b.w)<eps


    def __repr__(self):
        return "%.3f*i + %.3f*j + %.3f*k + %.3f" % \
            (self.x, self.y, self.z, self.w)


    def __str__(self):
        return "%.3f*i + %.3f*j + %.3f*k + %.3f" % \
            (self.x, self.y, self.z, self.w)


    def normalize(self):
        "Returns a normalized quaternion (unit length)"
        mag2 = self.x*self.x + self.y*self.y + self.z*self.z + self.w*self.w
        if abs(mag2) > eps and abs(mag2 - 1.0) > eps:
            mag = math.sqrt(mag2)
            return Quaternion(self.x / mag, self.y / mag,
                              self.z / mag, self.w / mag)
        else:
            return self


    def conjugate(self):
        return Quaternion(-self.x, -self.y, -self.z, self.w)


    def mulByQuat(self, r):
        "Multiplying q1 with q2 applies the rotation q2 to q1"
        x1 = self.x
        y1 = self.y
        z1 = self.z
        w1 = self.w

        x2 = r.x
        y2 = r.y
        z2 = r.z
        w2 = r.w

        w3 = w1*w2 - x1*x2 - y1*y2 - z1*z2
        x3 = w1*x2 + x1*w2 + y1*z2 - z1*y2
        y3 = w1*y2 - x1*z2 + y1*w2 + z1*x2
        z3 = w1*z2 + x1*y2 - y1*x2 + z1*w2

        return Quaternion(x3, y3, z3, w3)


    def mulByVec(self, v):
        """Multiplying a quaternion q with a vector v applies the q-rotation
        to v"""
        vAsQ = Quaternion(v.x, v.y, v.z, 0.0)
        q2 = self.mulByQuat(vAsQ).mulByQuat(self.conjugate())
        return Vector3(q2.x, q2.y, q2.z)


    @classmethod
    def fromAxisAngle(cls, v, angle):
        "Create a quaternion from an axis vector and an angle."
        vn = v.normalize()
        sinAngle = math.sin(angle / 2.0)
        return Quaternion(vn.x * sinAngle,
                          vn.y * sinAngle,
                          vn.z * sinAngle,
                          math.cos(angle / 2.0))


    def toAxisAngle(self):
        "Accepts a quaternion and returns the axis vector and angle."
        angle = 2.0 * math.acos(self.w)
        scale = math.sqrt(1-self.w*self.w)
        if scale < eps:
            axis = Vector3(0,1,0)
        else:
            axis = Vector3(self.x / scale,
                           self.y / scale,
                           self.z / scale)
        return axis, angle


if __name__ == "__main__":
    print "Running tests."

    v1 = Vector3(0.0, 0.0, 0.0)
    v2 = Vector3(100.0, 100.0, 100.0)
    r1 = Quaternion.fromAxisAngle(Vector3(0,1,0), 0.0)
    r2 = Quaternion.fromAxisAngle(Vector3(0,1,0), math.pi/2.0)
    r3 = Quaternion.fromAxisAngle(Vector3(1,0,0), -math.pi/2.0)
    r4 = Quaternion.fromAxisAngle(Vector3(1,0,0), -math.pi/4.0)
    i = Quaternion(0.0, 0.0, 0.0, 1.0)

    assert Quaternion.mulByQuat(i, i.conjugate()) == i
    assert i.mulByQuat(r1) == r1
    assert r1.mulByQuat(i) == r1

    assert i.mulByVec(v1) == v1
    assert i.mulByVec(v2) == v2

    assert r2.mulByVec(Vector3(0,0,2)) == Vector3(2,0,0)
    assert r1.mulByVec(Vector3(0,0,2)) == Vector3(0,0,2)

    assert r3.mulByVec(Vector3(0,0,2)) == Vector3(0,2,0)
    assert r4.mulByVec(Vector3(0,0,2)) == Vector3(0,math.sqrt(2),math.sqrt(2))

    print "Passed."
