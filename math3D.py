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
"Routines which operate on quaternions."

import math


eps = 1e-8


def getCameraEyeCenterUp(cameraPos, cameraRot):
    p = cameraPos
    l = cameraPos.add(cameraRot.mulByVec(Vector3(0, 0,-1)).normalize())
    u = cameraRot.mulByVec(Vector3(0,+1, 0)).normalize()
    return p,l,u


def getBoxVertices(minP, maxP):
    p = [None]*8
    W = maxP.x - minP.x
    H = maxP.y - minP.y
    D = maxP.z - minP.z
    p[0] = minP
    p[1] = minP.add(Vector3(W,0,0))
    p[2] = minP.add(Vector3(0,0,D))
    p[3] = minP.add(Vector3(W,0,D))
    p[4] = minP.add(Vector3(0,H,0))
    p[5] = minP.add(Vector3(W,H,0))
    p[6] = minP.add(Vector3(0,H,D))
    p[7] = minP.add(Vector3(W,H,D))
    return p


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
        return Vector3(self.x-v.x, self.y-v.y, self.z-v.z)


    def dot(self, v):
        return self.x * v.x + self.y * v.y + self.z * v.z


    def cross(self, v):
        vx = self.y * v.z - self.z * v.y
        vy = self.z * v.x - self.x * v.z
        vz = self.x * v.y - self.y * v.x
        return Vector3(vx, vy, vz)


    def scale(self, s):
        return Vector3(self.x*s, self.y*s, self.z*s)


    def __eq__(a, b):
        return abs(a.x-b.x)<eps and  abs(a.y-b.y)<eps and  abs(a.z-b.z)<eps


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


class Plane:
    def __init__(self):
        self.p = Vector3(0,0,0)
        self.n = Vector3(0,1,0)
        self.D = -self.n.dot(self.p) # calculate up front for perf reasons


    def __str__(self):
        return "<math3D.Plane: p=%r, n=%r>" % (self.p, self.n)


    def distance(self, r):
        return self.n.dot(r) + self.D


    @classmethod
    def makeFromPoints(cls, p0, p1, p2):
        v = p1.sub(p0)
        u = p2.sub(p0)
        plane = Plane()
        plane.n = v.cross(u).normalize()
        plane.p = p0
        plane.D = -plane.n.dot(plane.p) # calculate up front for perf reasons
        return plane


class Frustum:
    """Camera frustum
    Based on <http://zach.in.tu-clausthal.de/teaching/cg_literatur/lighthouse3d_view_frustum_culling/index.html>
    """
    INTERSECT = 0
    INSIDE = 1
    OUTSIDE = 2

    TOP = 0
    BOTTOM = 1
    LEFT = 2
    RIGHT = 3
    NEARP = 4
    FARP = 5


    def __init__(self):
        self.pl = [None]*6
        self.ntl = Vector3(0,0,0)
        self.ntr = Vector3(0,0,0)
        self.nbl = Vector3(0,0,0)
        self.nbr = Vector3(0,0,0)
        self.ftl = Vector3(0,0,0)
        self.ftr = Vector3(0,0,0)
        self.fbl = Vector3(0,0,0)
        self.fbr = Vector3(0,0,0)
        self.nearD = 0
        self.farD = 0
        self.ratio = 0
        self.angle = 0
        self.tang = 0
        self.nw = 0
        self.nh = 0
        self.fw = 0
        self.fh = 0


    def __str__(self):
        return "<math3D.Frustum:\n" + "\n".join(map(str, self.pl)) + ">"


    def setCamInternals(self, angle, ratio, nearD, farD):
        """This function takes exactly the same parameters as the function
        gluPerspective. Each time the perspective definitions change, for
        instance when a window is resized, this function should be called as
        well.
        """
        self.ratio = ratio
        self.angle = angle
        self.nearD = nearD
        self.farD = farD

        # compute width and height of the near and far plane sections
        self.tang = math.tan((math.pi/180.0) * angle * 0.5)
        self.nh = self.nearD * self.tang
        self.nw = self.nh * self.ratio
        self.fh = self.farD * self.tang
        self.fw = self.fh * self.ratio


    def setCamDef(self, p, l, u):
        """This function takes three vectors that contain the information for
        the gluLookAt function: the position of the camera, a point to where
        the camera is pointing and the up vector. Each time the camera position
        or orientation changes, this function should be called as well.
        """
        # compute the Z axis of camera
        # this axis points in the opposite direction from
        # the looking direction
        Z = p.sub(l).normalize()

        # X axis of camera with given "up" vector and Z axis
        X = u.cross(Z).normalize()

        # the real "up" vector is the cross product of Z and X
        Y = Z.cross(X)

        # compute the centers of the near and far planes
        nc = p.sub(Z.scale(self.nearD))
        fc = p.sub(Z.scale(self.farD))

        # compute the 4 corners of the frustum on the near plane
        self.ntl = nc.add(Y.scale(self.nh)).sub(X.scale(self.nw))
        self.ntr = nc.add(Y.scale(self.nh)).add(X.scale(self.nw))
        self.nbl = nc.sub(Y.scale(self.nh)).sub(X.scale(self.nw))
        self.nbr = nc.sub(Y.scale(self.nh)).add(X.scale(self.nw))

        # compute the 4 corners of the frustum on the far plane
        self.ftl = fc.add(Y.scale(self.fh)).sub(X.scale(self.fw))
        self.ftr = fc.add(Y.scale(self.fh)).add(X.scale(self.fw))
        self.fbl = fc.sub(Y.scale(self.fh)).sub(X.scale(self.fw))
        self.fbr = fc.sub(Y.scale(self.fh)).add(X.scale(self.fw))

        # compute the six planes
        # the function set3Points assumes that the points
        # are given in counter clockwise order
        self.pl[Frustum.TOP] = Plane.makeFromPoints(self.ntr, self.ntl, self.ftl)
        self.pl[Frustum.BOTTOM] = Plane.makeFromPoints(self.nbl, self.nbr, self.fbr)
        self.pl[Frustum.LEFT] = Plane.makeFromPoints(self.ntl, self.nbl, self.fbl)
        self.pl[Frustum.RIGHT] = Plane.makeFromPoints(self.nbr, self.ntr, self.fbr)
        self.pl[Frustum.NEARP] = Plane.makeFromPoints(self.ntl, self.ntr, self.nbr)
        self.pl[Frustum.FARP] = Plane.makeFromPoints(self.ftr, self.ftl, self.fbl)


    #@profile
    def boxInFrustum(self, vertices):
        result = Frustum.INSIDE
        outR = 0
        inR = 0

        # for each plane do ...
        for i in range(6):
            # reset counters for corners in and out
            outR = 0
            inR = 0

            # for each corner of the box do ...
            # get out of the cycle as soon as a box as corners
            # both inside and out of the frustum
            for k in range(8):
                # is the corner outside or inside
                if self.pl[i].distance(vertices[k]) < 0:
                    outR += 1
                else:
                    inR += 1

                if not (inR==0 or outR==0):
                    break

            # if all corners are out
            if inR == 0:
                return Frustum.OUTSIDE
            # if some corners are out and others are in
            elif outR != 0:
                result = Frustum.INTERSECT

        return result


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

    # Rotate by 90 degrees about the y axis and make sure the up vector
    # is still straight up.
    a = r1.mulByQuat(r2)
    u = a.mulByVec(Vector3(0,1,0))
    assert u == Vector3(0,1,0)
    del a, u

    frustum = Frustum()
    frustum.setCamInternals(65, 640.0/480.0, .1, 1000)
    frustum.setCamDef(Vector3(0,0,0),
                      Vector3(0,0,1),
                      Vector3(0,1,0))

    minP = Vector3(-2,-2,-2)
    maxP = Vector3(-1,-1,-1)
    vertices = getBoxVertices(minP, maxP)
    assert frustum.boxInFrustum(vertices) == Frustum.OUTSIDE

    minP = Vector3(-1,-1,-1)
    maxP = Vector3(+1,+1,+1)
    vertices = getBoxVertices(minP, maxP)
    assert frustum.boxInFrustum(vertices) != Frustum.OUTSIDE

    print "Passed."
