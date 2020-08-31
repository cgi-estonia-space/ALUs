/**
 * This program is free software; you can redistribute it and/or modify it
 * under the terms of the GNU General Public License as published by the Free
 * Software Foundation; either version 3 of the License, or (at your option)
 * any later version.
 * This program is distributed in the hope that it will be useful, but WITHOUT
 * ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
 * FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
 * more details.
 *
 * You should have received a copy of the GNU General Public License along
 * with this program; if not, see http://www.gnu.org/licenses/
 */
#pragma once

#include "shapes.h"

namespace external {
namespace delaunay {

enum class EdgeType { VIRTUAL,SOFTBREAK,HARDBREAK };


/**
 * This class attempts to be a copy of org.jlinda.core.delaunay.Triangle class.
 */
class SnapTriangle{
   private:

   public:
    alus::PointDouble A, B, C;
    SnapTriangle *BAO, *CBO, *ACO;

   EdgeType AB = EdgeType::VIRTUAL;
   EdgeType BC = EdgeType::VIRTUAL;
   EdgeType CA = EdgeType::VIRTUAL;

    SnapTriangle *GetNeighbour(int i) {
        return i==0?BAO:i==1?CBO:ACO;
    }

    SnapTriangle(alus::PointDouble a, alus::PointDouble b, alus::PointDouble c){
        A = a;
        B = b;
        C = c;
    }

   void SetNeighbours(SnapTriangle *BAO, SnapTriangle *CBO, SnapTriangle *ACO) {
        this->BAO = BAO;
        this->CBO = CBO;
        this->ACO = ACO;
    }

    int ccw(alus::PointDouble point){
        const double dx1dy2 = (B.x - A.x) * (point.y - A.y);
        const double dy1dx2 = (B.y - A.y) * (point.x - A.x);
        return dx1dy2 > dy1dx2 ? 1 : dx1dy2 < dy1dx2 ? -1 : 0;
    }

   void SetNeighbour(int side, SnapTriangle *t) {
        if (side == 0) this->BAO = t;
        else if (side == 1) this->CBO = t;
        else this->ACO = t;
    }

    /**
     * Look for the position of the point in the neighbour of this Triangle
     * which is not adjacent to this Triangle
     * @param side number of the neighbour where opposite point is looked for.
     * @return the position of opposite point in the neighbour description
     */
   int GetOpposite(int side) {
        return side==0 ? BAO->GetOppSide(A) : side==1 ? CBO->GetOppSide(B) : side==2 ? ACO->GetOppSide(C) : -1;
    }

   int GetOppSide(alus::PointDouble p) {
        return A.index==p.index ? 1 : B.index==p.index ? 2 : C.index==p.index ? 0 : -1;
    }

    /**
     * Return a positive value if the point p4 lies inside the
     * circle passing through pa, pb, and pc; a negative value if
     * it lies outside; and zero if the four points are cocircular.
     * The points pa, pb, and pc must be in counterclockwise
     * order, or the sign of the result will be reversed.
     */
   double InCircle(alus::PointDouble p4) {

        const double adx = A.x-p4.x;
        const double ady = A.y-p4.y;
        const double bdx = B.x-p4.x;
        const double bdy = B.y-p4.y;
        const double cdx = C.x-p4.x;
        const double cdy = C.y-p4.y;

        const double abdet = adx * bdy - bdx * ady;
        const double bcdet = bdx * cdy - cdx * bdy;
        const double cadet = cdx * ady - adx * cdy;
        const double alift = adx * adx + ady * ady;
        const double blift = bdx * bdx + bdy * bdy;
        const double clift = cdx * cdx + cdy * cdy;

        return alift * bcdet + blift * cadet + clift * abdet;
    }

    alus::PointDouble GetVertex(int i) {
        return i==0?A:i==1?B:C;
    }

   void SetABC(alus::PointDouble A, alus::PointDouble B, alus::PointDouble C) {
        this->A = A;
        this->B = B;
        this->C = C;
    }
};

}//namespace
}//namespace
