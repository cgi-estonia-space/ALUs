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

#include "../include/snap_delaunay_triangulator.h"

#include <ctgmath>


namespace external {
namespace delaunay {

SnapDelaunayTriangulator::SnapDelaunayTriangulator() {
    this->HORIZON.x = std::nan("");
    this->HORIZON.y = std::nan("");
    this->HORIZON.index = -1;
}

SnapDelaunayTriangulator::~SnapDelaunayTriangulator(){
    size_t size = this->triangles.size();
    SnapTriangle *triangle;

    for(size_t i =0; i<size; i++){
        triangle = this->triangles.at(i);
        delete triangle;
    }
    this->triangles.clear();
}

/**
 * Points coming in are already unique, sorted and all valid.
 * @param p
 * @param size
 */
void SnapDelaunayTriangulator::Triangulate(alus::PointDouble *p, int size) {

    if(size < 3){
        throw std::runtime_error("Triangulation needs atleast 3 points to work");
    }

    InitTriangulation(p[0], p[1]);
    for(int i=2; i<size; i++){
        AddExternalVertex(p[i]);
    }

}

void SnapDelaunayTriangulator::InitTriangulation(alus::PointDouble c0, alus::PointDouble c1){
    SnapTriangle *t0 = new SnapTriangle(c0, c1, HORIZON); //TODO: should those be class members? You know, for deletion.
    SnapTriangle *t1 = new SnapTriangle(c1, c0, HORIZON);

    t0->SetNeighbours(t1, t1, t1);
    t1->SetNeighbours(t0, t0, t0);
    this->current_external_triangle_ = t1;
}

void SnapDelaunayTriangulator::AddExternalVertex(alus::PointDouble point){
    std::vector<SnapTriangle*>  newTriangles = BuildTrianglesBetweenNewVertexAndConvexHull(point);
    SnapTriangle * temp_triangle;

    for (size_t i=0; i<newTriangles.size(); i++) {
        temp_triangle = newTriangles.at(i);
        if (temp_triangle->C.index != HORIZON.index) Delaunay(temp_triangle, 0);
        triangles.push_back(temp_triangle);
    }
    //if (debugLevel>=SHORT) {
    //    for (Triangle t : newTriangles) debug(1,"add triangle " + t);
    //}
    //triangles.addAll(newTriangles);
}

std::vector<SnapTriangle*> SnapDelaunayTriangulator::BuildTrianglesBetweenNewVertexAndConvexHull(alus::PointDouble point){
    SnapTriangle *currentT = this->current_external_triangle_ ;
    SnapTriangle *nextExternalTriangle = this->current_external_triangle_->GetNeighbour(2);

    int lastCCW = currentT->ccw(point);
    int currentCCW = lastCCW;
    SnapTriangle *beforeFirstVisibleT = this->current_external_triangle_;
    SnapTriangle *firstVisibleT = nullptr;
    SnapTriangle *lastVisibleT = nullptr;
    SnapTriangle *afterLastVisibleT = nextExternalTriangle;

    std::vector<SnapTriangle*> newT;
    bool oneCycleCompleted = false;

    while (true) {
        currentT = currentT->ACO;
        currentCCW = currentT->ccw(point);
        if (currentCCW > 0) {
            if (lastCCW <= 0) {
                firstVisibleT = currentT;
                beforeFirstVisibleT = currentT->CBO;
            }
            if (firstVisibleT != nullptr) {
                //if (debugLevel>=VERBOSE) debug(2,"visible side : " + currentT.getA()+"-"+currentT.getB());
                currentT->C = point;
                newT.push_back(currentT); //TODO: possible place for bug. No idea if they wanted to copy value or pointer.
            }
            //else {
            //if (debugLevel>=VERBOSE) debug(2,"before first visible side : " + currentT.getA()+"-"+currentT.getB());
            //}
        } else if (firstVisibleT != nullptr && lastCCW > 0) {
            lastVisibleT = currentT->CBO;
            afterLastVisibleT = currentT;
        }
        lastCCW = currentCCW;
        if (firstVisibleT != nullptr && lastVisibleT != nullptr) break;
        if (oneCycleCompleted && firstVisibleT == nullptr && lastVisibleT == nullptr) break;
        if (currentT == this->current_external_triangle_ ) oneCycleCompleted = true; //TODO: should this be equals?
    }
    this->current_external_triangle_ = new SnapTriangle(point, beforeFirstVisibleT->A, HORIZON);
    nextExternalTriangle = new SnapTriangle(afterLastVisibleT->B, point, HORIZON);
    LinkExteriorTriangles(beforeFirstVisibleT, this->current_external_triangle_);
    if (firstVisibleT != nullptr || lastVisibleT != nullptr) {
        Link(this->current_external_triangle_, 0, firstVisibleT, 1);
        Link(nextExternalTriangle, 0, lastVisibleT, 2);
    } else Link(this->current_external_triangle_, 0, nextExternalTriangle, 0);
    LinkExteriorTriangles(nextExternalTriangle, afterLastVisibleT);

    LinkExteriorTriangles(this->current_external_triangle_, nextExternalTriangle);

    return newT;
}

void SnapDelaunayTriangulator::Link(SnapTriangle *t1, int side1, SnapTriangle *t2, int side2) {
    t1->SetNeighbour(side1, t2);
    t2->SetNeighbour(side2, t1);
}

void SnapDelaunayTriangulator::Link(SnapTriangle *t1, int side1, SnapTriangle *t2) {
    alus::PointDouble p1 = side1==0?t1->A:side1==1?t1->B:t1->C; //t1.getVertex(side1);
    if (p1.index == (side1==0?t2->A.index:side1==1?t2->B.index:t2->C.index)) {         //t2.getVertex(side1)) {
        t1->SetNeighbour(side1, t2);
        t2->SetNeighbour((side1 + 2) % 3, t1);
    } else {
        int side2 = (side1 + 1) % 3;
        if (p1.index == (side2==0?t2->A.index:side2==1?t2->B.index:t2->C.index)) {     //t2.getVertex(side2)) {
            t1->SetNeighbour(side1, t2);
            t2->SetNeighbour(side1, t1);
        } else {
            int side3 = (side1 + 2) % 3;
            if (p1.index == (side3==0?t2->A.index:side3==1?t2->B.index:t2->C.index)) { //t2.getVertex(side3)) {
                t1->SetNeighbour(side1, t2);
                t2->SetNeighbour(side2, t1);
            }
        }
    }
}

/**
 * Link t1 and t2 where t1 and t2 are both infinite exterior triangles,
 * t2 following t1 if one iterates around the triangulation in ccw.
 */
void SnapDelaunayTriangulator::LinkExteriorTriangles(SnapTriangle *t1, SnapTriangle *t2) {
    //assert (t1.C == HORIZON && t2.C == HORIZON && t1.A == t2.B);
    t1->ACO = t2;
    t2->CBO = t1;
}

/**
     * Check the delaunay property of this triangle. If the circumcircle contains
     * one of the opposite vertex, the two triangles forming the quadrilatera are
     * flipped. The method is iterative.
     * While triangulating an ordered set of coordinates about
     * <ul>
     * <li>40% of time is spent in flip() method,</li>
     * <li>15% of time is spent in fastInCircle() method and</li>
     * <li>10% of time is spent in getOpposite() method</li>
     * </ul>
     *
     * @param t triangle to check and to modify (if needed)
     */
void SnapDelaunayTriangulator::Delaunay(SnapTriangle *t, int side) {

    //if (t.getEdgeType(side) == Triangle.EdgeType.HARDBREAK) return;
    if ((side==0 ? t->AB : side==1 ? t->BC : t->CA) == EdgeType::HARDBREAK) return;

    SnapTriangle *opp = (side==0? t->BAO : side==1 ? t->CBO:t->ACO); //t.getNeighbour(side);
    if (opp->C.index == HORIZON.index) return;
    int i = t->GetOpposite(side);

    if (t->InCircle(i==0?opp->A:i==1?opp->B:opp->C) > 0) { //opp.getVertex(i)) > 0) {
        // Flip triangles without creating new Triangle objects
        Flip(t, side, opp, (i + 1) % 3);
        Delaunay(t, 1);
        Delaunay(t, 2);
        Delaunay(opp, 0);
        Delaunay(opp, 1);
    }
}

/**
     * If t0 and t1 are two triangles sharing a common edge AB,
     * the method replaces ABC and BAD triangles by DCA and DBC, respectively.
     *          B                      B
     *          *                      *
     *        / | \                  /   \
     *      /   |   \              /       \
     *     /    |    \            /         \
     *  C *     |     * D      C *-----------* D
     *     \    |    /            \         /
     *      \   |   /              \       /
     *       \  |  /                \     /
     *        \ | /                  \   /
     *          *                      *
     *          A                      A
     * To be fast, this method supposed that input triangles share a common
     * edge and that this common edge is known.
     * A check may be performed to ensure these conditions are verified.
     * TODO : change the breaklines edges
     */
void SnapDelaunayTriangulator::Flip(SnapTriangle *t0, int side0, SnapTriangle *t1, int side1) {
    const int side0_1 = (side0 + 1) % 3;
    const int side0_2 = (side0 + 2) % 3;
    const int side1_1 = (side1 + 1) % 3;
    const int side1_2 = (side1 + 2) % 3;

    alus::PointDouble t0A = side1_2==0?t1->A:side1_2==1?t1->B:t1->C;    //t1.getVertex(side1_2);
    alus::PointDouble t0B = side0_2==0?t0->A:side0_2==1?t0->B:t0->C;    //t0.getVertex(side0_2);
    alus::PointDouble t1B = side0_1==0?t0->A:side0_1==1?t0->B:t0->C;          //t0.getVertex(side0_1);
    // New neighbours
    SnapTriangle *newt0N1 = (side0_2==0?t0->BAO:side0_2==1?t0->CBO:t0->ACO); //t0.getNeighbour(side0_2);
    SnapTriangle *newt0N2 = (side1_1==0?t1->BAO:side1_1==1?t1->CBO:t1->ACO); //t1.getNeighbour(side1_1);
    SnapTriangle *newt1N0 = (side1_2==0?t1->BAO:side1_2==1?t1->CBO:t1->ACO); //t1.getNeighbour(side1_2);
    SnapTriangle *newt1N1 = (side0_1==0?t0->BAO:side0_1==1?t0->CBO:t0->ACO); //t0.getNeighbour(side0_1);
    t0->SetABC(t0A, t0B, t0->GetVertex(side0));
    t0->BAO = t1;
    Link(t0,1,newt0N1);
    Link(t0,2,newt0N2);
    t1->SetABC(t0A, t1B, t0B);
    Link(t1,0,newt1N0);
    Link(t1,1,newt1N1);
    t1->ACO = t0;
}

std::vector<alus::delaunay::DelaunayTriangle2D> SnapDelaunayTriangulator::Get2dTriangles(){
    alus::delaunay::DelaunayTriangle2D temp_triangle;
    std::vector<alus::delaunay::DelaunayTriangle2D> results;
    SnapTriangle *snap_triangle;
    const size_t size = this->triangles.size();


    for(size_t i=0; i<size; i++){
        snap_triangle = this->triangles.at(i);
        temp_triangle.ax = snap_triangle->A.x;
        temp_triangle.ay = snap_triangle->A.y;
        temp_triangle.a_index = snap_triangle->A.index;

        temp_triangle.bx = snap_triangle->B.x;
        temp_triangle.by = snap_triangle->B.y;
        temp_triangle.b_index = snap_triangle->B.index;

        temp_triangle.cx = snap_triangle->C.x;
        temp_triangle.cy = snap_triangle->C.y;
        temp_triangle.c_index = snap_triangle->C.index;

        results.push_back(temp_triangle);
    }
    return results;
}

}//namespace
}//namespace