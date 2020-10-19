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
#include "comparators.h"

#include <iomanip>

namespace {
    constexpr double ERROR_RANGE{0.0000001};
}

namespace alus {

size_t EqualsArrays(const float *a, const float *b, int elems, float delta){
    int i;
    size_t count =0;
    float temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > delta){
            std::cerr << std::fixed << std::setprecision(8);
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

size_t EqualsArrays(const float *a,const float *b, int elems){
    return EqualsArrays(a, b, elems, ERROR_RANGE);
}



size_t EqualsArraysd(const double *a, const double *b, int elems){
    return EqualsArraysd(a, b, elems, ERROR_RANGE);
}

size_t EqualsArraysd(const double *a, const double *b, int elems, double delta){
    int i;
    size_t count =0;
    double temp;

    for(i=0; i<elems; i++){
        temp = (a[i]>b[i])*(a[i]-b[i]) + (a[i]<=b[i])*(b[i]-a[i]);
        if(temp > delta){
            std::cerr << std::fixed << std::setprecision(8);
            std::cerr << "elements do not match - " <<i<<")"<< a[i] << ":"<<b[i] << '\n';
            count++;
            if(count > 50){
                return count;
            }
        }
    }
    return count;
}

size_t EqualsArrays2Dd(const double *const *a, const double *const *b, int x, int y){
    int i,j;
    size_t count=0;
    double temp;

    for(i=0; i<x; i++){
        for(j=0; j<y; j++){
            temp = (a[i][j]>b[i][j])*(a[i][j]-b[i][j]) + (a[i][j]<=b[i][j])*(b[i][j]-a[i][j]);
            if(temp > ERROR_RANGE){
                std::cerr << "elements do not match - " <<i<<","<<j<<")"<< a[i][j] << ":"<<b[i][j] << '\n';
                count++;
                if(count > 50){
                    return count;
                }
            }

        }
    }
    return count;
}

int EqualsDouble(const double a, const double b, const double delta){
    double temp = (a>b)*(a-b) + (a<=b)*(b-a);
    return (temp < delta);
}

/**
 * This function will find if the triangles are made of same points even if the points are not in the same order. Compares indicies.
 * @param a triangle 1
 * @param b triangle 2
 * @return return 1 if the triangles are equal and 0 if not.
 */
int EqualsTrianglesByIndices(delaunay::DelaunayTriangle2D a, delaunay::DelaunayTriangle2D b){

    int score = 0;
    score += (a.a_index == b.a_index) || (a.a_index == b.b_index) || (a.a_index == b.c_index);
    score += (a.b_index == b.a_index) || (a.b_index == b.b_index) || (a.b_index == b.c_index);
    score += (a.c_index == b.a_index) || (a.c_index == b.b_index) || (a.c_index == b.c_index);

    return (score == 3);
}

/**
 * This function will find if the triangles are made of same points even if the points are not in the same order.
 * Compares point coordinates.
 * @param a triangle 1
 * @param b triangle 2
 * @param delta Maximum error that 2 points are allowed to differentiate.
 * @return 1 for triangles equaling, 0 for not equaling.
 */
int EqualsTrianglesByPoints(delaunay::DelaunayTriangle2D a, delaunay::DelaunayTriangle2D b, double delta){
    int score = 0, temp_a, temp_b, temp_c;

    temp_a = EqualsDouble(a.ax, b.ax, delta) && EqualsDouble(a.ay, b.ay, delta);
    temp_b = EqualsDouble(a.ax, b.bx, delta) && EqualsDouble(a.ay, b.by, delta);
    temp_c = EqualsDouble(a.ax, b.cx, delta) && EqualsDouble(a.ay, b.cy, delta);
    score += temp_a || temp_b || temp_c;

    temp_a = EqualsDouble(a.bx, b.ax, delta) && EqualsDouble(a.by, b.ay, delta);
    temp_b = EqualsDouble(a.bx, b.bx, delta) && EqualsDouble(a.by, b.by, delta);
    temp_c = EqualsDouble(a.bx, b.cx, delta) && EqualsDouble(a.by, b.cy, delta);
    score += temp_a || temp_b || temp_c;

    temp_a = EqualsDouble(a.cx, b.ax, delta) && EqualsDouble(a.cy, b.ay, delta);
    temp_b = EqualsDouble(a.cx, b.bx, delta) && EqualsDouble(a.cy, b.by, delta);
    temp_c = EqualsDouble(a.cx, b.cx, delta) && EqualsDouble(a.cy, b.cy, delta);
    score += temp_a || temp_b || temp_c;

    return (score == 3);
}

void printTriangle(delaunay::DelaunayTriangle2D tri){
    std::cout<< tri.a_index << " (" << tri.ax << ";"<<tri.ay << ") " ;
    std::cout<< tri.b_index << " (" << tri.bx << ";"<<tri.by << ") " ;
    std::cout<< tri.c_index << " (" << tri.cx << ";"<<tri.cy << ") " << std::endl;
}

/**
 * Compares 2 arrays of triangles and outputs how many triangles from the first array were not found in the second.
 * You might find that the function outputs the first 20 triangles that do not match. This is because we want to
 * see if the triangles that do not match come in a sequence or are random. This helps a lot with debugging.
 * @param a
 * @param b
 * @param length
 * @param delta
 * @return the amount of mismatching triangles.
 */
size_t EqualsTriangles(delaunay::DelaunayTriangle2D *a, delaunay::DelaunayTriangle2D *b,size_t length, double delta){
    size_t count = 0, was_it_found = 0;
    size_t i,j;

    for(i=0; i<length; i++){
        was_it_found =0;
        for(j=0; j<length; j++){

            if(EqualsTrianglesByIndices(a[i], b[j])){
                if(!EqualsTrianglesByPoints(a[i], b[j], delta)){
                    std::cout<<"Triangles are equal by indicies but not by points. Triangle a: "<<std::endl;
                    printTriangle(a[i]);
                    std::cout<<"Triangle b: "<< std::endl;
                    printTriangle(b[j]);
                }else{
                    was_it_found = 1;
                }
                break;
            }
        }//inner loop
        if(!was_it_found){
            std::cout<< "Can not find a match for triangle: " << i << std::endl;
            printTriangle(a[i]);
            count++;
        }
        //I feel 20 triangles should give the impression if they are random.
        if(count > 20){
            return count;
        }

    }//outer loop

    return count;
}



}//namespace
