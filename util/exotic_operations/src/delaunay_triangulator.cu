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
#include "delaunay_triangulator.cuh"

#include "delaunay_triangle2D.h"
#include "cuda_util.hpp"

namespace alus{
namespace delaunay{

inline __device__ int GetBIndex(DelaunayTriangle2Dgpu *triangles, int my_index, int last_point, double closest_x, double closest_y, int current_triangle){
    DelaunayTriangle2Dgpu temp_triangle;
    double closest_distance;
    int closest_index;
    double comparable_distance;

    if(current_triangle == -1){
        return last_point;
    }


    temp_triangle = triangles[current_triangle];
    const double b_distance = sqrt(pow(temp_triangle.bx - closest_x, 2) + pow(temp_triangle.by - closest_y, 2));
    const double c_distance = sqrt(pow(temp_triangle.cx - closest_x, 2) + pow(temp_triangle.cy - closest_y, 2));
    closest_distance = (b_distance < c_distance) * b_distance + (b_distance >= c_distance) * c_distance;
    closest_index = (b_distance < c_distance) * temp_triangle.b_index + (b_distance >= c_distance) * temp_triangle.c_index;
    while(temp_triangle.previous != -1){
        printf("previous is %d \n", temp_triangle.previous);
        temp_triangle = triangles[temp_triangle.previous];

        comparable_distance = sqrt(pow(temp_triangle.bx - closest_x, 2) + pow(temp_triangle.by - closest_y, 2));
        closest_index = (closest_distance < comparable_distance) * closest_index + (closest_distance >= comparable_distance) * temp_triangle.b_index;
        closest_distance = (closest_distance < comparable_distance) * closest_distance + (closest_distance >= comparable_distance) * comparable_distance;

        comparable_distance = sqrt(pow(temp_triangle.cx - closest_x, 2) + pow(temp_triangle.cy - closest_y, 2));
        closest_index = (closest_distance < comparable_distance) * closest_index + (closest_distance >= comparable_distance) * temp_triangle.c_index;
        closest_distance = (closest_distance < comparable_distance) * closest_distance + (closest_distance >= comparable_distance) * comparable_distance;
    }
    return closest_index;



}

inline __device__ unsigned int GetNewTriangleIndex(unsigned int *empty_triangle_index, unsigned int size){
    /*
     * reads the 32-bit word "old" located at the address address in global or shared memory, computes
     * ((old >= val) ? 0 : (old+1)), and stores the result back to memory at the same address.
     * These three operations are performed in one atomic transaction. The function returns "old".
     */
    unsigned int result = atomicInc(empty_triangle_index, size +1);
    //printf("making new triangle from %d and %d. Got %d \n", *empty_triangle_index, size+1, result);
    return result;
}

/**
 * Finds the point where 2 straights meet.
 * @param straight1
 * @param straight2
 * @param x
 * @param y
 */
inline __device__ void StraightIntersectionPoint(StraightEquation2D straight1, StraightEquation2D straight2, double *x, double *y){
    *x = (straight1.c - straight2.c) / (straight2.m - straight1.m);
    *y = straight1.m * (*x) + straight1.c;
}

/**
 * If point a is situated on a straight made from points 1 and 2, then is point a between points 1 and 2?
 * @param x1 x coord for point 1
 * @param y1 y coord for point 1
 * @param x2 x coord for point 2
 * @param y2 y coord for point 2
 * @param ax x coord for point a
 * @param ay y coord for point a
 * @return returns 1 if the point is between the 2 and 0 if not.
 */
inline __device__ int IsPointBetween2(double x1, double y1, double x2, double y2, double ax, double ay){
    const double distEtalon = sqrt(pow(x1-x2, 2) + pow(y1-y2, 2));

    const double dist1 = sqrt(pow(x1-ax, 2) + pow(y1-ay, 2));
    const double dist2 = sqrt(pow(x2-ax, 2) + pow(y2-ay, 2));

    return (dist1 < distEtalon) && (dist2 < distEtalon);
}

/**
 * Creates an equation of straight for the current 2 points.
 * @param x1
 * @param y1
 * @param x2
 * @param y2
 * @return
 */
inline __device__ StraightEquation2D StraightEquation(double x1, double y1, double x2, double y2){
    StraightEquation2D result;
    double const uy = y2-y1;
    double const ux = x2-x1;
    //ux(y-y1) = uy(x-x1); y-y1 = uy(x-x1)/ux; y= uy(x-x1)/ux + y1
    result.m = uy/ux;
    result.c = uy*(-x1)/ux + y1;
    return result;
}
/**
 * Calculates if this point has not yet been accepted into the perimeter set and is in view of the origin point.
 * The last part means that a straight formed between that point and the origin point must not pass between any of
 * the points already added to the set.
 * @param point_index
 * @param px
 * @param py
 * @param triangles
 * @param latest_triangle
 * @return
 */
inline __device__ int IsPointStrangerAndInView(int point_index, double px, double py, DelaunayTriangle2Dgpu *triangles, int latest_triangle){
    DelaunayTriangle2Dgpu temp_triangle;
    int point_score = 0;
    int between_score = 0;
    double temp_x, temp_y;
    StraightEquation2D temp_straight;

    //no points in the set
    if(latest_triangle == -1){
        return 1;
    }

    do {
        temp_triangle = triangles[latest_triangle];
        point_score = (temp_triangle.b_index == point_index) + (temp_triangle.c_index == point_index);

        if (!point_score){
            temp_straight = StraightEquation(temp_triangle.ax, temp_triangle.ay, px, py);
            StraightIntersectionPoint(temp_triangle.bc_equation,
                                      temp_straight,
                                      &temp_x,
                                      &temp_y);
            //TODO: are straights parallel to those of owner to slaves?
            between_score = IsPointBetween2(temp_triangle.ax, temp_triangle.ay, px, py, temp_x, temp_y);
        }
        printf("comparing %d %d %d to %d and got point score %d, between score %d. BC straight %f %f, temp straight %f %f  Straight intersected at %f %f\n",
               temp_triangle.a_index,
               temp_triangle.b_index,
               temp_triangle.c_index,
               point_index,
               point_score,
               between_score,
               temp_triangle.bc_equation.m,
               temp_triangle.bc_equation.c,
               temp_straight.m,
               temp_straight.c,
               temp_x,
               temp_y);
        latest_triangle = temp_triangle.previous;
    }while(latest_triangle != -1 && point_score == 0 && between_score == 0);

    //printf("point score %d, between score %d\n", !point_score, !between_score);
    return !(point_score || between_score);
    /*if(ps == 0 && bs == 0) yes;
    if(ps == 1 && bs == 0) no;
    if(ps == 0 && bs == 1) no;
    if(ps == 1 && bs == 1) no;*/
}



/**
 * Enter the coordinates of the points in the set and prepare to receive the triangles back in an array.
 * Keep in mind that every triangle is in triple, which means that the output array has 3 times more elements than the inputs.
 * It is the caller's duty to check that coordinate arrays have 3 or more elements.
 * @param x_coords
 * @param y_coords
 * @param height
 * @param width
 * @param triangles
 */
__global__ void DelaunayTriangulation(double *x_coords, double *y_coords, const int width, const int height,
                                      DelaunayTriangle2Dgpu *triangles,unsigned int *empty_triangle_index){
    const int idx = threadIdx.x + (blockDim.x * blockIdx.x);
    const int idy = threadIdx.y + (blockDim.y * blockIdx.y);
    const int arr_size = width*height;
    double closest_distance, comparable_distance;
    double comparable_x, comparable_y;
    int closest_distance_index;
    int current_triangle = -1;
    int last_point = -1; //the last point added to the set before the current point.
    int i=0, temp_index;
    DelaunayTriangle2Dgpu temp_triangle;
    int closest_distance_found = 0;

    if(idx < width && idy < height){
        const int my_index = idx + idy *width;
        const double my_x = x_coords[my_index];
        const double my_y = y_coords[my_index];

        //for(int j=0; j<5; j++) {
        for(int j=0; j<(arr_size-1); j++) {

            closest_distance_found = 0;
            for (i = 0; i < arr_size; i++) {
                comparable_x = x_coords[i];
                comparable_y = y_coords[i];
                if (i == my_index || i == last_point || !IsPointStrangerAndInView(i, comparable_x, comparable_y, triangles, current_triangle)) {
                    //if(idx == 1)
                    //printf("Rejected point %f %f %d with owner %d\n", comparable_x, comparable_y,i, my_index);
                    continue;
                }
                //TODO: make sure that the comparable point does not cross an already drawn straight between its 2 points.

                comparable_distance = sqrt(pow(my_x - comparable_x, 2) + pow(my_y - comparable_y, 2));
                if(closest_distance_found) {
                    if (comparable_distance < closest_distance) {
                        closest_distance = comparable_distance;
                        closest_distance_index = i;
                    }
                }else{
                    closest_distance_found = 1;
                    closest_distance = comparable_distance;
                    closest_distance_index = i;
                }
            } //inner loop
            if(!closest_distance_found){
                break; // we are done. No more valid points available.
            }

            if (last_point == -1) {
                last_point = closest_distance_index;
            } else {

                temp_triangle.previous = current_triangle;
                comparable_x = x_coords[closest_distance_index];
                comparable_y = y_coords[closest_distance_index];
                temp_index = GetBIndex(triangles, my_index, last_point, comparable_x, comparable_y, current_triangle);

                temp_triangle.owner = my_index;
                temp_triangle.ax = my_x;
                temp_triangle.ay = my_y;
                temp_triangle.a_index = my_index;
                temp_triangle.bx = x_coords[temp_index];
                temp_triangle.by = y_coords[temp_index];
                temp_triangle.b_index = temp_index;
                temp_triangle.cx = comparable_x;
                temp_triangle.cy = comparable_y;
                temp_triangle.c_index = closest_distance_index;
                temp_triangle.bc_equation = StraightEquation(temp_triangle.bx, temp_triangle.by, temp_triangle.cx, temp_triangle.cy);


                current_triangle = (int)GetNewTriangleIndex(empty_triangle_index, 3*(arr_size-2));
                printf("getting a new index for triangle %d \n", current_triangle);
                triangles[current_triangle] = temp_triangle;
                //TODO: it will not work this way. Sometimes we draw a triangle that is incorrect.
                // So after drawing one we need to check if there are any points that fall into its circumcircle.
                // If there are, find the closest in view (IsPointBetween2) and turn that one into 2 triangles(flip).
                // Do not make a new check for those 2. Someone will draw the right triangle. Later reduce out all
                // incorrect triangles and all duplicates.

                //TODO: Did I tell you that you need to keep the triangles in counter clockwise orderwhen marking
                // them down? This is needed for circumcircle calculations.

                //TODO: Memory is a problem. A set can have 2n -2 -b triangles, where n is the nr of points and b is
                // the nr of vertices on a convex hull. However we will not be calculating the convex hull vertices,
                // as we do not have the time. So we will have to think that we will make 2n triangles and 6n in this
                // algorithm, as every one will be reported 3 times.
                last_point = closest_distance_index;
            }//if last_point
        }//outer loop
    } //deciding if

}

cudaError_t LaunchDelaunayTriangulation(double *x_coords, double *y_coords, const int width, const int height, DelaunayTriangle2Dgpu *triangles){
    dim3 block_size(20,20); //TODO: relook this number, as the computation itself is 1D.
    dim3 grid_size(cuda::getGridDim(20, width), cuda::getGridDim(20, height));

    unsigned int *empty_triangle_index;
    unsigned int zero = 0;

    //if something as simple as this fails, kill everything now! The system is probably toast.
    CHECK_CUDA_ERR(cudaMalloc((void**)&empty_triangle_index, sizeof(unsigned int)));
    CHECK_CUDA_ERR(cudaMemcpy(empty_triangle_index, &zero, sizeof(unsigned int), cudaMemcpyHostToDevice));

    DelaunayTriangulation<<<grid_size, block_size>>>(x_coords, y_coords, width, height, triangles, empty_triangle_index);

    cudaError_t result = cudaGetLastError();
    CHECK_CUDA_ERR(cudaFree(empty_triangle_index));

    return result;
}

}//namespace
}//namespace
