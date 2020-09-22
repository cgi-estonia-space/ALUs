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
#include <vector>
#include <fstream>

#include "gmock/gmock.h"

#include "delaunay_triangulator.h"
#include "delaunay_triangle2D.h"
#include "delaunay_triangulator.cuh"
#include "CudaFriendlyObject.h"
#include "tests_common.hpp"
#include "cuda_util.hpp"
#include "comparators.h"

using namespace alus::tests;

namespace{



class DelaunayTriangulatonTester: public alus::cuda::CudaFriendlyObject{
   private:
    static const std::string az_rg_data_file_;
    static const std::string triangles_data_file_;

   public:
    static constexpr double RG_AZ_RATIO{0.16742323135844578};
    static constexpr double INVALID_INDEX{-9999.0};

    std::vector<double> x_coords_;
    std::vector<double> y_coords_;
    std::vector<alus::delaunay::DelaunayTriangle2D> triangles_;
    int width_, height_, triangle_size_;
    double *device_x_coords_ = nullptr;
    double *device_y_coords_ = nullptr;
    alus::delaunay::DelaunayTriangle2D *device_triangles_ = nullptr;

    DelaunayTriangulatonTester() = default;
    ~DelaunayTriangulatonTester(){
        this->DeviceFree();
    }

    void ReadSmallTestData(){
        alus::delaunay::DelaunayTriangle2D temp_triangle;
        this->width_ = 6;
        this->height_ = 1;
        this->triangle_size_ = 5; //maximum triangles include 2n -2 -b
        x_coords_.resize(this->width_);
        y_coords_.resize(this->width_);

        x_coords_.at(0) = 1;
        y_coords_.at(0) = 1;

        x_coords_.at(1) = 2;
        y_coords_.at(1) = 2.5;

        x_coords_.at(2) = 3;
        y_coords_.at(2) = 1;

        x_coords_.at(3) = 4;
        y_coords_.at(3) = 2.5;

        x_coords_.at(4) = 3.5;
        y_coords_.at(4) = 4;

        x_coords_.at(5) = 2.5;
        y_coords_.at(5) = 4;

        this->triangles_.resize(this->triangle_size_);

        // triangle 1
        temp_triangle.ax = 1;
        temp_triangle.ay = 1;
        temp_triangle.a_index = 0;

        temp_triangle.bx = 2;
        temp_triangle.by = 2.5;
        temp_triangle.b_index = 1;

        temp_triangle.cx = 3;
        temp_triangle.cy = 1;
        temp_triangle.c_index = 2;
        this->triangles_.at(0) = temp_triangle;

        // triangle 2
        temp_triangle.ax = 1;
        temp_triangle.ay = 1;
        temp_triangle.a_index = 0;

        temp_triangle.bx = 2;
        temp_triangle.by = 2.5;
        temp_triangle.b_index = 1;

        temp_triangle.cx = 2.5;
        temp_triangle.cy = 4;
        temp_triangle.c_index = 5;
        this->triangles_.at(1) = temp_triangle;

        // triangle 3
        temp_triangle.ax = 4;
        temp_triangle.ay = 2.5;
        temp_triangle.a_index = 3;

        temp_triangle.bx = 2;
        temp_triangle.by = 2.5;
        temp_triangle.b_index = 1;

        temp_triangle.cx = 3;
        temp_triangle.cy = 1;
        temp_triangle.c_index = 2;
        this->triangles_.at(2) = temp_triangle;

        // triangle 4
        temp_triangle.ax = 4;
        temp_triangle.ay = 2.5;
        temp_triangle.a_index = 3;

        temp_triangle.bx = 3.5;
        temp_triangle.by = 4;
        temp_triangle.b_index = 4;

        temp_triangle.cx = 2.5;
        temp_triangle.cy = 4;
        temp_triangle.c_index = 5;
        this->triangles_.at(3) = temp_triangle;

        // triangle 5
        temp_triangle.bx = 2;
        temp_triangle.by = 2.5;
        temp_triangle.b_index = 1;

        temp_triangle.ax = 4;
        temp_triangle.ay = 2.5;
        temp_triangle.a_index = 3;

        temp_triangle.cx = 2.5;
        temp_triangle.cy = 4;
        temp_triangle.c_index = 5;
        this->triangles_.at(4) = temp_triangle;
    }

    void ReadBigTestData(){
        alus::delaunay::DelaunayTriangle2D temp_triangle;
        double temp_index;
        double dump_x, dump_y; //test data also includes slave figures. We don't need those for this test.
        std::ifstream rg_az_stream(this->az_rg_data_file_);
        std::ifstream triangles_stream(this->triangles_data_file_);

        if(!rg_az_stream.is_open()){
            throw std::ios::failure("masterSlaveAzRgData.txt is not open.");
        }
        if(!triangles_stream.is_open()){
            throw std::ios::failure("masterTrianglesTestData.txt is not open.");
        }

        rg_az_stream>> this->width_ >> this->height_;
        triangles_stream >> this->triangle_size_;

        int coord_size = this->width_ * this->height_;

        this->x_coords_.resize(coord_size);
        this->y_coords_.resize(coord_size);
        this->triangles_.resize(this->triangle_size_);


        for(int i=0; i<coord_size; i++){
            rg_az_stream >> this->x_coords_.at(i) >> this->y_coords_.at(i) >> dump_x >> dump_y;
            this->y_coords_.at(i) = this->y_coords_.at(i) * this->RG_AZ_RATIO;
        }

        for(int i=0; i<this->triangle_size_; i++){
            triangles_stream >> temp_triangle.ax >> temp_triangle.ay >> temp_index;
            temp_index += 0.001; //fixing a possible float inaccuracy
            temp_triangle.a_index = (int)temp_index;

            triangles_stream >> temp_triangle.bx >> temp_triangle.by >> temp_index;
            temp_index += 0.001; //fixing a possible float inaccuracy
            temp_triangle.b_index = (int)temp_index;

            triangles_stream >> temp_triangle.cx >> temp_triangle.cy >> temp_index;
            temp_index += 0.001; //fixing a possible float inaccuracy
            temp_triangle.c_index = (int)temp_index;

            this->triangles_.at(i) = temp_triangle;
        }

        rg_az_stream.close();
        triangles_stream.close();
    }

    void HostToDevice(){
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_x_coords_, this->width_* this->height_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_y_coords_, this->width_* this->height_ * sizeof(double)));
        CHECK_CUDA_ERR(cudaMalloc((void**)&this->device_triangles_, this->triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2Dgpu)));

        CHECK_CUDA_ERR(cudaMemcpy(this->device_x_coords_, this->x_coords_.data(), this->width_* this->height_ * sizeof(double), cudaMemcpyHostToDevice));
        CHECK_CUDA_ERR(cudaMemcpy(this->device_y_coords_, this->y_coords_.data(), this->width_* this->height_ * sizeof(double), cudaMemcpyHostToDevice));
    }
    void DeviceToHost() {
        CHECK_CUDA_ERR(cudaMemcpy(this->triangles_.data(), this->device_triangles_, this->triangle_size_ * sizeof(alus::delaunay::DelaunayTriangle2Dgpu), cudaMemcpyDeviceToHost));
    }
    void DeviceFree() {
        if(this->device_x_coords_ != nullptr){
            cudaFree(this->device_x_coords_);
            this->device_x_coords_ = nullptr;
        }
        if(this->device_y_coords_ != nullptr){
            cudaFree(this->device_y_coords_);
            this->device_y_coords_ = nullptr;
        }
        if(this->device_triangles_ != nullptr){
            cudaFree(this->device_triangles_);
            this->device_triangles_ = nullptr;
        }
    }

};

const std::string DelaunayTriangulatonTester::az_rg_data_file_ = "./goods/backgeocoding/masterSlaveAzRgData.txt";
const std::string DelaunayTriangulatonTester::triangles_data_file_ = "./goods/backgeocoding/masterTrianglesTestData.txt";

//don't run this unless you are developing the delaunay gpu algorithm or it has been finished.
/*TEST(DelaunayTest, TriangulationTest){
    alus::delaunay::DelaunayTriangle2D temp_triangle;
    DelaunayTriangulatonTester tester;
    tester.readTestData();
    tester.HostToDevice();

    CHECK_CUDA_ERR(alus::delaunay::LaunchDelaunayTriangulation(tester.device_x_coords_, tester.device_y_coords_, tester.width_, tester.height_, tester.device_triangles_));

    tester.DeviceToHost();

    for(int i=0; i<tester.triangle_size_; i++){
        temp_triangle = tester.triangles_.at(i);
        std::cout<<i<<") "<< temp_triangle.ax << " " << temp_triangle.ay << " " ;
        std::cout<<  temp_triangle.bx << " " << temp_triangle.by << " ";
        std::cout<<  temp_triangle.cx << " " << temp_triangle.cy;
        std::cout<< " previous: " << temp_triangle.previous<< std::endl;
    }

}*/

TEST(DelaunayTest, SmallCPUTriangulationTest){
    DelaunayTriangulatonTester tester;
    tester.ReadSmallTestData();

    alus::delaunay::DelaunayTriangulator trianglulator;
    trianglulator.TriangulateCPU(tester.x_coords_.data(), tester.y_coords_.data(), tester.width_);

    int count = alus::EqualsTriangles(trianglulator.host_triangles_.data(), tester.triangles_.data(), trianglulator.triangle_count_, 0.00001);
    EXPECT_EQ(count,0) << "Triangle results do not match. Mismatches: " << count << '\n';
}

//TODO: We will bring this back once we have the full algorithm, so we can test speeds and accuracies of end results.
/*TEST(DelaunayTest, BigCPUTriangulationTest2){
    DelaunayTriangulatonTester tester;
    tester.readBigTestData();

    alus::delaunay::DelaunayTriangulator trianglulator;
    trianglulator.TriangulateCPU(tester.x_coords_.data(), tester.y_coords_.data(), tester.width_ * tester.height_);
    std::cout <<"nr of triangles: " << trianglulator.triangle_count_ <<std::endl;

    int count = alus::EqualsTriangles(trianglulator.host_triangles_.data(), tester.triangles_.data(), trianglulator.triangle_count_, 0.00001);
    EXPECT_EQ(count,0) << "Triangle results do not match. Mismatches: " << count << '\n';

}*/

TEST(DelaunayTest, BigCPUTriangulationTest){
    DelaunayTriangulatonTester tester;
    tester.ReadBigTestData();

    alus::delaunay::DelaunayTriangulator trianglulator;
    trianglulator.TriangulateCPU2(tester.x_coords_.data(), tester.y_coords_.data(), tester.width_ * tester.height_);
    std::cout <<"nr of triangles: " << trianglulator.triangle_count_ <<std::endl;

    int count = alus::EqualsTriangles(trianglulator.host_triangles_.data(), tester.triangles_.data(), trianglulator.triangle_count_, 0.00001);
    EXPECT_EQ(count,0) << "Triangle results do not match. Mismatches: " << count << '\n';
}

}//namespace