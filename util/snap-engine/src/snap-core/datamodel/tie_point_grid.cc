/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.TiePointGrid.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/snap-engine). It was originally stated:
 *
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
#include "snap-core/datamodel/tie_point_grid.h"

#include <algorithm>
#include <cmath>
#include <iostream>

#include "ceres-core/ceres_assert.h"
#include "snap-core/dataio/product_subset_def.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/util/math/math_utils.h"

namespace alus {
namespace snapengine {

TiePointGrid::TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                           double sub_sampling_x, double sub_sampling_y)
    : RasterDataNode(name, ProductData::TYPE_FLOAT32, grid_width * grid_height) {
    Assert::Argument(grid_width >= 2, "gridWidth >= 2");
    Assert::Argument(grid_height >= 2, "gridHeight >= 2");
    Assert::Argument(sub_sampling_x > 0.0F, "subSamplingX > 0.0");
    Assert::Argument(sub_sampling_y > 0.0F, "subSamplingY > 0.0");
    grid_width_ = grid_width;
    grid_height_ = grid_height;
    offset_x_ = offset_x;
    offset_y_ = offset_y;
    sub_sampling_x_ = sub_sampling_x;
    sub_sampling_y_ = sub_sampling_y;
    discontinuity_ = DISCONT_NONE;
}
TiePointGrid::TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                           double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points)
    : TiePointGrid(name, grid_width, grid_height, offset_x, offset_y, sub_sampling_x, sub_sampling_y, tie_points,
                   DISCONT_NONE) {}
TiePointGrid::TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                           double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points,
                           bool contains_angles)
    : TiePointGrid(name, grid_width, grid_height, offset_x, offset_y, sub_sampling_x, sub_sampling_y, tie_points) {
    Assert::Argument(tie_points.size() == (static_cast<size_t>(grid_width * grid_height)),
                     "tiePoints.length == gridWidth * gridHeight");
    if (contains_angles) {
        SetDiscontinuity(GetDiscontinuity(tie_points));
    }
}
TiePointGrid::TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                           double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points,
                           int discontinuity)
    : TiePointGrid(name, grid_width, grid_height, offset_x, offset_y, sub_sampling_x, sub_sampling_y) {
    Assert::Argument(tie_points.size() == (static_cast<size_t>(grid_width * grid_height)),
                     "tiePoints.length == gridWidth * gridHeight");
    Assert::Argument(
        discontinuity == DISCONT_NONE || discontinuity == DISCONT_AT_180 || discontinuity == DISCONT_AT_360,
        "discontinuity");
    discontinuity_ = discontinuity;
    SetData(ProductData::CreateInstance(tie_points));
}
std::vector<float> TiePointGrid::GetTiePoints() {
    auto data = GetGridData()->GetElems();
    if (data.type() == typeid(std::vector<float>)) {
        return std::any_cast<std::vector<float>>(data);
    }
    throw std::runtime_error("GetTiePoints expected std::any with type std::vector<float>, but got something else");
}
std::shared_ptr<ProductData> TiePointGrid::GetGridData() {
    if (GetData() == nullptr) {
        try {
            SetData(ReadGridData());
        } catch (std::exception& e) {
            //                    todo:decide if LOG(INFO) or not
            //                    LOG(ERROR) << "Unable to load TPG: " << e.what();
            std::cerr << "Unable to load TPG: " << e.what() << std::endl;
        }
    }
    return GetData();
}
void TiePointGrid::SetData(const std::shared_ptr<ProductData>& data) {
    DataNode::SetData(data);
    if (GetDiscontinuity() == DISCONT_AUTO) {
        SetDiscontinuity(GetDiscontinuity(std::any_cast<std::vector<float>>(data->GetElems())));
    }
}

std::shared_ptr<ProductData> TiePointGrid::GetRasterData() {
    int width = GetRasterWidth();
    int height = GetRasterHeight();
    std::shared_ptr<ProductData> grid_data = GetGridData();
    // A tie-point grid's data may have the same dimensions as the requested raster data:
    // In this case we can simply return it instead of holding another one in this.rasterData.
    if (grid_data->GetNumElems() == width * height) {
        return grid_data;
    }
    // Create a new one by interpolation.
    if (raster_data_ == nullptr) {
        //            todo: make sure it is thread safe here
        //            synchronized (this) {
        //            }

        raster_data_ = CreateCompatibleRasterData(width, height);
        // GetPixels will interpolate between tie points
        GetPixels(0, 0, width, height, std::any_cast<std::vector<float>>(raster_data_->GetElems()), nullptr);
    }
    return raster_data_;
}
int TiePointGrid::GetPixelInt(int x, int y) { return static_cast<int>(round(GetPixelDouble(x, y))); }
void TiePointGrid::Dispose() {
    if (cos_grid_ != nullptr) {
        cos_grid_->Dispose();
        cos_grid_ = nullptr;
    }
    if (sin_grid_ != nullptr) {
        sin_grid_->Dispose();
        sin_grid_ = nullptr;
    }
    RasterDataNode::Dispose();
}
float TiePointGrid::GetPixelFloat(int x, int y) { return static_cast<float>(GetPixelDouble(x + 0.5F, y + 0.5F)); }
double TiePointGrid::GetPixelDouble(int x, int y) { return GetPixelDouble(x + 0.5, y + 0.5); }

std::shared_ptr<TiePointGrid> TiePointGrid::CloneTiePointGrid() {
    std::vector<float> src_tie_points = GetTiePoints();
    std::vector<float> dest_tie_points(src_tie_points.size());
    std::copy(src_tie_points.begin(), src_tie_points.end(), dest_tie_points.begin());
    std::shared_ptr<TiePointGrid> clone =
        std::make_shared<TiePointGrid>(GetName(), GetGridWidth(), GetGridHeight(), GetOffsetX(), GetOffsetY(),
                                       GetSubSamplingX(), GetSubSamplingY(), dest_tie_points, GetDiscontinuity());
    clone->SetUnit(GetUnit());
    clone->SetDescription(GetDescription());
    return clone;
}
std::shared_ptr<TiePointGrid> TiePointGrid::CreateZenithFromElevationAngleTiePointGrid(
    const std::shared_ptr<TiePointGrid>& elevation_angle_grid) {
    std::vector<float> elevation_angles = elevation_angle_grid->GetTiePoints();
    std::vector<float> zenith_angles(elevation_angles.size());
    for (std::size_t i = 0; i < zenith_angles.size(); i++) {
        zenith_angles.at(i) = 90.0F - elevation_angles.at(i);
    }
    return std::make_shared<TiePointGrid>(elevation_angle_grid->GetName(), elevation_angle_grid->GetGridWidth(),
                                          elevation_angle_grid->GetGridHeight(), elevation_angle_grid->GetOffsetX(),
                                          elevation_angle_grid->GetOffsetY(), elevation_angle_grid->GetSubSamplingX(),
                                          elevation_angle_grid->GetSubSamplingY(), zenith_angles);
}

double TiePointGrid::Interpolate(double wi, double wj, int i0, int j0) {
    std::vector<float> tie_points = GetTiePoints();
    int w = GetGridWidth();
    int i1 = i0 + 1;
    int j1 = j0 + 1;
    return MathUtils::Interpolate2D(wi, wj, tie_points.at(i0 + j0 * w), tie_points.at(i1 + j0 * w),
                                    tie_points.at(i0 + j1 * w), tie_points.at(i1 + j1 * w));
}

void TiePointGrid::InitDiscont() {
    auto base = SharedFromBase<TiePointGrid>();
    std::vector<float> tie_points = base->GetTiePoints();
    std::vector<float> sin_tie_points(tie_points.size());
    std::vector<float> cos_tie_points(tie_points.size());
    for (std::size_t i = 0; i < tie_points.size(); i++) {
        double tie_point = tie_points.at(i);
        sin_tie_points.at(i) = static_cast<float>(sin(MathUtils::DTOR * tie_point));
        cos_tie_points.at(i) = static_cast<float>(cos(MathUtils::DTOR * tie_point));
    }
    sin_grid_ = std::make_shared<TiePointGrid>(base->GetName(), base->GetGridWidth(), base->GetGridHeight(),
                                               base->GetOffsetX(), base->GetOffsetY(), base->GetSubSamplingX(),
                                               base->GetSubSamplingY(), sin_tie_points);
    cos_grid_ = std::make_shared<TiePointGrid>(base->GetName(), base->GetGridWidth(), base->GetGridHeight(),
                                               base->GetOffsetX(), base->GetOffsetY(), base->GetSubSamplingX(),
                                               base->GetSubSamplingY(), cos_tie_points);
}
double TiePointGrid::GetPixelDouble(double x, double y) {
    if (discontinuity_ != DISCONT_NONE) {
        if (IsDiscontNotInit()) {
            InitDiscont();
        }
        double sin_angle = sin_grid_->GetPixelDouble(x, y);
        double cos_angle = cos_grid_->GetPixelDouble(x, y);
        double v = MathUtils::RTOD * atan2(sin_angle, cos_angle);
        if (discontinuity_ == DISCONT_AT_360 && v < 0.0) {
            return 360.0F + v;  // = 180 + (180 - abs(v))
        }
        return v;
    }
    double fi = (x - offset_x_) / sub_sampling_x_;
    double fj = (y - offset_y_) / sub_sampling_y_;
    int i = MathUtils::FloorAndCrop(fi, 0, GetGridWidth() - 2);
    int j = MathUtils::FloorAndCrop(fj, 0, GetGridHeight() - 2);
    return Interpolate(fi - i, fj - j, i, j);
}
std::vector<float> TiePointGrid::GetPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                           std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    std::vector<double> fpixels = GetPixels(x, y, w, h, std::vector<double>(0), pm);
    for (std::size_t i = 0; i < fpixels.size(); i++) {
        pixels.at(i) = static_cast<float>(fpixels.at(i));
    }
    return pixels;
}
std::vector<int> TiePointGrid::GetPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                         std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    std::vector<double> fpixels = GetPixels(x, y, w, h, std::vector<double>(0), pm);
    for (std::size_t i = 0; i < fpixels.size(); i++) {
        pixels.at(i) = static_cast<int>(round(fpixels.at(i)));
    }
    return pixels;
}
std::vector<double> TiePointGrid::GetPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                            [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    pixels = EnsureMinLengthArray(pixels, w * h);
    if (discontinuity_ != DISCONT_NONE) {
        if (IsDiscontNotInit()) {
            InitDiscont();
        }
        int i = 0;
        for (int y_coordinate = y; y_coordinate < y + h; y_coordinate++) {
            for (int x_coordinate = x; x_coordinate < x + w; x_coordinate++) {
                pixels.at(i) = GetPixelDouble(x_coordinate, y_coordinate);
                i++;
            }
        }
    } else {
        double x0 = 0.5F - offset_x_;
        double y0 = 0.5F - offset_y_;
        int x1 = x;
        int y1 = y;
        int x2 = x + w - 1;
        int y2 = y + h - 1;
        int ni = GetGridWidth();
        int nj = GetGridHeight();
        int i;
        int j;
        double fi;
        double fj;
        double wi;
        double wj;
        int pos = 0;
        for (y = y1; y <= y2; y++) {
            fj = (y + y0) / sub_sampling_y_;
            j = MathUtils::FloorAndCrop(fj, 0, nj - 2);
            wj = fj - j;
            for (x = x1; x <= x2; x++) {
                fi = (x + x0) / sub_sampling_x_;
                i = MathUtils::FloorAndCrop(fi, 0, ni - 2);
                wi = fi - i;
                pixels[pos++] = Interpolate(wi, wj, i, j);
            }
        }
    }
    return pixels;
}
void TiePointGrid::ReadRasterData(int offset_x, int offset_y, int width, int height,
                                  std::shared_ptr<ProductData> raster_data,
                                  [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) {
    std::shared_ptr<ProductData> src = GetRasterData();
    int i_src;
    int i_dest = 0;
    //        pm.beginTask("Reading raster data...", height);
    //        try {
    for (int y = 0; y < height; y++) {
        for (int x = 0; x < width; x++) {
            i_src = (offset_y + y) * width + (offset_x + x);
            raster_data->SetElemDoubleAt(i_dest, src->GetElemDoubleAt(i_src));
            i_dest++;
        }
        //                pm.worked(1);
    }
    //        } finally {
    //            pm.done();
    //        }
}
void TiePointGrid::SetDiscontinuity(int discontinuity) {
    if (discontinuity != DISCONT_NONE && discontinuity != DISCONT_AUTO && discontinuity != DISCONT_AT_180 &&
        discontinuity != DISCONT_AT_360) {
        throw std::invalid_argument("unsupported discontinuity mode");
    }
    discontinuity_ = discontinuity;
}
int TiePointGrid::GetDiscontinuity(std::vector<float> tie_points) {
    // todo: replaced with simpler std::max_element approach, if this causes issues in the future implement range like
    // in java
    if (*std::max_element(tie_points.begin(), tie_points.end()) > 180.0) {
        return DISCONT_AT_360;
    }
    return DISCONT_AT_180;
}

int TiePointGrid::GetRasterHeight() {
    if (GetProduct() != nullptr) {
        return GetProduct()->GetSceneRasterHeight();
    }
    return static_cast<int>(round((GetGridHeight() - 1) * GetSubSamplingY() + 1));
}

int TiePointGrid::GetRasterWidth() {
    if (GetProduct() != nullptr) {
        return GetProduct()->GetSceneRasterWidth();
    }
    return static_cast<int>(round((GetGridWidth() - 1) * GetSubSamplingX() + 1));
}
std::shared_ptr<ProductData> TiePointGrid::ReadGridData() {
    std::shared_ptr<ProductData> product_data = CreateCompatibleRasterData(GetGridWidth(), GetGridHeight());
    GetProductReader()->ReadTiePointGridRasterData(SharedFromBase<TiePointGrid>(), 0, 0, GetGridWidth(),
                                                   GetGridHeight(), product_data, nullptr);
    return product_data;
}

std::shared_ptr<TiePointGrid> TiePointGrid::CreateSubset(const std::shared_ptr<TiePointGrid>& source_tie_point_grid,
                                                         const std::shared_ptr<ProductSubsetDef>& subset_def) {
    const int src_t_p_g_width = source_tie_point_grid->GetGridWidth();
    const int src_t_p_g_height = source_tie_point_grid->GetGridHeight();
    const double src_t_p_g_sub_sampling_x = source_tie_point_grid->GetSubSamplingX();
    const double src_t_p_g_sub_sampling_y = source_tie_point_grid->GetSubSamplingY();
    int subset_offset_x = 0;
    int subset_offset_y = 0;
    int subset_step_x = 1;
    int subset_step_y = 1;
    const int src_raster_width = source_tie_point_grid->GetRasterWidth();
    const int src_raster_height = source_tie_point_grid->GetRasterHeight();
    int subset_width = src_raster_width;
    int subset_height = src_raster_height;
    if (subset_def) {
        subset_step_x = subset_def->GetSubSamplingX();
        subset_step_y = subset_def->GetSubSamplingY();
        if (!subset_def->GetRegionMap().empty() &&
            subset_def->GetRegionMap().find(source_tie_point_grid->GetName()) != subset_def->GetRegionMap().end()) {
            subset_offset_x = subset_def->GetRegionMap().at(source_tie_point_grid->GetName()).x;
            subset_offset_y = subset_def->GetRegionMap().at(source_tie_point_grid->GetName()).y;
            subset_width = subset_def->GetRegionMap().at(source_tie_point_grid->GetName()).width;
            subset_height = subset_def->GetRegionMap().at(source_tie_point_grid->GetName()).height;
        } else if (subset_def->GetRegion() != nullptr) {
            subset_offset_x = subset_def->GetRegion()->x;
            subset_offset_y = subset_def->GetRegion()->y;
            subset_width = subset_def->GetRegion()->width;
            subset_height = subset_def->GetRegion()->height;
        }
    }

    const double new_t_p_g_sub_sampling_x = src_t_p_g_sub_sampling_x / subset_step_x;
    const double new_t_p_g_sub_sampling_y = src_t_p_g_sub_sampling_y / subset_step_y;
    const float pixel_center = 0.5f;
    const double new_t_p_g_offset_x =
        (source_tie_point_grid->GetOffsetX() - pixel_center - subset_offset_x) / subset_step_x + pixel_center;
    const double new_t_p_g_offset_y =
        (source_tie_point_grid->GetOffsetY() - pixel_center - subset_offset_y) / subset_step_y + pixel_center;
    const double new_offset_x = fmod(new_t_p_g_offset_x, new_t_p_g_sub_sampling_x);
    const double new_offset_y = fmod(new_t_p_g_offset_y, new_t_p_g_sub_sampling_y);
    const double diff_x = new_offset_x - new_t_p_g_offset_x;
    const double diff_y = new_offset_y - new_t_p_g_offset_y;
    int data_offset_x;
    if (diff_x < 0.0f) {
        data_offset_x = 0;
    } else {
        data_offset_x = static_cast<int>(round(diff_x / new_t_p_g_sub_sampling_x));
    }
    int data_offset_y;
    if (diff_y < 0.0f) {
        data_offset_y = 0;
    } else {
        data_offset_y = static_cast<int>(round(diff_y / new_t_p_g_sub_sampling_y));
    }

    int new_t_p_g_width = static_cast<int>(ceil(subset_width / src_t_p_g_sub_sampling_x)) + 2;
    if (data_offset_x + new_t_p_g_width > src_t_p_g_width) {
        new_t_p_g_width = src_t_p_g_width - data_offset_x;
    }
    int new_t_p_g_height = static_cast<int>(ceil(subset_height / src_t_p_g_sub_sampling_y)) + 2;
    if (data_offset_y + new_t_p_g_height > src_t_p_g_height) {
        new_t_p_g_height = src_t_p_g_height - data_offset_y;
    }

    std::vector<float> old_tie_points = source_tie_point_grid->GetTiePoints();
    std::vector<float> tie_points(new_t_p_g_width * new_t_p_g_height);
    for (int y = 0; y < new_t_p_g_height; y++) {
        const int src_pos = src_t_p_g_width * (data_offset_y + y) + data_offset_x;
        //        todo:check if this works like expected
        std::copy(old_tie_points.begin() + src_pos, old_tie_points.begin() + new_t_p_g_width,
                  tie_points.begin() + y * new_t_p_g_width);
    }

    auto tie_point_grid = std::make_shared<TiePointGrid>(
        source_tie_point_grid->GetName(), new_t_p_g_width, new_t_p_g_height, new_offset_x, new_offset_y,
        new_t_p_g_sub_sampling_x, new_t_p_g_sub_sampling_y, tie_points);
    tie_point_grid->SetUnit(source_tie_point_grid->GetUnit());
    tie_point_grid->SetDescription(source_tie_point_grid->GetDescription());
    tie_point_grid->SetDiscontinuity(source_tie_point_grid->GetDiscontinuity());
    return tie_point_grid;
}

}  // namespace snapengine
}  // namespace alus