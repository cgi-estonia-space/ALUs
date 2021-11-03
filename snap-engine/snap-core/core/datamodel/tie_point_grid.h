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
#pragma once

#include <cstddef>
#include <memory>
#include <stdexcept>
#include <string_view>
#include <vector>

#include "ceres-core/core/i_progress_monitor.h"
#include "raster_data_node.h"

namespace alus {
namespace snapengine {

class PixelsAreReadOnlyException : public std::runtime_error {
public:
    PixelsAreReadOnlyException() : runtime_error("pixels are read-only in tie-point grids") {}
};

/**
 * A tie-point grid contains the data for geophysical parameter in remote sensing data products. Tie-point grid are
 * two-dimensional images which hold their pixel values (samples) in a {@code float} array. <p>
 * <p>
 * Usually, tie-point grids are a sub-sampling of a data product's scene resolution.
 *
 * original java version author Norman Fomferra
 */
class TiePointGrid : public virtual RasterDataNode {
private:
    int grid_width_;
    int grid_height_;
    double offset_x_;
    double offset_y_;
    double sub_sampling_x_;
    double sub_sampling_y_;
    int discontinuity_;

    //   todo:these need to be in main memory if using multi threading (not cpu cache)
    std::shared_ptr<TiePointGrid> sin_grid_;
    std::shared_ptr<TiePointGrid> cos_grid_;
    std::shared_ptr<ProductData> raster_data_;

    std::shared_ptr<ProductData> ReadGridData();

    double Interpolate(double wi, double wj, int i0, int j0);
    bool IsDiscontNotInit() { return sin_grid_ == nullptr || cos_grid_ == nullptr; }
    void InitDiscont();

public:
    /**
     * The discontinuity of the tie point values shall be detected automatically.
     */
    static constexpr int DISCONT_AUTO = -1;

    /**
     * Tie point values are assumed to have none discontinuities.
     */
    static constexpr int DISCONT_NONE = 0;

    /**
     * Tie point values have angles in the range -180...+180 degrees and may comprise a discontinuity at 180 (resp.
     * -180) degrees.
     */
    static constexpr int DISCONT_AT_180 = 180;

    /**
     * Tie point values have are angles in the range 0...+360 degrees and may comprise a discontinuity at 360 (resp. 0)
     * degrees.
     */
    static constexpr int DISCONT_AT_360 = 360;

    /**
     * Constructs a new {@code TiePointGrid} with the given tie point grid properties.
     *
     * @param name         the name of the new object
     * @param gridWidth    the width of the tie-point grid in pixels
     * @param gridHeight   the height of the tie-point grid in pixels
     * @param offsetX      the X co-ordinate of the first (upper-left) tie-point in pixels
     * @param offsetY      the Y co-ordinate of the first (upper-left) tie-point in pixels
     * @param subSamplingX the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                     this tie-point grid belongs to. Must not be less than one.
     * @param subSamplingY the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                     this tie-point grid belongs to. Must not be less than one.
     */
    TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                 double sub_sampling_x, double sub_sampling_y);

    /**
     * Constructs a new {@code TiePointGrid} with the given tie point grid properties.
     *
     * @param name         the name of the new object
     * @param gridWidth    the width of the tie-point grid in pixels
     * @param gridHeight   the height of the tie-point grid in pixels
     * @param offsetX      the X co-ordinate of the first (upper-left) tie-point in pixels
     * @param offsetY      the Y co-ordinate of the first (upper-left) tie-point in pixels
     * @param subSamplingX the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                     this tie-point grid belongs to. Must not be less than one.
     * @param subSamplingY the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                     this tie-point grid belongs to. Must not be less than one.
     * @param tiePoints    the tie-point data values, must be an array of the size {@code gridWidth * gridHeight}
     */

    TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                 double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points);

    /**
     * Constructs a new {@code TiePointGrid} with the given tie point grid properties.
     *
     * @param name           the name of the new object
     * @param gridWidth      the width of the tie-point grid in pixels
     * @param gridHeight     the height of the tie-point grid in pixels
     * @param offsetX        the X co-ordinate of the first (upper-left) tie-point in pixels
     * @param offsetY        the Y co-ordinate of the first (upper-left) tie-point in pixels
     * @param subSamplingX   the sub-sampling in X-direction given in the pixel co-ordinates of the data product to
     * which this tie-point grid belongs to. Must not be less than one.
     * @param subSamplingY   the sub-sampling in X-direction given in the pixel co-ordinates of the data product to
     * which this tie-point grid belongs to. Must not be less than one.
     * @param tiePoints      the tie-point data values, must be an array of the size {@code gridWidth * gridHeight}
     * @param containsAngles if true, the {@link #getDiscontinuity() angular discontinuity} is derived from the provided
     * tie-point data values
     */
    TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                 double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points,
                 bool contains_angles);

    /**
     * Constructs a new {@code TiePointGrid} with the given tie point grid properties.
     *
     * @param name          the name of the new object
     * @param gridWidth     the width of the tie-point grid in pixels
     * @param gridHeight    the height of the tie-point grid in pixels
     * @param offsetX       the X co-ordinate of the first (upper-left) tie-point in pixels
     * @param offsetY       the Y co-ordinate of the first (upper-left) tie-point in pixels
     * @param subSamplingX  the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                      this tie-point grid belongs to. Must not be less than one.
     * @param subSamplingY  the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which
     *                      this tie-point grid belongs to. Must not be less than one.
     * @param tiePoints     the tie-point data values, must be an array of the size {@code gridWidth * gridHeight}
     * @param discontinuity the discontinuity mode, can be either {@link #DISCONT_NONE}, {@link #DISCONT_AUTO}, {@link
     * #DISCONT_AT_180} or
     *                      {@link #DISCONT_AT_360}
     */
    TiePointGrid(std::string_view name, int grid_width, int grid_height, double offset_x, double offset_y,
                 double sub_sampling_x, double sub_sampling_y, const std::vector<float>& tie_points, int discontinuity);

    /**
     * @return The grid's width (= number of columns).
     */
    int GetGridWidth() const { return grid_width_; }

    /**
     * @return The grid's height (= number of rows).
     */
    int GetGridHeight() const { return grid_height_; }

    /**
     * Retrieves the x co-ordinate of the first (upper-left) tie-point in pixels.
     */
    double GetOffsetX() const { return offset_x_; }

    /**
     * Retrieves the y co-ordinate of the first (upper-left) tie-point in pixels.
     */
    double GetOffsetY() const { return offset_y_; }

    /**
     * Returns the sub-sampling in X-direction given in the pixel co-ordinates of the data product to which this
     * tie-point grid belongs to.
     *
     * @return the sub-sampling in X-direction, never less than one.
     */
    double GetSubSamplingX() const { return sub_sampling_x_; }

    /**
     * Returns the sub-sampling in Y-direction given in the pixel co-ordinates of the data product to which this
     * tie-point grid belongs to.
     *
     * @return the sub-sampling in Y-direction, never less than one.
     */
    double GetSubSamplingY() const { return sub_sampling_y_; }

    /**
     * @return The data array representing the single tie-points.
     */
    std::vector<float> GetTiePoints();

    /**
     * @return The data buffer representing the single tie-points.
     */
    std::shared_ptr<ProductData> GetGridData();

    /**
     * Returns {@code true}
     *
     * @return true
     */
    bool IsFloatingPointType() override { return true; }

    /**
     * Returns the geophysical data type of this {@code RasterDataNode}. The value returned is always one of the
     * {@code ProductData.TYPE_XXX} constants.
     *
     * @return the geophysical data type
     *
     * @see ProductData
     */

    int GetGeophysicalDataType() override { return ProductData::TYPE_FLOAT32; }

    void SetData(const std::shared_ptr<ProductData>& data) override;

    /**
     * Gets the linear interpolated raster data containing
     * {@link #getRasterWidth() rasterWidth} x {@link #getRasterHeight() rasterHeight} samples.
     *
     * @return The raster data for this tie-point grid.
     */
    std::shared_ptr<ProductData> GetRasterData() override;

    /**
     * @return The native width of the raster in pixels.
     */
    int GetRasterWidth() override;

    /**
     * @return The native height of the raster in pixels.
     */
    int GetRasterHeight() override;

    /**
     * Determines the angular discontinuity of the given tie point values.
     *
     * @return the angular discontinuity, will always be either {@link #DISCONT_AT_180} or
     * {@link #DISCONT_AT_360}
     */
    static int GetDiscontinuity(std::vector<float> tie_points);

    /**
     * Gets the angular discontinuity.
     *
     * @return the angular discontinuity, will always be either {@link #DISCONT_NONE}, {@link #DISCONT_AUTO}, {@link
     * #DISCONT_AT_180} or
     * {@link #DISCONT_AT_360}
     */
    int GetDiscontinuity() const { return discontinuity_; }

    /**
     * Sets the angular discontinuity.
     *
     * @param discontinuity angular discontinuity, can be either {@link #DISCONT_NONE}, {@link #DISCONT_AUTO}, {@link
     * #DISCONT_AT_180} or
     *                      {@link #DISCONT_AT_360}
     */
    void SetDiscontinuity(int discontinuity);

    /**
     * The method will always fail on tie-point grids as they are read-only.
     *
     * @param rasterData The raster data whose reference will be stored.
     */
    void SetRasterData([[maybe_unused]] std::shared_ptr<ProductData> raster_data) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * Gets the interpolated sample for the pixel located at (x,y) as an integer value. <p>
     * <p>
     * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
     *
     * @param x The X co-ordinate of the pixel location
     * @param y The Y co-ordinate of the pixel location
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     */

    int GetPixelInt(int x, int y) override;

    void Dispose() override;

    /**
     * Computes the interpolated sample for the pixel located at (x,y). <p>
     * <p>
     * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
     *
     * @param x The X co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @param y The Y co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     */
    float GetPixelFloat(int x, int y) override;

    /**
     * Computes the interpolated sample for the pixel located at (x,y) given as floating point co-ordinates. <p>
     * <p>
     * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
     *
     * @param x The X co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @param y The Y co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     */
    float GetPixelFloat(float x, float y) { return static_cast<float>(GetPixelDouble(x, y)); }

    /**
     * Gets the interpolated sample for the pixel located at (x,y) as a double value. <p>
     * <p>
     * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
     *
     * @param x The X co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @param y The Y co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     */
    double GetPixelDouble(int x, int y) override;

    /**
     * Gets the interpolated sample for the pixel located at (x,y) as a double value. <p>
     * <p>
     * If the pixel co-ordinates given by (x,y) are not covered by this tie-point grid, the method extrapolates.
     *
     * @param x The X co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @param y The Y co-ordinate of the pixel location, given in the pixel co-ordinates of the data product to which
     *          this tie-point grid belongs to.
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     */
    double GetPixelDouble(double x, double y);

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixelInt([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int pixel_value) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixelFloat([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] float pixel_value) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixelDouble([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] double pixel_value) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * Retrieves an array of tie point data interpolated to the product with and height as integer array. If the given
     * array is {@code null} a new one was created and returned.
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the integer array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<int> GetPixels(int x, int y, int w, int h, std::vector<int> pixels,
                               std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves an array of tie point data interpolated to the product width and height as float array. If the given
     * array is {@code null} a new one is created and returned.
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the float array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<double> GetPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                  std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves an array of tie point data interpolated to the product with and height as double array. If the given
     * array is {@code null} a new one was created and returned.
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the double array to be filled with data
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<float> GetPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                 std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                   [[maybe_unused]] std::vector<int> pixels) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                   [[maybe_unused]] std::vector<float> pixels) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void SetPixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                   [[maybe_unused]] std::vector<double> pixels) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * Retrieves an array of tie point data interpolated to the product with and height as float array. If the given
     * array is {@code null} a new one was created and returned.
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the integer array to be filled with data
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<int> ReadPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                std::shared_ptr<ceres::IProgressMonitor> pm) override {
        return GetPixels(x, y, w, h, pixels, pm);
    }

    /**
     * Retrieves an array of tie point data interpolated to the product with and height as float array. If the given
     * array is {@code null} a new one was created and returned. *
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the float array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<float> ReadPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                  std::shared_ptr<ceres::IProgressMonitor> pm) override {
        return GetPixels(x, y, w, h, pixels, pm);
    }

    /**
     * Retrieves an array of tie point data interpolated to the product with and height as double array. If the given
     * array is {@code null} a new one was created and returned.
     *
     * @param x      the x coordinate of the array to be read
     * @param y      the y coordinate of the array to be read
     * @param w      the width of the array to be read
     * @param h      the height of the array to be read
     * @param pixels the double array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than {@code w*h}.
     */
    std::vector<double> ReadPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                   std::shared_ptr<ceres::IProgressMonitor> pm) override {
        return GetPixels(x, y, w, h, pixels, pm);
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void WritePixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                     [[maybe_unused]] std::vector<int> pixels,
                     [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void WritePixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                     [[maybe_unused]] std::vector<float> pixels,
                     [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * This method is not implemented because pixels are read-only in tie-point grids.
     */
    void WritePixels([[maybe_unused]] int x, [[maybe_unused]] int y, [[maybe_unused]] int w, [[maybe_unused]] int h,
                     [[maybe_unused]] std::vector<double> pixels,
                     [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * Reads raster data from this dataset into the user-supplied raster data buffer. <p>
     * <p>
     * This method always directly (re-)reads this tie-point grid's data from its associated data source into the given
     * data buffer.
     *
     * @param offsetX    the X-offset in the raster co-ordinates where reading starts
     * @param offsetY    the Y-offset in the raster co-ordinates where reading starts
     * @param width      the width of the raster data buffer
     * @param height     the height of the raster data buffer
     * @param rasterData a raster data buffer receiving the pixels to be read
     * @param pm         a monitor to inform the user about progress
     * @throws java.io.IOException      if an I/O error occurs
     * @throws IllegalArgumentException if the raster is null
     * @throws IllegalStateException    if this product raster was not added to a product so far, or if the product to
     *                                  which this product raster belongs to, has no associated product reader
     * @see ProductReader#readBandRasterData(Band, int, int, int, int, ProductData,
     * com.bc.ceres.core.ceres::IProgressMonitor)
     */
    void ReadRasterData(int offset_x, int offset_y, int width, int height, std::shared_ptr<ProductData> raster_data,
                        std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * {@inheritDoc}
     */
    void ReadRasterDataFully([[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        //        todo: this can be unwrapped at later time when we have more clarity
        GetGridData();  // trigger reading the grid points
    }

    /**
     * {@inheritDoc}
     */
    void WriteRasterData([[maybe_unused]] int offset_x, [[maybe_unused]] int offset_y, [[maybe_unused]] int width,
                         [[maybe_unused]] int height, [[maybe_unused]] std::shared_ptr<ProductData> raster_data,
                         [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        throw PixelsAreReadOnlyException();
    }

    /**
     * {@inheritDoc}
     */
    void WriteRasterDataFully([[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override {
        //        todo: this can be unwrapped at later time when we have more clarity
        throw PixelsAreReadOnlyException();
    }

    //    todo: implement only if we need this
    //    protected RenderedImage CreateSourceImage() override {
    //        MultiLevelModel model = CreateMultiLevelModel();
    //        return new DefaultMultiLevelImage(new AbstractMultiLevelSource(model) {
    //            @Override public RenderedImage createImage(int level) {
    //                return new TiePointGridOpImage(TiePointGrid.this, ResolutionLevel.create(getModel(), level));
    //            }
    //        });
    //    }

    std::shared_ptr<TiePointGrid> CloneTiePointGrid();

    // ////////////////////////////////////////////////////////////////////////
    // Public static helpers

    static std::shared_ptr<TiePointGrid> CreateZenithFromElevationAngleTiePointGrid(
        const std::shared_ptr<TiePointGrid>& elevation_angle_grid);

    static std::shared_ptr<TiePointGrid> CreateSubset(const std::shared_ptr<TiePointGrid>& source_tie_point_grid,
                                                      const std::shared_ptr<ProductSubsetDef>& subset_def);

protected:
    template <typename T>
    static std::vector<T> EnsureMinLengthArray(std::vector<T> array, std::size_t length) {
        //            todo:this logic changed a bit, might want to look over if we have issues
        if (array.empty()) {
            return std::vector<T>(length);
        }

        if (array.size() < length) {
            throw std::invalid_argument("The length of the given array is less than " + std::to_string(length));
        }

        return array;
    }
};
}  // namespace snapengine
}  // namespace alus
