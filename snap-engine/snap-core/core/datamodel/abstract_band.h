/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.AbstractBand.java
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

#include <memory>
#include <string_view>
#include <vector>

#include "snap-core/core/datamodel/raster_data_node.h"

namespace alus {
namespace ceres {
// pre-declare
class IProgressMonitor;
}  // namespace ceres
namespace snapengine {
// pre-declare
class ProductData;

/**
 * The <code>AbstractBand</code> class provides a set of pixel access methods but does not provide an implementation of
 * the actual reading and writing of pixel data from or into a raster.
 *
 * original java version authors: Norman Fomferra, Sabine Embacher
 */
class AbstractBand : public RasterDataNode {
private:
    /**
     * The raster's width.
     */
    int raster_width_;

    /**
     * The raster's height.
     */
    int raster_height_;

    std::shared_ptr<ProductData> GetRasterDataSafe();

    std::shared_ptr<ProductData> ReadSubRegionRasterData(int x, int y, int w, int h,
                                                         const std::shared_ptr<ceres::IProgressMonitor>& pm);

protected:
    //    todo: std::vector does not need such check like array, might want to check logic over when we get it working,
    //    currently since additions use index, we just port
    static std::vector<int> EnsureMinLengthArray(std::vector<int> array, int length);

    static std::vector<float> EnsureMinLengthArray(std::vector<float> array, int length);

    static std::vector<double> EnsureMinLengthArray(std::vector<double> array, int length);

public:
    AbstractBand(std::string_view name, int data_type, int raster_width, int raster_height);
    /**
     * @return The width of the raster in pixels.
     */
    int GetRasterWidth() override { return raster_width_; }

    /**
     * @return The height of the raster in pixels.
     */
    int GetRasterHeight() override { return raster_height_; }

    /**
     * Retrieves the range of pixels specified by the coordinates as integer array. Reads the data from disk if ot is
     * not in memory yet. If the data is loaded, just copies the data.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels integer array to be filled with data
     * @param pm     a monitor to inform the user about progress
     */

    void WritePixels(int x, int y, int w, int h, std::vector<int> pixels,
                     std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as float array. Reads the data from disk if ot is not
     * in memory yet. If the data is loaded, just copies the data.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels float array to be filled with data
     * @param pm     a monitor to inform the user about progress
     */
    //! java version had synchronized (only this)
    void WritePixels(int x, int y, int w, int h, std::vector<float> pixels,
                     std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as double array. Reads the data from disk if ot is not
     * in memory yet. If the data is loaded, just copies the data.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels double array to be filled with data
     * @param pm     a monitor to inform the user about progress
     */
    void WritePixels(int x, int y, int w, int h, std::vector<double> pixels,
                     std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Gets the sample for the pixel located at (x,y) as an integer value.
     *
     * @param x The X co-ordinate of the pixel location
     * @param y The Y co-ordinate of the pixel location
     * @throws NullPointerException if this band has no raster data
     * @throws java.lang.ArrayIndexOutOfBoundsException
     *                              if the co-ordinates are not in bounds
     */

    int GetPixelInt(int x, int y) override;

    /**
     * Gets the sample for the pixel located at (x,y) as a float value.
     *
     * @param x The X co-ordinate of the pixel location
     * @param y The Y co-ordinate of the pixel location
     * @throws NullPointerException if this band has no raster data
     * @throws java.lang.ArrayIndexOutOfBoundsException
     *                              if the co-ordinates are not in bounds
     */
    float GetPixelFloat(int x, int y) override;

    /**
     * Gets the sample for the pixel located at (x,y) as a double value.
     *
     * @param x The X co-ordinate of the pixel location
     * @param y The Y co-ordinate of the pixel location
     * @throws NullPointerException if this band has no raster data
     * @throws java.lang.ArrayIndexOutOfBoundsException
     *                              if the co-ordinates are not in bounds
     */
    double GetPixelDouble(int x, int y) override;

    /**
     * Sets the pixel at the given pixel co-ordinate to the given pixel value.
     *
     * @param x          The X co-ordinate of the pixel location
     * @param y          The Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value
     * @throws NullPointerException if this band has no raster data
     */
    void SetPixelInt(int x, int y, int pixel_value) override;

    /**
     * Sets the pixel at the given pixel coordinate to the given pixel value.
     *
     * @param x          The X co-ordinate of the pixel location
     * @param y          The Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value
     * @throws NullPointerException if this band has no raster data
     */
    void SetPixelFloat(int x, int y, float pixel_value) override;

    /**
     * Sets the pixel value at the given pixel coordinate to the given pixel value.
     *
     * @param x          The X co-ordinate of the pixel location
     * @param y          The Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value
     * @throws NullPointerException if this band has no raster data
     */
    void SetPixelDouble(int x, int y, double pixel_value) override;

    /**
     * Sets a range of pixels specified by the coordinates as integer array. Copies the data to the memory buffer of
     * data at the specified location. Throws exception when the target buffer is not in memory.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels integer array to be written
     * @throws NullPointerException if this band has no raster data
     */
    void SetPixels(int x, int y, int w, int h, std::vector<int> pixels) override;

    /**
     * Sets a range of pixels specified by the coordinates as float array. Copies the data to the memory buffer of data
     * at the specified location. Throws exception when the target buffer is not in memory.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels float array to be written
     */
    void SetPixels(int x, int y, int w, int h, std::vector<float> pixels) override;

    /**
     * Sets a range of pixels specified by the coordinates as double array. Copies the data to the memory buffer of data
     * at the specified location. Throws exception when the target buffer is not in memory.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels double array to be written
     */
    void SetPixels(int x, int y, int w, int h, std::vector<double> pixels) override;

    /**
     * Retrieves the band data at the given offset (x, y), width and height as integer data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader to
     * retrieve the data from the disk file. If the given buffer is <code>null</code> a new one was created and
     * returned.
     *
     * @param x      x offest of upper left corner
     * @param y      y offset of upper left corner
     * @param w      width of the desired data array
     * @param h      height of the desired data array
     * @param pixels array of integer pixels to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */
    std::vector<int> ReadPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the band data at the given offset (x, y), width and height as float data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader to
     * retrieve the data from the disk file. If the given buffer is <code>null</code> a new one was created and
     * returned.
     *
     * @param x      x offest of upper left corner
     * @param y      y offset of upper left corner
     * @param w      width of the desired data array
     * @param h      height of the desired data array
     * @param pixels array of float pixels to be filled with data.
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */
    std::vector<float> ReadPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                  [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the band data at the given offset (x, y), width and height as double data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader to
     * retrieve the data from the disk file. If the given buffer is <code>null</code> a new one was created and
     * returned.
     *
     * @param x      x offest of upper left corner
     * @param y      y offset of upper left corner
     * @param w      width of the desired data array
     * @param h      height of the desired data array
     * @param pixels array of double pixels to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */
    std::vector<double> ReadPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                   std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as integer array. Throws exception when the data is
     * not read from disk yet. If the given array is <code>null</code> a new one was created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels integer array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws NullPointerException     if this band has no raster data
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */
    std::vector<int> GetPixels(int x, int y, int w, int h, std::vector<int> pixels,
                               [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as float array. Throws exception when the data is not
     * read from disk yet. If the given array is <code>null</code> a new one was created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels float array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws NullPointerException     if this band has no raster data
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */

    std::vector<float> GetPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                 [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as double array. Throws exception when the data is not
     * read from disk yet. If the given array is <code>null</code> a new one was created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels double array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws NullPointerException     if this band has no raster data
     * @throws IllegalArgumentException if the length of the given array is less than <code>w*h</code>.
     */
    std::vector<double> GetPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                  [[maybe_unused]] std::shared_ptr<ceres::IProgressMonitor> pm) override;
};
}  // namespace snapengine
}  // namespace alus
