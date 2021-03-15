/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.RasterDataNode.java
 * ported for native code.
 * Copied from a snap-engine's(https://github.com/senbox-org/snap-engine) repository originally stated
 * to be implemented by "Copyright (C) 2010 Brockmann Consult GmbH (info@brockmann-consult.de)"
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
#include <string>
#include <string_view>
#include <vector>

//#include "mask.h"
#include "ceres-core/i_progress_monitor.h"
#include "ceres-core/null_progress_monitor.h"
#include "snap-core/datamodel/data_node.h"
#include "snap-core/datamodel/i_geo_coding.h"
#include "snap-core/datamodel/product.h"
#include "snap-core/datamodel/product_data.h"
#include "snap-core/datamodel/scaling.h"

namespace alus {
namespace snapengine {

template <typename T>
class ProductNodeGroup;
// todo: currently placeholder (not sure if we will implemnt)
/**
 * The <code>RasterDataNode</code> class ist the abstract base class for all objects in the product package that contain
 * rasterized data. i.e. <code>Band</code> and <code>TiePointGrid</code>. It unifies the access to raster data in the
 * product model. A raster is considered as a rectangular raw data array with a fixed width and height. A raster data
 * node can scale its raw raster data samples in order to return geophysically meaningful pixel values.
 *
 * original java version author Norman Fomferra
 *
 * @see #GetRasterData()
 * @see #GetRasterWidth()
 * @see #GetRasterHeight()
 * @see #IsScalingApplied()
 * @see #IsLog10Scaled()
 * @see #GetScalingFactor()
 * @see #GetScalingOffset()
 */
class RasterDataNode : public DataNode, public Scaling {
private:
    /**
     * Number of bytes used for internal read buffer.
     */
    static int READ_BUFFER_MAX_SIZE;

    double scaling_factor_;
    double scaling_offset_;
    bool log10_scaled_;
    bool scaling_applied_;
    bool no_data_value_used_;
    std::shared_ptr<ProductData> no_data_;
    double geophysical_no_data_value_;  // invariant, depending on _noData
    std::string valid_pixel_expression_;
    std::shared_ptr<IGeoCoding> geo_coding_;
    // todo::add support when needed
    //    TimeCoding time_coding_;
    //    AffineTransform image_to_model_transform_;
    //    MathTransform2D model_to_scene_transform_;
    //    MathTransform2D scene_to_model_transform_;
    //    Stx stx_;
    //    ImageInfo image_info_;
    //    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> overlay_masks_;
    //    Pointing pointing_;
    //    MultiLevelImage source_image_;
    //    MultiLevelImage geophysical_image_;
    //    MultiLevelImage valid_mask_image_;
    //    ROI valid_mask_r_o_i_;
    //    std::shared_ptr<ProductNodeGroup<std::shared_ptr<RasterDataNode>>> ancillary_variables_;
    //    std::vector<std::string> ancillary_relations_;
    //    AncillaryBandRemover ancillary_band_remover_;

    //    void ResetGeophysicalImage() { geophysical_image_ = nullptr; }

    void SetGeophysicalNoDataValue() { geophysical_no_data_value_ = Scale(GetNoDataValue()); }

protected:
    // todo: remove if tests are still ok
    //    // looks like c++ needs this for init
    //    RasterDataNode() = default;
    /**
     * Constructs an object of type <code>RasterDataNode</code>.
     *
     * @param name     the name of the new object
     * @param dataType the data type used by the raster, must be one of the multiple
     * <code>ProductData.TYPE_<i>X</i></code> constants, with the exception of <code>ProductData.TYPE_UINT32</code>
     * @param numElems the number of elements in this data node.
     */
    RasterDataNode(std::string_view name, int data_type, long num_elems);

public:
    static constexpr std::string_view PROPERTY_NAME_IMAGE_INFO = "imageInfo";
    static constexpr std::string_view PROPERTY_NAME_LOG_10_SCALED = "log10Scaled";
    static constexpr std::string_view PROPERTY_NAME_SCALING_FACTOR = "scalingFactor";
    static constexpr std::string_view PROPERTY_NAME_SCALING_OFFSET = "scalingOffset";
    static constexpr std::string_view PROPERTY_NAME_NO_DATA_VALUE = "noDataValue";
    static constexpr std::string_view PROPERTY_NAME_NO_DATA_VALUE_USED = "noDataValueUsed";
    static constexpr std::string_view PROPERTY_NAME_VALID_PIXEL_EXPRESSION = "validPixelExpression";
    static constexpr std::string_view PROPERTY_NAME_GEO_CODING = "geoCoding";
    static constexpr std::string_view PROPERTY_NAME_TIME_CODING = "timeCoding";
    static constexpr std::string_view PROPERTY_NAME_STX = "stx";
    static constexpr std::string_view PROPERTY_NAME_ANCILLARY_VARIABLES = "ancillaryVariables";
    static constexpr std::string_view PROPERTY_NAME_ANCILLARY_RELATIONS = "ancillaryRelations";
    static constexpr std::string_view PROPERTY_NAME_IMAGE_TO_MODEL_TRANSFORM = "imageToModelTransform";
    static constexpr std::string_view PROPERTY_NAME_MODEL_TO_SCENE_TRANSFORM = "modelToSceneTransform";
    static constexpr std::string_view PROPERTY_NAME_SCENE_TO_MODEL_TRANSFORM = "sceneToModelTransform";

    /**
     * Text returned by the <code>{@link #getPixelString(int, int)}</code> method if no data is available at the given
     * pixel position.
     */
    static constexpr std::string_view NO_DATA_TEXT = "NaN"; /*I18N*/
    /**
     * Text returned by the <code>{@link #getPixelString(int, int)}</code> method if no data is available at the given
     * pixel position.
     */
    static constexpr std::string_view INVALID_POS_TEXT = "Invalid pos."; /*I18N*/
    /**
     * Text returned by the <code>{@link #getPixelString(int, int)}</code> method if an I/O error occurred while pixel
     * data was reloaded.
     */
    static constexpr std::string_view IO_ERROR_TEXT = "I/O error"; /*I18N*/

    virtual int GetRasterWidth() = 0;
    virtual int GetRasterHeight() = 0;

    /**
     * Resets the valid mask of this raster.
     * The mask will be lazily regenerated when requested the next time.
     */
    void ResetValidMask() {
        //        valid_mask_r_o_i_ = nullptr;
        //        valid_mask_image_ = nullptr;
        //        stx_ = nullptr;
        // todo::add support?
    }

    /**
     * @return The native size of the raster in pixels.
     */
    std::shared_ptr<custom::Dimension> GetRasterSize();

    /**
     * Returns <code>true</code> if the pixel data contained in this band is "naturally" a floating point number type.
     *
     * @return true, if so
     */
    bool IsFloatingPointType() override;

    /**
     * Returns the geophysical data type of this <code>RasterDataNode</code>. The value returned is always one of the
     * <code>ProductData.TYPE_XXX</code> constants.
     *
     * @return the geophysical data type
     * @see ProductData
     * @see #isScalingApplied()
     */
    virtual int GetGeophysicalDataType();

    /**
     * Gets the scaling factor which is applied to raw {@link ProductData}. The default value is
     * <code>1.0</code> (no factor).
     *
     * @return the scaling factor
     * @see #isScalingApplied()
     */
    double GetScalingFactor() const { return scaling_factor_; }

    /**
     * Returns the geo-coding of this {@link RasterDataNode}.
     *
     * @return the geo-coding, or {@code null} if not available.
     */
    std::shared_ptr<IGeoCoding> GetGeoCoding();

    /**
     * Sets the geo-coding for this {@link RasterDataNode}.
     * Also sets the geo-coding of the parent {@link Product} if it has no geo-coding yet.
     * <p>On property change, the method calls {@link #fireProductNodeChanged(String)} with the property
     * name {@link #PROPERTY_NAME_GEO_CODING}.
     *
     * @param geoCoding the new geo-coding
     * @see Product#setSceneGeoCoding(GeoCoding)
     */
    void SetGeoCoding(const std::shared_ptr<IGeoCoding>& geo_coding);

    /**
     * Sets the scaling factor which is applied to raw {@link ProductData}.
     *
     * @param scalingFactor the scaling factor
     * @see #isScalingApplied()
     */
    void SetScalingFactor(double scaling_factor);

    void SetScalingApplied() {
        scaling_applied_ = GetScalingFactor() != 1.0 || GetScalingOffset() != 0.0 || IsLog10Scaled();
    }

    /**
     * Sets the scaling offset which is applied to raw {@link ProductData}.
     *
     * @param scalingOffset the scaling offset
     * @see #isScalingApplied()
     */
    void SetScalingOffset(double scaling_offset);

    /**
     * Gets the scaling offset which is applied to raw {@link ProductData}. The default value is
     * <code>0.0</code> (no offset).
     *
     * @return the scaling offset
     * @see #isScalingApplied()
     */
    double GetScalingOffset() const { return scaling_offset_; }

    /**
     * Gets whether or not the {@link ProductData} of this band has a negative binomial distribution and
     * thus the common logarithm (base 10) of the values is stored in the raw data. The default value is
     * <code>false</code>.
     *
     * @return whether or not the data is logging-10 scaled
     * @see #isScalingApplied()
     */
    bool IsLog10Scaled() const { return log10_scaled_; }

    /**
     * Sets the raster data of this dataset.
     * <p> Note that this method does not copy data at all. If the supplied raster data is compatible with this product
     * raster, then simply its reference is stored. Modifications in the supplied raster data will also affect this
     * dataset's data.
     *
     * @param rasterData The raster data for this raster data node.
     * @see #getRasterData()
     */
    virtual void SetRasterData(std::shared_ptr<ProductData> raster_data);

    /**
     * Returns the pixel located at (x,y) as an integer value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x the X co-ordinate of the pixel location
     * @param y the Y co-ordinate of the pixel location
     * @return the pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual int GetPixelInt(int x, int y) = 0;

    /**
     * Returns the pixel located at (x,y) as a float value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x the X co-ordinate of the pixel location
     * @param y the Y co-ordinate of the pixel location
     * @return the pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual float GetPixelFloat(int x, int y) = 0;

    /**
     * Returns the pixel located at (x,y) as a double value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x the X co-ordinate of the pixel location
     * @param y the Y co-ordinate of the pixel location
     * @return the pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual double GetPixelDouble(int x, int y) = 0;

    /**
     * Sets the pixel located at (x,y) to the given integer value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x          the X co-ordinate of the pixel location
     * @param y          the Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual void SetPixelInt(int x, int y, int pixel_value) = 0;

    /**
     * Sets the pixel located at (x,y) to the given float value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x          the X co-ordinate of the pixel location
     * @param y          the Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual void SetPixelFloat(int x, int y, float pixel_value) = 0;

    /**
     * Sets the pixel located at (x,y) to the given double value.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     *
     * @param x          the X co-ordinate of the pixel location
     * @param y          the Y co-ordinate of the pixel location
     * @param pixelValue the new pixel value at (x,y)
     * @throws ArrayIndexOutOfBoundsException if the co-ordinates are not in bounds
     * @throws IllegalStateException          if this object has no internal data buffer
     */
    virtual void SetPixelDouble(int x, int y, double pixel_value) = 0;

    /**
     * Sets a range of pixels specified by the coordinates as integer array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #writePixels(int, int, int, int, double[], ProgressMonitor)} method
     * to write pixels directly to the associated {@link Product#setProductWriter(ProductWriter) product writer}.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels integer array to be written
     * @throws NullPointerException  if this band has no raster data
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual void SetPixels(int x, int y, int w, int h, std::vector<int> pixels) = 0;

    /**
     * Sets a range of pixels specified by the coordinates as float array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #writePixels(int, int, int, int, double[], ProgressMonitor)} method
     * to write pixels directly to the associated {@link Product#setProductWriter(ProductWriter) product writer}.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels float array to be written
     * @throws NullPointerException  if this band has no raster data
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual void SetPixels(int x, int y, int w, int h, std::vector<float> pixels) = 0;

    /**
     * Sets a range of pixels specified by the coordinates as double array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #writePixels(int, int, int, int, double[], ProgressMonitor)} method
     * to write pixels directly to the associated {@link Product#setProductWriter(ProductWriter) product writer}.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written.
     * @param pixels double array to be written
     * @throws NullPointerException  if this band has no raster data
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual void SetPixels(int x, int y, int w, int h, std::vector<double> pixels) = 0;

    /**
     * Releases all of the resources used by this object instance and all of its owned children. Its primary use is to
     * allow the garbage collector to perform a vanilla job.
     * <p>This method should be called only if it is for sure that this object instance will never be used again. The
     * results of referencing an instance of this class after a call to <code>dispose()</code> are undefined.
     * <p>Overrides of this method should always call <code>super.dispose();</code> after disposing this instance.
     */
    void Dispose() override {
        //        if (image_info_ != nullptr) {
        //            image_info_->Dispose();
        //            image_info_ = nullptr;
        //        }
        //        if (source_image_ != nullptr) {
        //            source_image_->Dispose();
        //            source_image_ = nullptr;
        //        }
        //        if (valid_mask_r_o_i_ != nullptr) {
        //            valid_mask_r_o_i_ = nullptr;
        //        }
        //        if (valid_mask_image_ != nullptr) {
        //            valid_mask_image_->Dispose();
        //            valid_mask_image_ = nullptr;
        //        }
        //        if (geophysical_image_ != nullptr && geophysical_image_ != source_image_) {
        //            geophysical_image_->Dispose();
        //            geophysical_image_ = nullptr;
        //        }
        //        overlay_masks_->RemoveAll();
        //        overlay_masks_->ClearRemovedList();
        //
        //        if (ancillary_variables_ != nullptr) {
        //            ancillary_variables_->RemoveAll();
        //            ancillary_variables_->ClearRemovedList();
        //        }

        DataNode::Dispose();
    }

    /**
     * The method behaves exactly as {@link #readRasterData(int, int, int, int, ProductData)},
     * but clients can additionally pass a {@link ProgressMonitor}.
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
     */
    virtual void ReadRasterData(int offset_x, int offset_y, int width, int height,
                                std::shared_ptr<ProductData> raster_data,
                                std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Reads raster data from the node's associated data source into the given data
     * buffer.
     *
     * @param offsetX    the X-offset in the raster co-ordinates where reading starts
     * @param offsetY    the Y-offset in the raster co-ordinates where reading starts
     * @param width      the width of the raster data buffer
     * @param height     the height of the raster data buffer
     * @param rasterData a raster data buffer receiving the pixels to be read
     * @throws java.io.IOException      if an I/O error occurs
     * @throws IllegalArgumentException if the raster is null
     * @throws IllegalStateException    if this product raster was not added to a product so far, or if the product to
     *                                  which this product raster belongs to, has no associated product reader
     * @see ProductReader#readBandRasterData(Band, int, int, int, int, ProductData, com.bc.ceres.core.ProgressMonitor)
     */
    void ReadRasterData(int offset_x, int offset_y, int width, int height, std::shared_ptr<ProductData> raster_data);

    /**
     * @throws java.io.IOException if an I/O error occurs
     * @see #readRasterDataFully(ProgressMonitor)
     * @see #unloadRasterData()
     */
    void ReadRasterDataFully();

    /**
     * Reads the complete underlying raster data.
     * <p>After this method has been called successfully, <code>hasRasterData()</code> should always return
     * <code>true</code> and <code>getRasterData()</code> should always return a valid <code>ProductData</code> instance
     * with at least <code>getRasterWidth()*getRasterHeight()</code> elements (samples).
     * <p>In opposite to the <code>loadRasterData</code> method, the <code>readRasterDataFully</code> method always
     * reloads the data of this product raster, independently of whether its has already been loaded or not.
     *
     * @param pm a monitor to inform the user about progress
     * @throws java.io.IOException if an I/O error occurs
     * @see #loadRasterData
     * @see #readRasterData(int, int, int, int, ProductData, com.bc.ceres.core.ProgressMonitor)
     */
    virtual void ReadRasterDataFully(std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Writes data from this product raster into the specified region of the user-supplied raster.
     * <p> It is important to know that this method does not change this product raster's internal state nor does it
     * write into this product raster's internal raster.
     *
     * @param rasterData a raster data buffer receiving the pixels to be read
     * @param offsetX    the X-offset in raster co-ordinates where reading starts
     * @param offsetY    the Y-offset in raster co-ordinates where reading starts
     * @param width      the width of the raster data buffer
     * @param height     the height of the raster data buffer
     * @param pm         a monitor to inform the user about progress
     * @throws java.io.IOException      if an I/O error occurs
     * @throws IllegalArgumentException if the raster is null
     * @throws IllegalStateException    if this product raster was not added to a product so far, or if the product to
     *                                  which this product raster belongs to, has no associated product reader
     * @see ProductReader#readBandRasterData(Band, int, int, int, int, ProductData, com.bc.ceres.core.ProgressMonitor)
     */
    virtual void WriteRasterData(int offset_x, int offset_y, int width, int height,
                                 std::shared_ptr<ProductData> raster_data,
                                 std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Writes the complete underlying raster data.
     *
     * @param pm a monitor to inform the user about progress
     * @throws java.io.IOException if an I/O error occurs
     */
    virtual void WriteRasterDataFully(std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * @see #readPixels(int, int, int, int, int[], ProgressMonitor)
     */
    std::vector<int> ReadPixels(int x, int y, int w, int h, std::vector<int> pixels) {
        return ReadPixels(x, y, w, h, pixels, std::make_shared<ceres::NullProgressMonitor>());
    }

    /**
     * Retrieves the band data at the given offset (x, y), width and height as int data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader or
     * operator to read or compute the data. <p> If the {@code pixels} array is <code>null</code> a new one will be
     * created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read
     * @param pixels array to be filled with data
     * @param pm     a progress monitor
     * @return the pixels read
     * @throws IOException           if an /IO error occurs
     * @throws IllegalStateException if this object has no attached {@link Product#setProductReader(ProductReader)
     * product reader}
     */
    virtual std::vector<int> ReadPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                        std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * @see #readPixels(int, int, int, int, float[], ProgressMonitor)
     */
    std::vector<float> ReadPixels(int x, int y, int w, int h, std::vector<float> pixels) {
        return ReadPixels(x, y, w, h, pixels, std::make_shared<ceres::NullProgressMonitor>());
    }

    /**
     * Retrieves the band data at the given offset (x, y), width and height as int data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader or
     * operator to read or compute the data. <p> If the {@code pixels} array is <code>null</code> a new one will be
     * created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read
     * @param pixels array to be filled with data
     * @param pm     a progress monitor
     * @return the pixels read
     * @throws IllegalStateException if this object has no attached {@link Product#setProductReader(ProductReader)
     * product reader}
     */
    virtual std::vector<float> ReadPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                          std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * @see #readPixels(int, int, int, int, double[], ProgressMonitor)
     */
    std::vector<double> ReadPixels(int x, int y, int w, int h, std::vector<double> pixels) {
        return ReadPixels(x, y, w, h, pixels, std::make_shared<ceres::NullProgressMonitor>());
    }

    /**
     * Retrieves the band data at the given offset (x, y), width and height as int data. If the data is already in
     * memory, it merely copies the data to the buffer provided. If not, it calls the attached product reader or
     * operator to read or compute the data. <p> If the {@code pixels} array is <code>null</code> a new one will be
     * created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read
     * @param pixels array to be filled with data
     * @param pm     a progress monitor
     * @return the pixels read
     * @throws IllegalStateException if this object has no attached {@link Product#setProductReader(ProductReader)
     * product reader}
     */
    virtual std::vector<double> ReadPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                           std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    //    /**
    //     * @see #writePixels(int, int, int, int, int[], ProgressMonitor)
    //     */
    //    @SuppressWarnings("unused") // may be useful API for scripting languages
    //   public void writePixels(int x, int y, int w, int h, int[] pixels) throws IOException {
    //        writePixels(x, y, w, h, pixels, ProgressMonitor.NULL);
    //    }

    /**
     * Writes the range of given pixels specified to the specified coordinates as integers.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written
     * @param pixels array of pixels to write
     * @param pm     a progress monitor
     * @throws IllegalStateException if this object has no attached {@link Product#setProductWriter(ProductWriter)
     * product writer}
     * @throws IOException           if an I/O error occurs
     */
    virtual void WritePixels(int x, int y, int w, int h, std::vector<int> pixels,
                             std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    //    /**
    //     * @see #writePixels(int, int, int, int, float[], ProgressMonitor)
    //     */
    //    @SuppressWarnings("unused") // may be useful API for scripting languages
    //   public synchronized void writePixels(int x, int y, int w, int h, float[] pixels) throws IOException {
    //        writePixels(x, y, w, h, pixels, ProgressMonitor.NULL);
    //    }

    /**
     * Writes the range of given pixels specified to the specified coordinates as floats.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written
     * @param pixels array of pixels to write
     * @param pm     a progress monitor
     * @throws IllegalStateException if this object has no attached {@link Product#setProductWriter(ProductWriter)
     * product writer}
     * @throws IOException           if an I/O error occurs
     */
    virtual void WritePixels(int x, int y, int w, int h, std::vector<float> pixels,
                             std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    //    /**
    //     * @see #writePixels(int, int, int, int, double[], ProgressMonitor)
    //     */
    //    @SuppressWarnings("unused") // may be useful API for scripting languages
    //   public void writePixels(int x, int y, int w, int h, double[] pixels) throws IOException {
    //        writePixels(x, y, w, h, pixels, ProgressMonitor.NULL);
    //    }

    /**
     * Writes the range of given pixels specified to the specified coordinates as doubles.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be written
     * @param h      height of the pixel array to be written
     * @param pixels array of pixels to write
     * @param pm     a progress monitor
     * @throws IllegalStateException if this object has no attached {@link Product#setProductWriter(ProductWriter)
     * product writer}
     * @throws IOException           if an I/O error occurs
     */
    virtual void WritePixels(int x, int y, int w, int h, std::vector<double> pixels,
                             std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Returns true if the raster data of this <code>RasterDataNode</code> is loaded or elsewhere available, otherwise
     * false.
     *
     * @return true, if so.
     */
    bool HasRasterData() { return GetRasterData() != nullptr; }

    /**
     * Gets the raster data for this dataset. If the data hasn't been loaded so far the method returns
     * <code>null</code>.
     *
     * @return the raster data for this band, or <code>null</code> if data has not been loaded
     */
    virtual std::shared_ptr<ProductData> GetRasterData() { return GetData(); }

    /**
     * Creates raster data that is compatible to this dataset's data type. The data buffer returned contains exactly
     * <code>width*height</code> elements of a compatible data type.
     *
     * @param width  the width of the raster data to be created
     * @param height the height of the raster data to be created
     * @return raster data compatible with this product raster
     * @see #createCompatibleRasterData
     * @see #createCompatibleSceneRasterData
     */
    std::shared_ptr<ProductData> CreateCompatibleRasterData(int width, int height) {
        return CreateCompatibleProductData(width * height);
    }

    /**
     * Tests whether or not a no-data value has been specified. The no-data value is not-specified unless either
     * {@link #setNoDataValue(double)} or {@link #setGeophysicalNoDataValue(double)} is called.
     *
     * @return true, if so
     * @see #isNoDataValueUsed()
     * @see #setNoDataValue(double)
     */
    bool IsNoDataValueSet() { return no_data_ != nullptr; }

    /**
     * Gets the no-data value as a primitive <code>double</code>.
     * <p>Note that the value returned is NOT necessarily the same as the value returned by
     * {@link #getGeophysicalNoDataValue()} because no scaling is applied.
     * <p>The no-data value is used to determine valid pixels. For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     * <p>The method returns <code>0.0</code>, if no no-data value has been specified so far.
     *
     * @return the no-data value. It is returned as a <code>double</code> in order to cover all other numeric types.
     * @see #setNoDataValue(double)
     * @see #isNoDataValueSet()
     * @see #isNoDataValueUsed()
     */
    double GetNoDataValue() { return IsNoDataValueSet() ? no_data_->GetElemDouble() : 0.0; }

    /**
     * Applies the scaling <code>v * scalingFactor + scalingOffset</code> the the given input value. If the
     * <code>log10Scaled</code> property is true, the result is taken to the power of 10 <i>after</i> the actual
     * scaling.
     *
     * @param v the input value
     * @return the scaled value
     */
    double Scale(double v) override;

    /**
     * Sets whether or not the {@link ProductData} of this band has a negative binomial distribution and
     * thus the common logarithm (base 10) of the values is stored in the raw data.
     *
     * @param log10Scaled whether or not the data is logging-10 scaled
     * @see #isScalingApplied()
     */
    void SetLog10Scaled(bool log10_scaled);

    /**
     * Tests whether scaling of raw raster data values is applied before they are returned as geophysically meaningful
     * pixel values. <p>The methods which return geophysical pixel values are all {@link #getPixels(int, int, int, int,
     * int[])},
     * {@link #setPixels(int, int, int, int, int[])}, {@link #readPixels(int, int, int, int, int[])} and
     * {@link #writePixels(int, int, int, int, int[])} methods as well as the <code>getPixel&lt;Type&gt;</code> and
     * <code>setPixel&lt;Type&gt;</code> methods such as  {@link #getPixelFloat(int, int)} * and
     * {@link #setPixelFloat(int, int, float)}.
     *
     * @return <code>true</code> if a conversion is applyied to raw data samples before the are retuned.
     * @see #getScalingOffset
     * @see #getScalingFactor
     * @see #isLog10Scaled
     */
    bool IsScalingApplied() const { return scaling_applied_; }

    /**
     * Sets whether or not the no-data value is used.
     * If the no-data value is enabled and the no-data value has not been set so far,
     * a default no-data value it is set with a value of to zero.
     * <p>The no-data value is used to determine valid pixels. For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     * <p>On property change, the method calls {@link #fireProductNodeChanged(String)} with the property
     * name {@link #PROPERTY_NAME_NO_DATA_VALUE_USED}.
     *
     * @param noDataValueUsed true, if so
     * @see #isNoDataValueUsed()
     */
    void SetNoDataValueUsed(bool no_data_value_used);

    /**
     * Sets the no-data value as a primitive <code>double</code>.
     * <p>Note that the given value is related to the "raw", un-scaled raster data.
     * In order to set the geophysical, scaled no-data value use the method
     * {@link #setGeophysicalNoDataValue(double)}.
     * <p>The no-data value is used to determine valid pixels. For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     * <p>On property change, the method calls {@link #fireProductNodeChanged(String)} with the property
     * name {@link #PROPERTY_NAME_NO_DATA_VALUE}.
     *
     * @param noDataValue the no-data value. It is passed as a <code>double</code> in order to cover all other numeric
     * types.
     * @see #getNoDataValue()
     * @see #isNoDataValueSet()
     */
    void SetNoDataValue(double no_data_value);

    /**
     * Tests whether or not the no-data value is used.
     * <p>The no-data value is used to determine valid pixels. For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     *
     * @return true, if so
     * @see #setNoDataValueUsed(boolean)
     * @see #isNoDataValueSet()
     */
    bool IsNoDataValueUsed() const { return no_data_value_used_; }

    /**
     * Gets the expression that is used to determine whether a pixel is valid or not.
     * For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     *
     * @return the valid mask expression.
     */
    std::string GetValidPixelExpression() { return valid_pixel_expression_; }
    /**
     * Sets the expression that is used to determine whether a pixel is valid or not.
     * <p>The valid-pixel expression is used to determine valid pixels. For more information
     * on valid pixels, please refer to the documentation of the {@link #isPixelValid(int, int, javax.media.jai.ROI)}
     * method.
     * <p>On property change, the method calls {@link #fireProductNodeChanged(String)} with the property
     * name {@link #PROPERTY_NAME_VALID_PIXEL_EXPRESSION}.
     *
     * @param validPixelExpression the valid mask expression, can be null
     */
    void SetValidPixelExpression(std::string_view valid_pixel_expression);

    /**
     * Determines whether this raster data node contains integer samples.
     *
     * @return true if this raster data node contains integer samples.
     */
    bool HasIntPixels() { return ProductData::IsIntType(GetDataType()); }

    /**
     * @return The overlay mask group.
     */
    //    std::shared_ptr<ProductNodeGroup<std::shared_ptr<Mask>>> GetOverlayMaskGroup() { return overlay_masks_; }

    /**
     * Applies the inverse scaling <code>(v - scalingOffset) / scalingFactor</code> the the given input value. If the
     * <code>log10Scaled</code> property is true, the common logarithm is applied to the input <i>before</i> the actual
     * scaling.
     *
     * @param v the input value
     * @return the scaled value
     */

    double ScaleInverse(double v) override;

    /**
     * Retrieves the range of pixels specified by the coordinates as integer array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #readPixels(int, int, int, int, double[], ProgressMonitor)} method
     * to read or compute pixel values without a raster data buffer.
     * <p>
     * If the {@code pixels} array is <code>null</code> a new one will be created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels integer array to be filled with data
     * @param pm     a progress monitor
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual std::vector<int> GetPixels(int x, int y, int w, int h, std::vector<int> pixels,
                                       std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Retrieves the range of pixels specified by the coordinates as float array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #readPixels(int, int, int, int, double[], ProgressMonitor)} method
     * to read or compute pixel values without a raster data buffer.
     * <p>
     * If the {@code pixels} array is <code>null</code> a new one will be created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels float array to be filled with data
     * @param pm     a progress monitor
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual std::vector<float> GetPixels(int x, int y, int w, int h, std::vector<float> pixels,
                                         std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    /**
     * Retrieves the range of pixels specified by the coordinates as double array.
     * <p>
     * Note that this method can only be used if this object's internal raster data buffer has been
     * {@link #setRasterData(ProductData) set} or {@link #loadRasterData() loaded}.
     * You can use the {@link #readPixels(int, int, int, int, double[], ProgressMonitor)} method
     * to read or compute pixel values without a raster data buffer.
     * <p>
     * If the {@code pixels} array is <code>null</code> a new one will be created and returned.
     *
     * @param x      x offset into the band
     * @param y      y offset into the band
     * @param w      width of the pixel array to be read
     * @param h      height of the pixel array to be read.
     * @param pixels double array to be filled with data
     * @param pm     a monitor to inform the user about progress
     * @throws IllegalStateException if this object has no internal data buffer
     */
    virtual std::vector<double> GetPixels(int x, int y, int w, int h, std::vector<double> pixels,
                                          std::shared_ptr<ceres::IProgressMonitor> pm) = 0;

    virtual void SetModified(bool modified) override;
};
}  // namespace snapengine
}  // namespace alus
