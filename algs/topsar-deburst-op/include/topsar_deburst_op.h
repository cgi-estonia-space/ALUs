/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.sentinel1.gpf.TOPSARDeburstOp.java
 * ported for native code.
 * Copied from (https://github.com/senbox-org/s1tbx). It was originally stated:
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
#include <optional>
#include <stdexcept>
#include <string>
#include <string_view>
#include <unordered_map>
#include <utility>
#include <vector>

#include "ceres-core/i_progress_monitor.h"
#include "custom/rectangle.h"
#include "i_meta_data_writer.h"  //todo move under custom, this is not ported
#include "sentinel1_utils.h"
#include "snap-core/datamodel/band.h"
#include "snap-core/datamodel/i_geo_coding.h"
#include "snap-core/datamodel/metadata_element.h"
#include "snap-core/datamodel/product.h"
#include "snap-gpf/i_tile.h"
#include "subswath_info.h"

namespace alus::s1tbx {

struct BurstInfo {
    int sy0 = -1;
    int sy1 = -1;
    int swath0;
    int swath1;
    int burst_num0 = 0;
    int burst_num1 = 0;
    double target_time;
    double mid_time;
};

struct SubSwathEffectStartEndPixels {
    int x_min;
    int x_max;
};

// todo: make operator interface!?
/**
 * De-Burst a Sentinel-1 TOPSAR product
 */
class TOPSARDeburstOp /*: virtual public IOperator*/ {
private:
    std::shared_ptr<snapengine::Product> source_product_;
    std::shared_ptr<snapengine::Product> target_product_;
    std::vector<std::string> selected_polarisations_;

    std::shared_ptr<snapengine::MetadataElement> abs_root_;
    std::vector<snapengine::custom::Rectangle> target_rectangles_;
    std::string acquisition_mode_{};
    std::string product_type_{};
    int num_of_sub_swath_ = 0;
    //    int sub_swath_index_ = 0; //also not used in current snap version
    int target_width_ = 0;
    int target_height_ = 0;

    double target_first_line_time_ = 0;
    double target_last_line_time_ = 0;
    double target_line_time_interval_ = 0;
    double target_slant_range_time_to_first_pixel_ = 0;
    double target_slant_range_time_to_last_pixel_ = 0;
    double target_delta_slant_range_time_ = 0;

    std::vector<SubSwathEffectStartEndPixels> sub_swath_effect_start_end_pixels_;

    // java original has member, we use it from su_
    std::unique_ptr<s1tbx::Sentinel1Utils> su_;

    static constexpr int NUM_OF_BOUNDARY_POINTS = 6;
    static constexpr std::string_view PRODUCT_SUFFIX{"_Deb"};

    /**
     * Get product type from abstracted metadata.
     */
    void GetProductType();

    /**
     * Get acquisition mode from abstracted metadata.
     */
    void GetAcquisitionMode();

    /**
     * Compute azimuth time for the first and last line in the target product.
     */
    void ComputeTargetStartEndTime();

    /**
     * Compute slant range time to the first and last pixels in the target product.
     */
    void ComputeTargetSlantRangeTimeToFirstAndLastPixels();

    /**
     * Compute target product dimension.
     */
    void ComputeTargetWidthAndHeight();

    void ComputeSubSwathEffectStartEndPixels();

    /**
     * Create target product.
     */
    void CreateTargetProduct();

    [[nodiscard]] std::string GetTargetBandNameFromSourceBandName(std::string_view src_band_name) const;

    std::optional<std::string> GetSourceBandNameFromTargetBandName(std::string_view tgt_band_name,
                                                                   std::string_view acquisition_mode,
                                                                   std::string_view swath_index_str);

    static std::string GetPrefix(std::string_view tgt_band_name);

    [[nodiscard]] bool ContainSelectedPolarisations(std::string_view band_name) const;

    /**
     * Create target product tie point grid.
     */
    void CreateTiePointGrids();

    /**
     * Update target product metadata.
     */
    void UpdateTargetProductMetadata();

    void UpdateAbstractMetadata();

    void AddBurstBoundary(const std::shared_ptr<snapengine::MetadataElement>& abs_tgt) const;

    std::shared_ptr<snapengine::MetadataElement> CreatePointElement(
        double line_time, double pixel_time, const std::shared_ptr<snapengine::IGeoCoding>& target_geo_coding) const;

    void UpdateOriginalMetadata();

    void UpdateSwathTiming();

    static std::string GetMissionPrefix(const std::shared_ptr<snapengine::MetadataElement>& abs_root);

    void UpdateCalibrationVector();

    std::string GetMergedPixels(std::string_view pol);

    std::string GetMergedVector(std::string_view vector_name, std::string_view pol, int vector_index);

    //    todo: add support if actually used, this introduces more complexity (for v1 not needed)
    [[maybe_unused]] void ComputeTileInOneSwathShort(
        [[maybe_unused]] int tx0, [[maybe_unused]] int ty0, [[maybe_unused]] int tx_max, [[maybe_unused]] int ty_max,
        [[maybe_unused]] int first_sub_swath_index,
        [[maybe_unused]] const std::vector<std::shared_ptr<snapengine::custom::Rectangle>>& source_rectangle,
        [[maybe_unused]] std::string_view tgt_band_name,
        [[maybe_unused]] const std::shared_ptr<snapengine::ITile>& tgt_tile, [[maybe_unused]] BurstInfo& burst_info) {
        //        todo: port if used
        throw std::runtime_error("Current support is for single swath float");
    }

    void ComputeTileInOneSwathFloat(int tx0, int ty0, int tx_max, int ty_max, int first_sub_swath_index,
                                    const std::vector<snapengine::custom::Rectangle>& source_rectangle,
                                    std::string_view tgt_band_name, const std::shared_ptr<snapengine::ITile>& tgt_tile,
                                    BurstInfo& burst_info);

    [[maybe_unused]] void ComputeMultipleSubSwathsShort(
        [[maybe_unused]] int tx0, [[maybe_unused]] int ty0, [[maybe_unused]] int tx_max, [[maybe_unused]] int ty_max,
        [[maybe_unused]] int first_sub_swath_index, [[maybe_unused]] int last_sub_swath_index,
        [[maybe_unused]] const std::vector<snapengine::custom::Rectangle>& source_rectangle,
        [[maybe_unused]] std::string_view tgt_band_name,
        [[maybe_unused]] const std::shared_ptr<snapengine::ITile>& tgt_tile,
        [[maybe_unused]] const BurstInfo& burst_info) {
        //        todo: port if used
        throw std::runtime_error("Current support is for single swath float");
    }

    [[maybe_unused]] void ComputeMultipleSubSwathsFloat(
        [[maybe_unused]] int tx0, [[maybe_unused]] int ty0, [[maybe_unused]] int tx_max, [[maybe_unused]] int ty_max,
        [[maybe_unused]] int first_sub_swath_index, [[maybe_unused]] int last_sub_swath_index,
        [[maybe_unused]] const std::vector<snapengine::custom::Rectangle>& source_rectangle,
        [[maybe_unused]] std::string_view tgt_band_name,
        [[maybe_unused]] const std::shared_ptr<snapengine::ITile>& tgt_tile,
        [[maybe_unused]] const BurstInfo& burst_info) {
        //        todo: port if used
        throw std::runtime_error("Current support is for single swath float");
    }

    /**
     * Get source tile rectangle.
     *
     * @param tx0           X coordinate for the upper left corner pixel in the target tile.
     * @param ty0           Y coordinate for the upper left corner pixel in the target tile.
     * @param tw            The target tile width.
     * @param th            The target tile height.
     * @param sub_swath_index The subswath index.
     * @return The source tile rectangle.
     */
    snapengine::custom::Rectangle GetSourceRectangle(int tx0, int ty0, int tw, int th, int sub_swath_index) const;

    [[nodiscard]] int GetSampleIndexInSourceProduct(int tx, const SubSwathInfo& sub_swath) const;

    bool GetLineIndicesInSourceProduct(int ty, const SubSwathInfo& sub_swath, BurstInfo& burst_times) const;

    [[nodiscard]] int ComputeYMin(const SubSwathInfo& sub_swath) const;

    [[nodiscard]] int ComputeYMax(const SubSwathInfo& sub_swath) const;

    [[nodiscard]] int ComputeXMin(const SubSwathInfo& sub_swath) const;

    [[nodiscard]] int ComputeXMax(const SubSwathInfo& sub_swath) const;

    int GetSubSwathIndex(int tx, int ty, int first_sub_swath_index, int last_sub_swath_index,
                         BurstInfo& burst_info) const;

    // todo: port if use
    [[maybe_unused]] double GetSubSwathNoise([[maybe_unused]] int tx, [[maybe_unused]] double target_line_time,
                                             [[maybe_unused]] const std::unique_ptr<SubSwathInfo>& sw,
                                             [[maybe_unused]] std::string_view pol);

    // currently operator init should be done using CreateTOPSARDeburstOp to ensure init call
    explicit TOPSARDeburstOp(std::shared_ptr<snapengine::Product> source_product)
        : source_product_(std::move(source_product)){};

public:
    TOPSARDeburstOp(const TOPSARDeburstOp&) = delete;
    TOPSARDeburstOp& operator=(const TOPSARDeburstOp&) = delete;
    /**
     * To ensure Initialize() is called (currently we have no Operator interface)
     * @param product
     * @return
     */
    static std::shared_ptr<TOPSARDeburstOp> CreateTOPSARDeburstOp(const std::shared_ptr<snapengine::Product>& product);

    /**
     * Initializes this operator and sets the one and only target product.
     * <p>The target product can be either defined by a field of type {@link Product} annotated with the
     * {@link TargetProduct TargetProduct} annotation or
     * by calling {@link #setTargetProduct} method.</p>
     * <p>The framework calls this method after it has created this operator.
     * Any client code that must be performed before computation of tile data
     * should be placed here.</p>
     *
     * @throws OperatorException If an error occurs during operator initialisation.
     * @see #getTargetProduct()
     */
    // currently we don't have similar Operator concept like java version, but we might add it
    void Initialize();

    /**
     * Called by the framework in order to compute the stack of tiles for the given target bands.
     * <p>The default implementation throws a runtime exception with the message "not implemented".</p>
     *
     * @param targetTiles The current tiles to be computed for each target band.
     * @param pm          A progress monitor which should be used to determine computation cancelation requests.
     * @throws OperatorException if an error occurs during computation of the target rasters.
     */
    // currently we don't have operator interface like java version
    void ComputeTileStack(
        std::unordered_map<std::shared_ptr<snapengine::Band>, std::shared_ptr<snapengine::ITile>>& target_tiles,
        const snapengine::custom::Rectangle& target_rectangle, const std::shared_ptr<ceres::IProgressMonitor>& pm);

    // this comes from Operator in original Snap implementation
    /**
     * Gets a {@link Tile} for a given band and image region.
     *
     * @param rasterDataNode the raster data node of a data product,
     *                       e.g. a {@link Band Band} or
     *                       {@link TiePointGrid TiePointGrid}.
     * @param region         the image region in pixel coordinates
     * @return a tile.
     * @throws OperatorException if the tile request cannot be processed
     */
    std::shared_ptr<snapengine::ITile> GetSourceTile(
        const std::shared_ptr<snapengine::RasterDataNode>& raster_data_node,
        const snapengine::custom::Rectangle& region, int src_band_indx);

    /**
     * !Temporary! custom function to write operator outputs (exists because of our temporary custom format)
     * @param metadata_writer
     */
    void WriteProductFiles(const std::shared_ptr<snapengine::IMetaDataWriter>& metadata_writer);

    /**
     * Calculates target image
     * @return
     */
    void Compute();

    /**
     * Gets the target product for the operator
     *
     */
    const std::shared_ptr<snapengine::Product>& GetTargetProduct() const;
};

}  // namespace alus::s1tbx
