/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.dataio.ProductSubsetDef.java
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
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <vector>

// todo:: find a better place for awt replacements (also dimension)
#include "custom/rectangle.h"
#include "custom/dimension.h"
#include "snap-core/subset/abstract_subset_region.h"

namespace alus::snapengine {

/**
 * The <code>ProductSubsetDef</code> class describes a subset or portion of a remote sensing data product.
 * <p> Subsets can be spatial or spectral or both. A spatial subset is given through a rectangular region in pixels or
 * by geometry. The pixel and geometry regions are mutually exclusive. The spectral subset as a list of band (or
 * channel) names.
 *
 * @author Norman Fomferra
 * @author Sabine Embacher
 * modified 20200206 to support geometry subset by Denisa Stefanescu
 * @version $Revision$ $Date$
 */
class ProductSubsetDef {
private:
    /**
     * The subset region
     */
    std::shared_ptr<AbstractSubsetRegion> subset_region_;

    /**
     * The optional name of the subset
     */
    std::string subset_name_;

    /**
     * The spatial subset for each RasterDataNode
     */
    std::unordered_map<std::string, custom::Rectangle> region_map_;

    /**
     * Subsampling in X direction.
     */
    int sub_sampling_x_ = 1;

    /**
     * Subsampling in Y direction.
     */
    int sub_sampling_y_ = 1;

    /**
     * The band subset.
     */
    std::vector<std::string> node_name_list_;

    /**
     * ignores or not ignores Metadata at writing or reading a product
     */
    bool ignore_metadata_ = false;

    bool treat_virtual_bands_as_real_bands_ = false;

    /**
     * Gets the index for the given node name. If the name is not contained in this subset, <code>-1</code> is
     * returned.
     *
     * @param name the node name
     *
     * @return the node index or <code>-1</code>
     */
    int GetNodeNameIndex(std::string_view name);

    /**
     * Constructs a new and empty subset info.
     */
public:
    ProductSubsetDef();

    /**
     * Constructs a new and empty subset info.
     * @param subsetName The name of the subset to be created.
     */
    explicit ProductSubsetDef(std::string_view subset_name);

    std::string GetSubsetName() { return subset_name_; }

    void SetSubsetName(std::string_view subset_name) { subset_name_ = subset_name; }

    void SetTreatVirtualBandsAsRealBands(bool flag) { treat_virtual_bands_as_real_bands_ = flag; }

    bool GetTreatVirtualBandsAsRealBands() const { return treat_virtual_bands_as_real_bands_; }

    /**
     * Gets the names of all product nodes contained in this subset. A return value of <code>null</code> means all nodes
     * are selected.
     *
     * @return an array of names, or <code>null</code> if the no node subset is given
     */
    std::vector<std::string> GetNodeNames();

    /**
     * Sets the names of all product nodes contained in this subset. A value of <code>null</code> means all nodes are
     * selected.
     *
     * @param names the band names, can be <code>null</code> in order to reset the node subset
     */
    void SetNodeNames(const std::vector<std::string>& names);

    /**
     * Adds a new product node name to this subset.
     *
     * @param name the node's name, must not be empty or <code>null</code>
     */
    void AddNodeName(std::string_view name);

    /**
     * Adds the given product node names to this subset.
     *
     * @param names the nodename's to be added
     */
    void AddNodeNames(const std::vector<std::string>& names);

    /**
     * Adds the given product node names to this subset.
     *
     * @param names the nodename's to be added
     */
    void AddNodeNames(const std::unordered_set<std::string>& names);

    /**
     * Removes a band from the spectral subset. If the band is not contained in this subset, the method returns
     * <code>false</code>.
     *
     * @param name the band's name
     *
     * @return <code>true</code> for success, <code>false</code> otherwise
     */
    bool RemoveNodeName(std::string_view name);

    /**
     * Checks whether or not a node name is already contained in this subset.
     *
     * @param name the node name
     *
     * @return true if so
     */
    bool ContainsNodeName(std::string_view name);

    /**
     * Checks whether or not a node (a band, a tie-point grid or metadata element) with the given name will be part of
     * the product subset.
     *
     * @param name the node name
     *
     * @return true if so
     */
    bool IsNodeAccepted(std::string_view name);

    const std::unordered_map<std::string, custom::Rectangle>& GetRegionMap() const { return region_map_; }

    void SetRegionMap(const std::unordered_map<std::string, custom::Rectangle>& region_map);

    /**
     * Gets the sub-sampling in X- and Y-direction (vertical and horizontal).
     *
     * @param subSamplingX sub-sampling in X-direction, must always be greater than zero
     * @param subSamplingY sub-sampling in Y-direction, must always be greater than zero
     */
    void SetSubSampling(int sub_sampling_x, int sub_sampling_y);

    /**
     * Gets the sub-sampling in X-direction (horizontal).
     *
     * @return the sub-sampling in X-direction which is always greater than zero
     */
    int GetSubSamplingX() const { return sub_sampling_x_; }

    /**
     * Gets the sub-sampling in Y-direction (vertical).
     *
     * @return the sub-sampling in Y-direction which is always greater than zero
     */
    int GetSubSamplingY() const { return sub_sampling_y_; }

    //    todo:support only if needed
    /**
     * Gets the required size for a raster required to hold all pixels for the spatial subset for the given
     maximum
     * raster width and height.
     *
     * @param maxWidth  the maximum raster width
     * @param maxHeight the maximum raster height
     *
     * @return the required raster size, never <code>null</code>
     */
    std::shared_ptr<custom::Dimension> GetSceneRasterSize(int max_width, int max_height);

    std::shared_ptr<custom::Dimension> GetSceneRasterSize(int max_width, int max_height, std::string band_name);

    /**
     * Sets the ignore metadata information
     *
     * @param ignoreMetadata if <code>true</code>, metadata may be ignored during write or read a product.
     */
    void SetIgnoreMetadata(bool ignore_metadata);

    /**
     * Gets the ignore metadata information
     */
    bool IsIgnoreMetadata() const { return ignore_metadata_; }

    /**
     * Checks whether or not this subset definition select the entire product.
     */
    bool IsEntireProductSelected();

    std::shared_ptr<AbstractSubsetRegion> GetSubsetRegion() { return subset_region_; }

    void SetSubsetRegion(const std::shared_ptr<AbstractSubsetRegion>& subset_region);

    /**
     * Gets the spatial subset as a rectangular region. Creates a new custom::Rectangle each time it is called.
     * This prevents from modifying this subset by modifying the returned region.
     *
     * TODO TO BE removed in future
     * @deprecated
     * Use {@link #getSubsetRegion()} instead.
     *
     * @return the spatial subset as a rectangular region, or <code>null</code> if no spatial region was defined
     */
    /*[[deprecated]]*/ std::shared_ptr<custom::Rectangle> GetRegion();
};
}  // namespace alus::snapengine
