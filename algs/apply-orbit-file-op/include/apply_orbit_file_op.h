/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.sar.gpf.orbits.ApplyOrbitFileOp.java
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
#include <string>
#include <string_view>

#include "i_meta_data_writer.h"
#include "io/orbits/sentinel1/i_orbit_file.h"
#include "metadata_element.h"
#include "product.h"

namespace alus {
namespace s1tbx {

// todo: make operator interface!?
class ApplyOrbitFileOp /*: virtual public IOperator*/ {
private:
    static constexpr std::string_view PRODUCT_SUFFIX = "_Orb";
    std::shared_ptr<snapengine::Product> source_product_;
    std::shared_ptr<snapengine::Product> target_product_;
    int source_image_width_;
    int source_image_height_;
    std::string orbit_type_;
    std::string mission_;
    int poly_degree_{3};
    bool continue_on_fail_{false};
    bool product_updated_{false};
    std::shared_ptr<IOrbitFile> orbit_provider_;
    std::shared_ptr<snapengine::MetadataElement> abs_root_;

    // extension to snap, apply the orbit file to source only
    bool modify_source_only_{false};

    /**
     * Update orbit state vectors using data from the orbit file.
     *
     * @throws Exception The exceptions.
     */
    void UpdateOrbitStateVectors();
    /**
     * Get source metadata
     */
    void GetSourceMetadata();
    /**
     * Create target product.
     */
    void CreateTargetProduct();

    void UpdateOrbits();

    ApplyOrbitFileOp(const ApplyOrbitFileOp&) = delete;
    ApplyOrbitFileOp& operator=(const ApplyOrbitFileOp&) = delete;

public:
    explicit ApplyOrbitFileOp(const std::shared_ptr<snapengine::Product>& source_product);

    ApplyOrbitFileOp(const std::shared_ptr<snapengine::Product>& source_product, bool modify_source_only);
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
    //    @Override  //looks like currently we don't have similar Operator concept, but we might add it
    void Initialize();
    /**
     * write target product files using respective writers
     */
    void WriteProductFiles(std::shared_ptr<snapengine::IMetaDataWriter> metadata_writer);

    // added to get output (snap has this attached to parent class Operator
    std::shared_ptr<snapengine::Product> GetTargetProduct() { return target_product_; };
};

}  // namespace s1tbx
}  // namespace alus
