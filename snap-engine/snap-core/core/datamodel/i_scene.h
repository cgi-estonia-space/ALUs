/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.snap.core.datamodel.Scene.java
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

namespace alus {
namespace snapengine {

class Product;
class ProductSubsetDef;
class IGeoCoding;
/**
 * Represents a geo-coded scene. This interface is not ment to be implemented by clients.
 */
class IScene {
public:
    virtual void SetGeoCoding(const std::shared_ptr<IGeoCoding>& geo_coding) = 0;

    virtual std::shared_ptr<IGeoCoding> GetGeoCoding() = 0;

    virtual bool TransferGeoCodingTo(const std::shared_ptr<IScene>& dest_scene,
                                     const std::shared_ptr<ProductSubsetDef>& subset_def) = 0;

    virtual int GetRasterWidth() = 0;

    virtual int GetRasterHeight() = 0;

    virtual std::shared_ptr<Product> GetProduct() = 0;

    IScene(const IScene&) = delete;
    IScene& operator=(const IScene&) = delete;
    virtual ~IScene() = default;
};

}  // namespace snapengine
}  // namespace alus
