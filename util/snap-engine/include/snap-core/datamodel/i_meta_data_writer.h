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
#pragma once

#include <memory>

namespace alus {
namespace snapengine {
class Product;

/**
 * Temporary abstract class which provides API for implementations of metadata writers from internal root element.
 */
class IMetaDataWriter {
protected:
    std::shared_ptr<Product> product_;

public:
    IMetaDataWriter() = default;
    IMetaDataWriter(const IMetaDataWriter&) = delete;
    IMetaDataWriter& operator=(const IMetaDataWriter&) = delete;
    virtual ~IMetaDataWriter() = default;

    IMetaDataWriter(const std::shared_ptr<Product>& product) : product_(product){};
    // write to file
    virtual void Write() = 0;
    // in case of default constructor user needs to provide target product
    virtual void SetProduct(const std::shared_ptr<Product>& product) = 0;
};
}  // namespace snapengine
}  // namespace alus