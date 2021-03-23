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
#include <string>
#include <string_view>

namespace alus {
namespace snapengine {
class MetadataElement;
class Product;
class IMetaDataReader {
protected:
    std::shared_ptr<Product> product_;
    std::string file_name_;

public:
    IMetaDataReader() = default;
    explicit IMetaDataReader(const std::shared_ptr<Product>& product) : product_(product){};
    explicit IMetaDataReader(const std::string_view file_name) : file_name_(file_name){};
    /**
     * Read from file using implementation
     *
     * @param name of root element to be read
     * @return
     */
    [[nodiscard]] virtual std::shared_ptr<MetadataElement> Read(std::string_view name) = 0;
    // in case of default constructor user needs to provide source product
    virtual void SetProduct(const std::shared_ptr<Product>& product) = 0;

    virtual ~IMetaDataReader() = default;
};
}  // namespace snapengine
}  // namespace alus