/**
 * This file is a filtered duplicate of a SNAP's org.esa.snap.core.datamodel.DataNode.java
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

#include "product_node.h"

#include <any>
#include <stdexcept>

#include "product_data.h"

namespace alus {
namespace snapengine {

/**
 * A <code>DataNode</code> is the base class for all nodes within a data product which carry data. The data is
 * represented by an instance of <code>{@link ProductData}</code>.
 */
class DataNode : public ProductNode {
   private:
    /**
     * The data type. Always one of <code>ProductData.TYPE_<i>X</i></code>.
     */
    int data_type_;
    long num_elems_;
    std::shared_ptr<ProductData> data_;
    bool read_only_;
    std::string unit_;
    bool synthetic_;

    //   protected:
    //    explicit ProductNode(const std::string_view name) : name_(name) {}
    //    ProductNode(const std::string_view name, const std::string_view description)
    //        : name_(name), description_(description) {}

    void CheckState() const {
        if (num_elems_ < 0) {
            throw std::runtime_error("number of elements must be at last 1");
        }
    }

   public:
    // these are really only needed for broadcasting which we don't use atm, but might...
    //    static constexpr std::string_view PROPERTY_NAME_DATA{"data"};
    //    static constexpr std::string_view PROPERTY_NAME_READ_ONLY{"readOnly"};
    //    static constexpr std::string_view PROPERTY_NAME_SYNTHETIC{"synthetic"};
    //    static constexpr std::string_view PROPERTY_NAME_UNIT{"unit"};

    /**
     * Constructs a new data node with the given name, data type and number of elements.
     */
    DataNode(std::string_view name, int data_type, long num_elems);
    DataNode(std::string_view name, std::shared_ptr<ProductData> data, bool read_only);

    void SetUnit(std::string_view unit);

    /**
     * Gets the data of this data node.
     */
    [[nodiscard]] auto GetData() const { return data_; }

    [[nodiscard]] auto GetUnit() const { return unit_; }
    /**
     * Gets the data type of this data node.
     *
     * @return the data type which is always one of the multiple <code>ProductData.TYPE_<i>X</i></code> constants
     */
    [[nodiscard]] auto GetDataType() const { return data_type_; }

    [[nodiscard]] auto IsReadOnly() const { return read_only_; }

    /**
     * Gets the data element size in bytes.
     *
     * @see ProductData#getElemSize(int)
     */
    [[nodiscard]] auto GetDataElemSize() const { return ProductData::GetElemSize(GetDataType()); }

    /**
     * Gets the number of data elements in this data node.
     */
    [[nodiscard]] auto GetNumDataElems() const {
        CheckState();
        return num_elems_;
    }

    [[nodiscard]] auto IsSynthetic() const { return synthetic_; }

    void SetSynthetic(bool synthetic);

    /**
     * Creates product data that is compatible to this dataset's data type. The data buffer returned contains exactly
     * <code>numElems</code> elements of a compatible data type.
     *
     * @param numElems the number of elements, must not be less than one
     * @return product data compatible with this data node
     */
    [[nodiscard]] std::shared_ptr<ProductData> CreateCompatibleProductData(int num_elems) const;

    /**
     * Sets the data elements of this data node.
     * @see ProductData#setElems(Object)
     */
    void SetDataElems(const std::any& elems);

    /**
     * Gets the data elements of this data node.
     *
     * @see ProductData#getElems()
     */
    [[nodiscard]] std::any GetDataElems() const { return GetData() == nullptr ? nullptr : GetData()->GetElems(); }
    //    /**
    //     * Sets the data elements of this data node.
    //     * @see ProductData#setElems(Object)
    //     */
    //   void SetDataElems(Object elems) {
    //
    //        if (IsReadOnly()) {
    //            throw std::invalid_argument("attribute is read-only");
    //        }
    //
    //        CheckState();
    //        if (data_ == nullptr) {
    //            if (num_elems_ > Integer.MAX_VALUE) {
    //                throw std::runtime_error("number of elements must be less than "+ (long)Integer.MAX_VALUE + 1);
    //            }
    //            data_ = CreateCompatibleProductData((int) num_elems_);
    //        }
    //        Object oldData = data_.GetElems();
    //        if (!ObjectUtils.EqualObjects(oldData, elems)) {
    //            data_.SetElems(elems);
    ////            fireProductNodeDataChanged();
    ////            setModified(true);
    //        }
    //    }
};
}  // namespace snapengine
}  // namespace alus