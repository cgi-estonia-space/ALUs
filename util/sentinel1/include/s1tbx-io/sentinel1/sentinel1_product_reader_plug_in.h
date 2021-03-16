/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.sentinel1.Sentinel1ProductReaderPlugin.java
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

#include <any>
#include <array>
#include <memory>
#include <string_view>

#include <boost/filesystem.hpp>

#include "snap-core/dataio/decode_qualification.h"
#include "snap-core/dataio/i_product_reader.h"
#include "snap-core/dataio/i_product_reader_plug_in.h"

namespace alus::s1tbx {

/**
 * The ReaderPlugIn for Sentinel1 products modified to helper class
 */
// todo: do shared_from_base?
class Sentinel1ProductReaderPlugIn : public std::enable_shared_from_this<Sentinel1ProductReaderPlugIn>,
                                     public snapengine::IProductReaderPlugIn {
private:
    static constexpr std::string_view ANNOTATION{"annotation"};
    static constexpr std::string_view MEASUREMENT{"measurement"};
    static constexpr std::array<std::string_view, 10> ANNOTATION_PREFIXES{"s1",  "s2-", "s3-", "s4-", "s5-",
                                                                          "s6-", "iw",  "ew",  "rs2", "asa"};

    static bool CheckFolder(const boost::filesystem::path& folder, std::string_view extension);

public:
    static bool IsLevel1(const boost::filesystem::path& path);

    static bool IsLevel2(const boost::filesystem::path& path);

    static bool IsLevel0(const boost::filesystem::path& path);

    static void ValidateInput(const boost::filesystem::path& path);

    Sentinel1ProductReaderPlugIn();
    /**
     * Checks whether the given object is an acceptable input for this product reader and if so, the method checks if it
     * is capable of decoding the input's content.
     *
     * @param input any input object
     * @return true if this product reader can decode the given input, otherwise false.
     */
    snapengine::DecodeQualification GetDecodeQualification(const std::any& input);

    /**
     * Creates an instance of the actual product reader class. This method should never return <code>null</code>.
     *
     * @return a new reader instance, never <code>null</code>
     */
    std::shared_ptr<snapengine::IProductReader> CreateReaderInstance();

    virtual ~Sentinel1ProductReaderPlugIn();
};
}  // namespace alus::s1tbx
