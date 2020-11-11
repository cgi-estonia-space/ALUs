#pragma once

#include <memory>

#include "custom/i_image_reader.h"

namespace alus::s1tbx {

class GeoTiffUtils {
public:
    //    todo: looks like inputstream will be given to concrete implementation of image reader which can decode given
    //    inputstream static IImageReader GetTiffIIOReader(const std::istream& stream);
    // TEMPORARY PLACEHOLDER SOLUTION TO PROVIDE GDAL (or some future alternative)
    static std::shared_ptr<snapengine::custom::IImageReader> GetTiffIIOReader();
};

}  // namespace alus::s1tbx
