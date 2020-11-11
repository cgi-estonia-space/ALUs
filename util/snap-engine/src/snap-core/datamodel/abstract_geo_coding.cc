#include "snap-core/datamodel/abstract_geo_coding.h"

namespace alus {
namespace snapengine {

MathTransformWKT AbstractGeoCoding::GetImageToMapTransform() {
    if (image_2_map_.empty()) {
        throw std::runtime_error("not yet ported/supported");
        //todo: port only if we use it somewhere
//        synchronized (this) {
//            if (image2Map == null) {
//                try {
//                    image2Map = CRS.findMathTransform(imageCRS, mapCRS);
//                } catch (FactoryException e) {
//                    throw new IllegalArgumentException(
//                        "Not able to find a math transformation from image to map CRS.", e);
//                }
//            }
//        }
    }
    return image_2_map_;
}
}  // namespace snapengine
}  // namespace alus
