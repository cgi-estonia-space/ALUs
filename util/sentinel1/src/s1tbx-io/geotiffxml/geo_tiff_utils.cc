#include "s1tbx-io/geotiffxml/geo_tiff_utils.h"

#include "custom/gdal_image_reader.h"

namespace alus::s1tbx {
//ImageReader GeoTiffUtils::GetTiffIIOReader(const std::istream& stream) {
//
//    ImageReader reader = nullptr;
//        final Iterator<ImageReader> imageReaders = ImageIO.getImageReaders(stream);
//        while (imageReaders.hasNext()) {
//            final ImageReader iioReader = imageReaders.next();
//            if (iioReader instanceof TIFFImageReader) {
//                reader = iioReader;
//                break;
//            }
//        }
//        if (reader == nullptr){
//            throw std::runtime_error("Unable to find suitable reader for given image stream");
//        }
//        //give stream to some concrete reader implementation
//        reader->SetInput(stream, true, true);
//        return reader;
//
//}

std::shared_ptr<snapengine::custom::IImageReader> GeoTiffUtils::GetTiffIIOReader() {
    std::shared_ptr<snapengine::custom::IImageReader> reader = std::make_shared<snapengine::custom::GdalImageReader>();

    return reader;
}

}  // namespace alus::s1tbx
