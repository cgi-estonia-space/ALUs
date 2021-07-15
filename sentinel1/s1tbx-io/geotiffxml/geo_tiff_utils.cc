/**
 * This file is a filtered duplicate of a SNAP's
 * org.esa.s1tbx.io.geotiffxml.GeoTiffUtils.java
 * ported for native code.
 * Copied from(https://github.com/senbox-org/s1tbx). It was originally stated:
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
