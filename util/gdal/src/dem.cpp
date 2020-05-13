#include "dem.hpp"

#include <iostream>

namespace alus {

Dem::Dem(alus::Dataset ds) : m_ds{std::move(ds)} {
    //m_ds.loadRasterBand(1);
}

void Dem::doWork() {
}

std::vector<double> Dem::getLocalDemFor(Dataset& image, unsigned int x0,
                                        unsigned int y0, unsigned int width,
                                        unsigned int height){
    auto const& dataBuffer = m_ds.getDataBuffer();
    auto const bandXsize = m_ds.getXSize();
    std::vector<double> altitudes;
    altitudes.reserve(dataBuffer.size());
    for (unsigned int iX = 0; iX < width; iX++) {
        for (unsigned int iY = 0; iY < height; iY++) {
            auto const coordinates =
                image.getPixelCoordinatesFromIndex(x0 + iX, y0 + iY);
            auto const demIndexes = m_ds.getPixelIndexFromCoordinates(
                std::get<0>(coordinates), std::get<1>(coordinates));
            double value = dataBuffer.at(bandXsize * std::get<1>(demIndexes) +
                                         std::get<0>(demIndexes));

            if (value < 0) value = 0;

            altitudes.push_back(value);
        }
    }

    return altitudes;
}

}  // namespace alus
