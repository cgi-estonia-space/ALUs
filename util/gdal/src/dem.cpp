#include "dem.hpp"

#include <iostream>

namespace slap {

Dem::Dem(slap::Dataset ds) : m_ds{std::move(ds)} {
}

void Dem::doWork() {
}

std::vector<double> Dem::getLocalDemFor(Dataset& image, unsigned int x0,
                                        unsigned int y0, unsigned int width,
                                        unsigned int height){
    auto const& dataBuffer = m_ds.getBand1Data();
    auto const bandXsize = m_ds.getBand1Xsize();
    std::vector<double> altitudes;
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

}  // namespace slap
