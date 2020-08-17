#include "dem.hpp"

#include <iostream>

namespace alus {

Dem::Dem(alus::Dataset ds) : m_ds_{std::move(ds)} {
    //m_ds.LoadRasterBand(1);
}


std::vector<double> Dem::GetLocalDemFor(Dataset& image, unsigned int x_0,
                                        unsigned int y_0, unsigned int width,
                                        unsigned int height){
    auto const& dataBuffer = m_ds_.GetDataBuffer();
    auto const bandXsize = m_ds_.GetXSize();
    std::vector<double> altitudes;
    altitudes.reserve(dataBuffer.size());
    for (unsigned int iX = 0; iX < width; iX++) {
        for (unsigned int iY = 0; iY < height; iY++) {
            auto const coordinates = image.GetPixelCoordinatesFromIndex(x_0 + iX, y_0 + iY);
            auto const demIndexes =
                m_ds_.GetPixelIndexFromCoordinates(std::get<0>(coordinates), std::get<1>(coordinates));
            double value = dataBuffer.at(bandXsize * std::get<1>(demIndexes) +
                                         std::get<0>(demIndexes));

            if (value < 0) value = 0;

            altitudes.push_back(value);
        }
    }

    return altitudes;
}

}  // namespace alus
