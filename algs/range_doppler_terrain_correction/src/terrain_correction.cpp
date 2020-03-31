#include "terrain_correction.hpp"

#include <algorithm>
#include <iostream>

#include "dem.hpp"

namespace slap {

void TerrainCorrection::doWork() {
    auto const result = m_demDs.getLocalDemFor(
        m_cohDs, 0, 0, m_cohDs.getBand1Xsize(), m_cohDs.getBand1Ysize());

    const auto [min, max] =
        std::minmax_element(std::begin(result), std::end(result));
    std::cout << "Our area has lowest point at " << *min << " and highest at "
              << *max << std::endl;
}

TerrainCorrection::~TerrainCorrection() {}
}  // namespace slap
