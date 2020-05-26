#pragma once

#include "dataset.hpp"
#include "dem.hpp"

namespace slap {

class TerrainCorrection {
   public:

    TerrainCorrection(slap::Dataset cohDs,
                      slap::Dataset metadata, slap::Dataset dem);

    void doWork();
    void localDemCuda();

    std::vector<double>getElevations() { return m_cohDsElevations; }

    ~TerrainCorrection();

   private:

    void localDemCPU();

    slap::Dataset m_cohDs;
    slap::Dataset m_metadataDs;
    slap::Dem m_demDs;
    std::vector<double> m_cohDsElevations;
};
}  // namespace slap