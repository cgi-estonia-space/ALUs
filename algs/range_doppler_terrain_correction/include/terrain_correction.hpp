#pragma once

#include "dataset.hpp"
#include "dem.hpp"

namespace alus {

class TerrainCorrection {
   public:

    TerrainCorrection(alus::Dataset cohDs, alus::Dataset metadata, alus::Dataset dem);

    void doWork();
    void localDemCuda();

    std::vector<double>getElevations() { return m_cohDsElevations; }

    ~TerrainCorrection();

   private:

    void localDemCPU();

    alus::Dataset m_cohDs;
    alus::Dataset m_metadataDs;
    alus::Dem m_demDs;
    std::vector<double> m_cohDsElevations;
};
}  // namespace alus