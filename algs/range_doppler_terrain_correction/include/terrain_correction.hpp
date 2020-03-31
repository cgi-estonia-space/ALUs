#pragma once

#include "dataset.hpp"
#include "dem.hpp"

namespace slap {

class TerrainCorrection {
   public:
    TerrainCorrection(slap::Dataset cohDs, slap::Dataset metadata,
                      slap::Dataset dem)
        : m_cohDs{std::move(cohDs)},
          m_metadataDs{std::move(metadata)},
          m_demDs{std::move(dem)} {}

    void doWork();

    ~TerrainCorrection();

   private:
    slap::Dataset m_cohDs;
    slap::Dataset m_metadataDs;
    slap::Dem m_demDs;
};
}  // namespace slap