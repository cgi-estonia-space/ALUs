#pragma once

#include <vector>

#include "dataset.hpp"
#include "dem.hpp"
#include "terrain_correction_metadata.h"

namespace alus {

class TerrainCorrection {
   public:
    TerrainCorrection(alus::Dataset coh_ds, alus::Dataset metadata, alus::Dataset dem);
    alus::terraincorrection::RangeDopplerTerrainMetadata metadata_;

    void DoWork();
    void LocalDemCuda();

    std::vector<double> GetElevations() { return coh_ds_elevations_; }

    ~TerrainCorrection();

   private:
    void LocalDemCpu();

    alus::Dataset coh_ds_;
    alus::Dataset metadata_ds_;
    alus::Dem dem_ds_;
    std::vector<double> coh_ds_elevations_;
    void FillMetadata();
};
}  // namespace alus