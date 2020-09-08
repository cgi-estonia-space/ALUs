#pragma once

#include <vector>

#include "dataset.hpp"
#include "dem.hpp"
#include "terrain_correction_metadata.h"

namespace alus {

class TerrainCorrection {
   public:
    TerrainCorrection(alus::Dataset coh_ds, alus::Dataset dem);
    alus::terraincorrection::RangeDopplerTerrainMetadata metadata_;

    void DoWork();
    void LocalDemCuda();

    /**
     * Method for executing Range Doppler Terrain Correction algorithm
     *
     * @attention Currently is not implemented and returns the same dataset that is used for tests.
     * @return Result dataset.
     */
    alus::Dataset ExecuteTerrainCorrection();

    std::vector<double> GetElevations() { return coh_ds_elevations_; }

    ~TerrainCorrection();

   private:
    void LocalDemCpu();

    alus::Dataset coh_ds_;
    alus::Dem dem_ds_;
    std::vector<double> coh_ds_elevations_;
    void FillMetadata();
};
}  // namespace alus