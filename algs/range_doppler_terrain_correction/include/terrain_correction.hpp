#pragma once

#include <memory>

#include "dataset.hpp"

namespace slap {

class TerrainCorrection {
   public:
    TerrainCorrection(std::shared_ptr<slap::Dataset> ds) : m_ds{ds} {}

    void doWork();

    ~TerrainCorrection();

   private:
    std::shared_ptr<slap::Dataset> m_ds;
};
}  // namespace slap