#pragma once

#include "dataset.hpp"

namespace slap {
class Dem {
   public:
    Dem(Dataset ds);

    void doWork();

   private:
    Dataset m_ds;

    double m_noDataValue;
};
}  // namespace slap