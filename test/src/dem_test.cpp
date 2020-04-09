#include "dem.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include "gmock/gmock.h"

#include "tests_common.hpp"

using namespace slap::tests;

namespace {

class DemTest : public ::testing::Test {
   public:
    std::optional<slap::Dataset> demDataset;

    DemTest() {
        demDataset = std::make_optional<slap::Dataset>(DEM_PATH_1);
        demDataset.value().loadRasterBand(1);
    }

    ~DemTest() {}
};

TEST_F(DemTest, getLocalDem) {
    slap::Dem dem{std::move(demDataset.value())};
    slap::Dataset ds{TIF_PATH_1};
    ds.loadRasterBand(1);

    const auto WIDTH{ds.getGDALDataset()->GetRasterXSize()};
    const auto HEIGHT{ds.getGDALDataset()->GetRasterYSize()};
    auto const result = dem.getLocalDemFor(ds, 0, 0, WIDTH, HEIGHT);
    ASSERT_EQ(result.size(), WIDTH * HEIGHT);

    // Saaremaa is not that mountainous :)
    const auto [min, max] = std::minmax_element(begin(result), end(result));
    EXPECT_GT(*min, 20);
    EXPECT_LT(*max, 60);
    EXPECT_GT(*max, 30);
}
}  // namespace
