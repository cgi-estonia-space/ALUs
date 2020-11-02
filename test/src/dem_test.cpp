#include "dem.hpp"

#include <algorithm>
#include <optional>
#include <string>

#include "gmock/gmock.h"

#include "tests_common.hpp"

using namespace alus::tests;

namespace {

class DemTest : public ::testing::Test {
   public:
    std::optional<alus::Dataset<double>> demDataset;

    DemTest() {
        demDataset = std::make_optional<alus::Dataset<double>>(DEM_PATH_1);
        demDataset.value().LoadRasterBand(1);
    }

    ~DemTest() {}
};

TEST_F(DemTest, getLocalDem) {
    alus::Dem dem{std::move(demDataset.value())};
    alus::Dataset<double> ds{TIF_PATH_1};
    ds.LoadRasterBand(1);

    const auto WIDTH{ds.GetGdalDataset()->GetRasterXSize()};
    const auto HEIGHT{ds.GetGdalDataset()->GetRasterYSize()};
    auto const result = dem.GetLocalDemFor(ds, 0, 0, WIDTH, HEIGHT);
    ASSERT_EQ(result.size(), WIDTH * HEIGHT);

    // Saaremaa is not that mountainous :)
    const auto [min, max] = std::minmax_element(begin(result), end(result));
    EXPECT_GT(*min, 20);
    EXPECT_LT(*max, 60);
    EXPECT_GT(*max, 30);
}
}  // namespace
