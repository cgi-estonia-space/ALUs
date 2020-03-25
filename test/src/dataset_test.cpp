#include "dataset.hpp"

#include "gmock/gmock.h"

#include <stdio.h>

namespace {

void gdalErrorHandler(CPLErr, CPLErrorNum, const char*) {
    // Not doing anything, simply silencing GDAL error log in our console by
    // supplying dummy handler.
}

std::string testFile{"/home/sven/Downloads/malbolgeEtalon_coh.tif"};

void copyTestFileOnce() {
    static bool isLoaded{false};

    if (isLoaded) {
        return;
    }

    // Load the file to unit test location.
    isLoaded = true;
}

class DatasetTest : public ::testing::Test {
   public:
    DatasetTest() {
        copyTestFileOnce();
        CPLSetErrorHandler(gdalErrorHandler);
    }

   private:
};

TEST_F(DatasetTest, onInvalidFilenameThrows) {
    std::string f{"filename"};
    ASSERT_THROW(slap::Dataset({f}), slap::DatasetError);
}

TEST_F(DatasetTest, loadsValidTifFile) {
    auto const ds = slap::Dataset(testFile);
}

}  // namespace
