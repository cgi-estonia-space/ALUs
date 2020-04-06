#include "tests_common.hpp"

namespace slap::tests {
void silentGdalErrorHandler(CPLErr, CPLErrorNum, const char*) {
    // Not doing anything, simply silencing GDAL error log in our console by
    // supplying dummy handler.
}

}  // namespace slap::tests