#include "tests_common.hpp"

namespace alus::tests {
void silentGdalErrorHandler(CPLErr, CPLErrorNum, const char*) {
    // Not doing anything, simply silencing GDAL error log in our console by
    // supplying dummy handler.
}

}  // namespace alus::tests
