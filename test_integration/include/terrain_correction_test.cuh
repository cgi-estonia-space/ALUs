#pragma once

#include <vector>

namespace alus {
namespace integrationtests {
bool AreVectorsEqual(const std::vector<double> &control, const std::vector<double> &test);
}
}