#pragma once

#include <memory>
#include <string>
#include <string_view>
#include <vector>

#include <boost/filesystem.hpp>

#include "snap-engine-utilities/orbit_vector.h"

namespace alus {
namespace s1tbx {
class IOrbitFile {
public:
    virtual std::vector<std::string> GetAvailableOrbitTypes() = 0;

    /**
     * download, find and read orbit file
     */
    virtual boost::filesystem::path RetrieveOrbitFile(std::string_view orbit_type) = 0;

    /**
     * Get orbit information for given time.
     *
     * @param utc The UTC in days.
     * @return The orbit information.
     */
    virtual std::shared_ptr<snapengine::OrbitVector> GetOrbitData(double utc) = 0;

    /**
     * Get the orbit file used
     *
     * @return the new orbit file
     */
    virtual boost::filesystem::path GetOrbitFile() = 0;

    virtual ~IOrbitFile() = default;
};
}  // namespace s1tbx
}  // namespace alus