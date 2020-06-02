#include "orbit_state_vector.h"

namespace alus::snapengine {

OrbitStateVector::OrbitStateVector(
    alus::snapengine::Utc time, double xPos, double yPos, double zPos, double xVel, double yVel, double zVel)
    : time_{time},
      timeMjd_{time.getMjd()},
      xPos_{xPos},
      yPos_{yPos},
      zPos_{zPos},
      xVel_{xVel},
      yVel_{yVel},
      zVel_{zVel} {}
}  // namespace alus::snapengine