#include "orbit_state_vectors.h"

#include <utility>

namespace alus {
namespace snapengine {

class OrbitStateVectors::PositionVelocity {
   public:
    PosVector position_{};
    PosVector velocity_{};
};
OrbitStateVectors::OrbitStateVectors(std::vector<coh::OrbitStateVector> orbit_state_vectors,
                                     double first_line_Utc,
                                     double line_time_interval,
                                     int source_image_height) {
    orbit_state_vectors_ = RemoveRedundantVectors(orbit_state_vectors);

    dt_ = (orbit_state_vectors_[orbit_state_vectors_.size() - 1].time_mjd_ - orbit_state_vectors_.at(0).time_mjd_) /
          (orbit_state_vectors_.size() - 1);

    sensor_position_ = std::vector<PosVector>(source_image_height);
    sensor_velocity_ = std::vector<PosVector>(source_image_height);
    for (auto i = 0; i < source_image_height; i++) {
        double time = first_line_Utc + i * line_time_interval;
        auto pv = GetPositionVelocity(time);
        sensor_position_.at(i) = pv->position_;
        sensor_velocity_.at(i) = pv->velocity_;
    }
}
OrbitStateVectors::OrbitStateVectors(std::vector<coh::OrbitStateVector> orbit_state_vectors) {
    orbit_state_vectors_ = RemoveRedundantVectors(orbit_state_vectors);
    dt_ = (orbit_state_vectors_[orbit_state_vectors_.size() - 1].time_mjd_ - orbit_state_vectors_.at(0).time_mjd_) /
          (orbit_state_vectors_.size() - 1);
}

std::vector<coh::OrbitStateVector> OrbitStateVectors::RemoveRedundantVectors(
    std::vector<coh::OrbitStateVector> orbit_state_vectors) {
    std::vector<coh::OrbitStateVector> vector_list;
    auto current_time = 0.0;
    for (uint i = 0; i < orbit_state_vectors.size(); i++) {
        if (i == 0) {
            current_time = orbit_state_vectors.at(i).time_mjd_;
            vector_list.push_back(orbit_state_vectors.at(i));
        } else if (orbit_state_vectors.at(i).time_mjd_ > current_time) {
            current_time = orbit_state_vectors.at(i).time_mjd_;
            vector_list.push_back(orbit_state_vectors.at(i));
        }
    }
    return vector_list;
}
std::shared_ptr<OrbitStateVectors::PositionVelocity> OrbitStateVectors::GetPositionVelocity(double time) {
    if (time_map_.find(time) != time_map_.end() && time_map_.at(time) != nullptr) {
        return time_map_.at(time);
    } else {
        int i0, in;
        if (orbit_state_vectors_.size() <= NV_) {
            i0 = 0;
            in = orbit_state_vectors_.size() - 1;
        } else {
            i0 = std::max((int)((time - orbit_state_vectors_.at(0).time_mjd_) / dt_) - NV_ / 2 + 1, 0);
            in = std::min(i0 + NV_ - 1, (int)orbit_state_vectors_.size() - 1);
            i0 = (in < (int)orbit_state_vectors_.size() - 1 ? i0 : in - NV_ + 1);
        }

        // lagrangeInterpolatingPolynomial
        auto pv = std::make_shared<PositionVelocity>();

        for (int i = i0; i <= in; ++i) {
            coh::OrbitStateVector orb_i = orbit_state_vectors_.at(i);

            double weight = 1;
            for (int j = i0; j <= in; ++j) {
                if (j != i) {
                    double time2 = orbit_state_vectors_.at(j).time_mjd_;
                    weight *= (time - time2) / (orb_i.time_mjd_ - time2);
                }
            }

            pv->position_.x += weight * orb_i.x_pos_;
            pv->position_.y += weight * orb_i.y_pos_;
            pv->position_.z += weight * orb_i.z_pos_;

            pv->velocity_.x += weight * orb_i.x_vel_;
            pv->velocity_.y += weight * orb_i.y_vel_;
            pv->velocity_.z += weight * orb_i.z_vel_;
        }

        time_map_.insert_or_assign(time, pv);

        return pv;
    }
}
std::unique_ptr<PosVector> OrbitStateVectors::GetPosition(double time, std::unique_ptr<PosVector> position) {
    auto n = std::make_unique<PosVector>();
    position = std::move(n);
    int i0, in;
    if (orbit_state_vectors_.size() <= NV_) {
        i0 = 0;
        in = orbit_state_vectors_.size() - 1;
    } else {
        i0 = std::max((int)((time - orbit_state_vectors_[0].time_mjd_) / dt_) - NV_ / 2 + 1, 0);
        in = std::min(i0 + NV_ - 1, (int)orbit_state_vectors_.size() - 1);
        i0 = (in < (int)orbit_state_vectors_.size() - 1 ? i0 : in - NV_ + 1);
    }

    // lagrangeInterpolatingPolynomial
    position->x = 0;
    position->y = 0;
    position->z = 0;

    for (int i = i0; i <= in; ++i) {
        auto orb_i = orbit_state_vectors_.at(i);

        double weight = 1;
        for (int j = i0; j <= in; ++j) {
            if (j != i) {
                double time2 = orbit_state_vectors_.at(j).time_mjd_;
                weight *= (time - time2) / (orb_i.time_mjd_ - time2);
            }
        }
        position->x += weight * orb_i.x_pos_;
        position->y += weight * orb_i.y_pos_;
        position->z += weight * orb_i.z_pos_;
    }
    return position;
}

std::unique_ptr<PosVector> OrbitStateVectors::GetVelocity(double time) {
    int i0, in;
    if (orbit_state_vectors_.size() <= NV_) {
        i0 = 0;
        in = orbit_state_vectors_.size() - 1;
    } else {
        i0 = std::max((int)((time - orbit_state_vectors_.at(0).time_mjd_) / dt_) - NV_ / 2 + 1, 0);
        in = std::min(i0 + NV_ - 1, (int)orbit_state_vectors_.size() - 1);
        i0 = (in < (int)orbit_state_vectors_.size() - 1 ? i0 : in - NV_ + 1);
    }

    // lagrangeInterpolatingPolynomial
    auto velocity = std::make_unique<PosVector>();

    for (int i = i0; i <= in; ++i) {
        coh::OrbitStateVector orb_i = orbit_state_vectors_.at(i);

        double weight = 1;
        for (int j = i0; j <= in; ++j) {
            if (j != i) {
                double time2 = orbit_state_vectors_[j].time_mjd_;
                weight *= (time - time2) / (orb_i.time_mjd_ - time2);
            }
        }
        velocity->x += weight * orb_i.x_vel_;
        velocity->y += weight * orb_i.y_vel_;
        velocity->z += weight * orb_i.z_vel_;
    }
    return velocity;
}
// definition of nested class

}  // namespace snapengine
}  // namespace alus
