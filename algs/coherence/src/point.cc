#include "point.h"

namespace alus {

Point::Point() {
    this->x_ = 0;
    this->y_ = 0;
    this->z_ = 0;
}

Point::Point(double x, double y, double z) : x_{x}, y_{y}, z_{z} {}

void Point::SetX(double x) { this->x_ = x; }
void Point::SetY(double y) { this->y_ = y; }
void Point::SetZ(double z) { this->z_ = z; }

Point Point::Min(Point p) {
    double dx = this->x_ - p.x_;
    double dy = this->y_ - p.y_;
    double dz = this->z_ - p.z_;
    return Point(dx, dy, dz);
}

// inner product
double Point::In(Point &p) {
    double dx = x_ * p.x_;
    double dy = y_ * p.y_;
    double dz = z_ * p.z_;
    return (dx + dy + dz);
}

double Point::Norm() { return sqrt(x_ * x_ + y_ * y_ + z_ * z_); }

}  // namespace alus
