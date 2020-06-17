/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include "tensorflow/core/profiler/convert/xplane_to_trace_events.h"

#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {

namespace {
Device BuildDeviceAndResource(const XPlaneVisitor& plane) {
  Device device;
  device.set_name(std::string(plane.Name()));
  device.set_device_id(plane.Id());
  plane.ForEachLine([&](const XLineVisitor& line) {
    Resource resource;
    resource.set_resource_id(line.Id());
    resource.set_name(std::string(line.Name()));
    (*device.mutable_resources())[line.Id()] = resource;
  });
  return device;
}
}  // namespace

void ConvertXSpaceToTraceEvents(const XSpace& xspace, Trace* trace) {
  auto* trace_devices = trace->mutable_devices();

  for (const auto& raw_plane : xspace.planes()) {
    XPlaneVisitor xplane(&raw_plane);
    // Convert devices and resources.
    int64 device_id = xplane.Id();
    (*trace_devices)[device_id] = BuildDeviceAndResource(xplane);

    // Convert events.
    xplane.ForEachLine([&](const XLineVisitor& xline) {
      int64 resource_id = xline.Id();  // Either thread id or CUDA stream id.
      xline.ForEachEvent([&](const XEventVisitor& xevent) {
        auto* event = trace->add_trace_events();
        auto& args = *event->mutable_args();
        event->set_device_id(device_id);
        event->set_resource_id(resource_id);
        if (xevent.HasDisplayName()) {
          event->set_name(string(xevent.DisplayName()));
          args["long_name"] = string(xevent.Name());
        } else {
          event->set_name(string(xevent.Name()));
        }
        event->set_timestamp_ps(xevent.TimestampPs());
        event->set_duration_ps(xevent.DurationPs());

        xevent.ForEachStat([&](const XStatVisitor& stat) {
          if (stat.ValueCase() == XStat::VALUE_NOT_SET) return;
          args[string(stat.Name())] = stat.ToString();
        });
      });
    });
  }
}

}  // namespace profiler
}  // namespace tensorflow
