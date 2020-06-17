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
#include "tensorflow/core/profiler/utils/derived_timeline.h"

#include "absl/strings/str_split.h"
#include "tensorflow/core/lib/gtl/map_util.h"
#include "tensorflow/core/profiler/protobuf/xplane.pb.h"
#include "tensorflow/core/profiler/utils/tf_op_utils.h"
#include "tensorflow/core/profiler/utils/tf_xplane_visitor.h"
#include "tensorflow/core/profiler/utils/trace_utils.h"
#include "tensorflow/core/profiler/utils/xplane_builder.h"
#include "tensorflow/core/profiler/utils/xplane_schema.h"
#include "tensorflow/core/profiler/utils/xplane_utils.h"
#include "tensorflow/core/profiler/utils/xplane_visitor.h"

namespace tensorflow {
namespace profiler {
namespace {

// Helper for deriving an XLine from events in another XLine.
class DerivedXLineBuilder {
 public:
  DerivedXLineBuilder(XPlaneBuilder* plane, int64 line_id,
                      absl::string_view name, int64 timestamp_ns,
                      std::vector<DerivedXLineBuilder*> dependent_lines,
                      bool try_expand)
      : line_(plane->GetOrCreateLine(line_id)),
        group_id_stats_(
            plane->GetOrCreateStatMetadata(GetStatTypeStr(StatType::kGroupId))),
        try_expand_(try_expand) {
    line_.SetName(name);
    line_.SetTimestampNs(timestamp_ns);
    dependent_lines_ = std::move(dependent_lines);
  }

  void ExpandOrAddEvents(
      const std::vector<const XEventMetadata*>& metadata_per_level,
      const XEventVisitor& events, absl::optional<int64> group_id) {
    for (int level = 0; level < metadata_per_level.size(); ++level) {
      ExpandOrAddLevelEvent(metadata_per_level[level], events, group_id, level);
    }
  }

  // Reset last events lower than the given level.
  void ResetLastEvents(int level = -1) {
    for (int i = level + 1; i < last_event_by_level_.size(); ++i) {
      last_event_by_level_[i] = absl::nullopt;
    }
  }

 private:
  // If the last event of the given level has the same metadata and try_expand_
  // is true, expands it to include the time until the given event's (offset_ps
  // + duration_ps). Otherwise, adds a new event and clears last_event_by_level_
  // for the levels below the given level and all levels of the dependent lines.
  // Clearing last_event_by_level_ prevents a nested event from growing larger
  // than the parent event(s).
  void ExpandOrAddLevelEvent(const XEventMetadata* event_metadata,
                             const XEventVisitor& event,
                             absl::optional<int64> group_id, int level) {
    int64 offset_ps = event.OffsetPs();
    int64 duration_ps = event.DurationPs();
    auto& last_event = last_event_by_level_[level];
    // If last_event is not nullptr, its offset must be less than or equal to
    // the given event's offset.
    DCHECK(!last_event || last_event->OffsetPs() <= offset_ps);
    if (try_expand_ && last_event &&
        last_event->MetadataId() == event_metadata->id()) {
      // If last_event is not nullptr and metadata is same, merge the given
      // event into last_event.
      last_event->SetDurationPs((offset_ps + duration_ps) -
                                last_event->OffsetPs());
    } else {
      // Otherwise, create a new event for the given level.
      last_event = line_.AddEvent(*event_metadata);
      last_event->SetOffsetPs(offset_ps);
      last_event->SetDurationPs(duration_ps);
      if (group_id) last_event->AddStatValue(*group_id_stats_, *group_id);
      // Reset last events lower than the given level.
      ResetLastEvents(level);
      if (level == 0) ResetDependentLines();
    }
  }

  void ResetDependentLines() {
    for (DerivedXLineBuilder* line : dependent_lines_) {
      line->ResetLastEvents();
    }
  }

  XLineBuilder line_;
  absl::flat_hash_map<int, absl::optional<XEventBuilder>> last_event_by_level_;
  XStatMetadata* group_id_stats_;
  std::vector<DerivedXLineBuilder*> dependent_lines_;
  bool try_expand_;
};

const absl::string_view kDerivedLineSteps = "Steps";
const absl::string_view kDerivedLineTensorFlowNameScope =
    "TensorFlow Name Scope";
const absl::string_view kDerivedLineTensorFlowOps = "TensorFlow Ops";
const absl::string_view kDerivedLineXlaModules = "XLA Modules";
const absl::string_view kDerivedLineXlaOps = "XLA Ops";
const absl::string_view kAnnotationDelimiter = "::";

void ProcessTfOpEvent(const XEventVisitor& event,
                      absl::string_view tf_op_full_name,
                      absl::optional<int64> group_id,
                      XPlaneBuilder* plane_builder,
                      DerivedXLineBuilder* tf_name_scope_line_builder,
                      DerivedXLineBuilder* tf_op_line_builder) {
  TfOp tf_op = ParseTfOpFullname(tf_op_full_name);
  if (tf_op.is_tf_op) {
    std::vector<const XEventMetadata*> tf_name_scope_metadata_per_level;
    for (const auto& tf_name_scope : ParseTfNameScopes(tf_op)) {
      tf_name_scope_metadata_per_level.push_back(
          plane_builder->GetOrCreateEventMetadata(tf_name_scope));
    }
    tf_name_scope_line_builder->ExpandOrAddEvents(
        tf_name_scope_metadata_per_level, event, group_id);
  }
  XEventMetadata* event_metadata =
      plane_builder->GetOrCreateEventMetadata(tf_op_full_name);
  // Set the display name to op_type so that the events of the same op_type have
  // the same color in the trace viewer.
  event_metadata->set_display_name(TfOpEventName(tf_op));
  tf_op_line_builder->ExpandOrAddEvents({event_metadata}, event, group_id);
}

}  // namespace

void DeriveEventsFromAnnotations(const SymbolResolver& symbol_resolver,
                                 const EventGroupNameMap& event_group_name_map,
                                 XPlane* device_trace) {
  // Merge and sort events by Timespan as they come from different lines.
  std::vector<XEventVisitor> events;
  uint64 start_timestamp_ns = 0;
  XPlaneVisitor device_plane = CreateTfXPlaneVisitor(device_trace);
  device_plane.ForEachLine([&](const XLineVisitor& line) {
    if (IsDerivedThreadId(line.Id())) return;  // Skip overhead line.
    start_timestamp_ns = line.TimestampNs();
    line.ForEachEvent(
        [&](const XEventVisitor& event) { events.push_back(event); });
  });
  absl::c_sort(events);

  XPlaneBuilder plane(device_trace);
  DerivedXLineBuilder tf_ops(&plane, kThreadIdTfOp, kDerivedLineTensorFlowOps,
                             start_timestamp_ns, {}, /*try_expand=*/true);
  DerivedXLineBuilder tf_name_scope(
      &plane, kThreadIdTfNameScope, kDerivedLineTensorFlowNameScope,
      start_timestamp_ns, {&tf_ops}, /*try_expand=*/true);
  DerivedXLineBuilder hlo_ops(&plane, kThreadIdHloOp, kDerivedLineXlaOps,
                              start_timestamp_ns, {}, /*try_expand=*/true);
  DerivedXLineBuilder hlo_modules(
      &plane, kThreadIdHloModule, kDerivedLineXlaModules, start_timestamp_ns,
      {&tf_ops, &tf_name_scope, &hlo_ops}, /*try_expand=*/false);
  DerivedXLineBuilder steps(&plane, kThreadIdStepInfo, kDerivedLineSteps,
                            start_timestamp_ns,
                            {&tf_ops, &tf_name_scope, &hlo_ops},
                            /*try_expand=*/true);

  // Process events in order by start time.
  for (const XEventVisitor& event : events) {
    absl::string_view tf_op_full_name;
    absl::string_view hlo_module_name;
    std::vector<absl::string_view> hlo_op_names;
    absl::optional<int64> group_id;
    bool is_kernel = false;
    event.ForEachStat([&](const XStatVisitor& stat) {
      if (stat.Type() == StatType::kGroupId) {
        group_id = stat.IntValue();
      } else if (stat.Type() == StatType::kLevel0) {
        tf_op_full_name = stat.StrValue();
      } else if (stat.Type() == StatType::kHloOp) {
        hlo_op_names = absl::StrSplit(stat.StrValue(), kAnnotationDelimiter);
      } else if (stat.Type() == StatType::kHloModule) {
        hlo_module_name = stat.StrValue();
      } else if (stat.Type() == StatType::kKernelDetails) {
        is_kernel = true;
      }
    });

    if (group_id) {
      if (auto group_name = gtl::FindOrNull(event_group_name_map, *group_id)) {
        steps.ExpandOrAddEvents({plane.GetOrCreateEventMetadata(*group_name)},
                                event, group_id);
      }
    }

    // For HLO/TF op lines, only use kernel events (i.e. excluding memcpy or
    // allocation events).
    if (!is_kernel) continue;

    if (!hlo_module_name.empty()) {
      hlo_modules.ExpandOrAddEvents(
          {plane.GetOrCreateEventMetadata(hlo_module_name)}, event, group_id);
    }

    if (!hlo_op_names.empty()) {  // GPU kernel compiled by XLA
      DCHECK(!hlo_module_name.empty());
      std::vector<const XEventMetadata*> hlo_op_metadata_per_level;
      for (absl::string_view hlo_op_name : hlo_op_names) {
        DCHECK(!hlo_op_name.empty());
        hlo_op_metadata_per_level.push_back(
            plane.GetOrCreateEventMetadata(hlo_op_name));
      }
      hlo_ops.ExpandOrAddEvents(hlo_op_metadata_per_level, event, group_id);
      auto tf_op_name = symbol_resolver(hlo_module_name, hlo_op_names.back());
      if (!tf_op_name.empty()) {
        ProcessTfOpEvent(event, tf_op_name, group_id, &plane, &tf_name_scope,
                         &tf_ops);
      }
    } else if (!tf_op_full_name.empty()) {  // GPU kernel not compiled by XLA
      ProcessTfOpEvent(event, tf_op_full_name, group_id, &plane, &tf_name_scope,
                       &tf_ops);
    }
  }
  RemoveEmptyLines(device_trace);
}

void GenerateDerivedTimeLines(const EventGroupNameMap& event_group_name_map,
                              XSpace* space) {
  // TODO(profiler): Once we capture HLO protos for xla/gpu, we should use that
  // to look up tensorflow op name from hlo_module/hlo_op.
  auto symbol_resolver = [&](absl::string_view hlo_module,
                             absl::string_view hlo_op) -> absl::string_view {
    return absl::string_view();
  };
  for (XPlane& plane : *space->mutable_planes()) {
    // Derived timelines only generated for device traces.
    if (plane.id() == kHostPlaneId) continue;
    DeriveEventsFromAnnotations(symbol_resolver, event_group_name_map, &plane);
  }
}

}  // namespace profiler
}  // namespace tensorflow
