/* Copyright 2017 The TensorFlow Authors. All Rights Reserved.

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

#include "tensorflow/c/eager/c_api.h"

#include <algorithm>
#include <cstddef>
#include <memory>
#include <string>
#include <vector>

// clang-format off
// Required for IS_MOBILE_PLATFORM
#include "tensorflow/core/platform/platform.h"
// clang-format on

#include "absl/algorithm/container.h"
#include "absl/container/fixed_array.h"
#include "absl/memory/memory.h"
#include "tensorflow/c/c_api.h"
#include "tensorflow/c/c_api_internal.h"
#include "tensorflow/c/eager/tensor_handle_interface.h"
#include "tensorflow/c/tf_tensor_internal.h"
#include "tensorflow/c/eager/c_api_experimental.h"
#include "tensorflow/c/eager/c_api_internal.h"
#include "tensorflow/core/common_runtime/device.h"
#include "tensorflow/core/common_runtime/eager/context.h"
#include "tensorflow/core/framework/device_attributes.pb.h"
#include "tensorflow/core/lib/core/errors.h"
#include "tensorflow/core/lib/core/status.h"
#include "tensorflow/core/framework/function.h"
#include "tensorflow/core/platform/errors.h"
#include "tensorflow/core/platform/platform.h"  // NOLINT
#include "tensorflow/core/protobuf/error_codes.pb.h"
#include "tensorflow/core/protobuf/device_filters.pb.h"
#include "tensorflow/core/util/device_name_utils.h"
#ifdef TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/compiler/tf2xla/xla_op_registry.h"
#endif  // TENSORFLOW_EAGER_USE_XLA
#include "tensorflow/core/common_runtime/copy_tensor.h"
#include "tensorflow/core/common_runtime/device_factory.h"
#include "tensorflow/core/common_runtime/device_mgr.h"
#include "tensorflow/core/common_runtime/device_set.h"
#include "tensorflow/core/common_runtime/eager/attr_builder.h"
#include "tensorflow/core/common_runtime/eager/copy_to_device_node.h"
#include "tensorflow/core/common_runtime/eager/execute.h"
#include "tensorflow/core/common_runtime/function.h"
#include "tensorflow/core/common_runtime/rendezvous_mgr.h"
#if !defined(IS_MOBILE_PLATFORM)
#include "tensorflow/core/distributed_runtime/eager/eager_client.h"
#include "tensorflow/core/distributed_runtime/eager/remote_mgr.h"
#include "tensorflow/core/distributed_runtime/remote_device.h"
#include "tensorflow/core/distributed_runtime/rpc/eager/grpc_eager_client.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_channel.h"
#include "tensorflow/core/distributed_runtime/rpc/grpc_server_lib.h"
#include "tensorflow/core/distributed_runtime/rpc/rpc_rendezvous_mgr.h"
#include "tensorflow/core/distributed_runtime/server_lib.h"
#include "tensorflow/core/distributed_runtime/worker_env.h"
#include "tensorflow/core/distributed_runtime/eager/cluster_function_library_runtime.h"
#endif  // !IS_MOBILE_PLATFORM
#include "tensorflow/core/framework/node_def_util.h"
#include "tensorflow/core/framework/rendezvous.h"
#include "tensorflow/core/framework/tensor_shape.pb.h"
#include "tensorflow/core/framework/types.h"
#include "tensorflow/core/lib/core/blocking_counter.h"
#include "tensorflow/core/lib/core/notification.h"
#include "tensorflow/core/lib/core/refcount.h"
#include "tensorflow/core/lib/core/stringpiece.h"
#include "tensorflow/core/lib/gtl/cleanup.h"
#include "tensorflow/core/lib/gtl/flatmap.h"
#include "tensorflow/core/lib/gtl/map_util.h"

#include "tensorflow/core/lib/random/random.h"
#include "tensorflow/core/platform/casts.h"
#include "tensorflow/core/platform/env.h"
#include "tensorflow/core/platform/mutex.h"
#include "tensorflow/core/platform/thread_annotations.h"
#include "tensorflow/core/profiler/lib/traceme.h"
#include "tensorflow/core/public/version.h"

using tensorflow::int64;
using tensorflow::string;

namespace {

const tensorflow::OpDef* GetOpDef(TFE_Op* op, TF_Status* status) {
  const tensorflow::OpDef* op_def = op->operation.OpDef();
  if (op_def) return op_def;
  status->status =
      tensorflow::OpDefForOp(op->operation.Name().c_str(), &op_def);
  return op_def;
}

bool IsCPU(
    absl::variant<tensorflow::Device*, tensorflow::CustomDevice*> variant) {
  if (VariantDeviceIsCustom(variant)) {
    return false;
  }
  tensorflow::Device* d = absl::get<tensorflow::Device*>(variant);
  return d == nullptr || d->tensorflow_gpu_device_info() == nullptr;
}

string DeviceName(const tensorflow::Device* d) {
  return (d == nullptr) ? "cpu:0" : d->name();
}

#if !defined(IS_MOBILE_PLATFORM)
tensorflow::Status AddRemoteDevicesToMgr(
    const std::vector<string>& added_remote_workers,
    tensorflow::WorkerCacheInterface* worker_cache,
    tensorflow::DynamicDeviceMgr* remote_device_mgr) {
  std::vector<std::unique_ptr<tensorflow::Device>> remote_devices;
  tensorflow::mutex remote_devices_mu;
  int num_added_workers = added_remote_workers.size();
  tensorflow::BlockingCounter counter(num_added_workers);
  std::vector<tensorflow::Status> statuses(num_added_workers);
  for (int i = 0; i < num_added_workers; i++) {
    tensorflow::NewRemoteDevices(
        tensorflow::Env::Default(), worker_cache, added_remote_workers[i],
        [i, &statuses, &counter, &remote_devices, &remote_devices_mu](
            const tensorflow::Status& s,
            std::vector<tensorflow::Device*>* devices) {
          statuses[i] = s;
          if (s.ok()) {
            tensorflow::mutex_lock l(remote_devices_mu);
            for (tensorflow::Device* d : *devices) {
              remote_devices.emplace_back(d);
            }
          }
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < num_added_workers; i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }

  TF_RETURN_IF_ERROR(remote_device_mgr->AddDevices(std::move(remote_devices)));
  return tensorflow::Status::OK();
}

tensorflow::Status GetAllRemoteDevices(
    const std::vector<string>& remote_workers,
    tensorflow::WorkerCacheInterface* worker_cache,
    std::unique_ptr<tensorflow::DynamicDeviceMgr>* device_mgr) {
  auto remote_device_mgr = absl::make_unique<tensorflow::DynamicDeviceMgr>();
  TF_RETURN_IF_ERROR(AddRemoteDevicesToMgr(remote_workers, worker_cache,
                                           remote_device_mgr.get()));
  *device_mgr = std::move(remote_device_mgr);
  return tensorflow::Status::OK();
}

tensorflow::Status RemoveRemoteDevicesFromMgr(
    const std::vector<string>& removed_remote_workers,
    tensorflow::DynamicDeviceMgr* remote_device_mgr) {
  const std::vector<tensorflow::Device*> remote_devices =
      (remote_device_mgr->ListDevices());
  std::vector<tensorflow::Device*> devices_to_remove;
  for (tensorflow::Device* d : remote_devices) {
    for (const string& remote_worker : removed_remote_workers) {
      if (tensorflow::DeviceNameUtils::IsSameAddressSpace(remote_worker,
                                                          d->name())) {
        devices_to_remove.emplace_back(d);
        break;
      }
    }
  }
  TF_RETURN_IF_ERROR(remote_device_mgr->RemoveDevices(devices_to_remove));
  return tensorflow::Status::OK();
}

tensorflow::Status ListRemoteWorkers(tensorflow::ServerInterface* server,
                                     const string& local_worker,
                                     std::vector<string>* remote_workers) {
  tensorflow::GrpcServer* grpc_server =
      dynamic_cast<tensorflow::GrpcServer*>(server);
  if (grpc_server == nullptr) {
    return tensorflow::errors::Internal(
        "Currently, TFE_NewContext only supports tensorflow::GrpcServer.");
  }
  grpc_server->master_env()->worker_cache->ListWorkers(remote_workers);
  remote_workers->erase(
      std::remove(remote_workers->begin(), remote_workers->end(), local_worker),
      remote_workers->end());
  return tensorflow::Status::OK();
}

void DifferentiateWorkerLists(const std::vector<string>* current_list,
                              const std::vector<string>* new_list,
                              std::vector<string>* added,
                              std::vector<string>* removed,
                              std::vector<string>* existing) {
  // Get STL set_difference and set_intersection with one list traversal.
  // Similar to the set_difference library function, the input lists
  // (`current_list` and `new_list`) must be sorted before calling the function.
  added->resize(new_list->size());
  removed->resize(current_list->size());
  existing->resize(current_list->size());
  std::vector<string>::const_iterator curr_it = current_list->begin();
  std::vector<string>::const_iterator new_it = new_list->begin();
  std::vector<string>::iterator added_it = added->begin();
  std::vector<string>::iterator removed_it = removed->begin();
  std::vector<string>::iterator existing_it = existing->begin();
  while (curr_it != current_list->end() && new_it != new_list->end()) {
    if (*curr_it < *new_it) {
      *removed_it++ = *curr_it++;
    } else if (*curr_it > *new_it) {
      *added_it++ = *new_it++;
    } else {
      *existing_it++ = *curr_it++;
      new_it++;
    }
  }
  removed_it = std::copy(curr_it, current_list->end(), removed_it);
  added_it = std::copy(new_it, new_list->end(), added_it);
  added->resize(added_it - added->begin());
  removed->resize(removed_it - removed->begin());
  existing->resize(existing_it - existing->begin());
}

tensorflow::Status GetReplacedFromExistingWorkers(
    const std::vector<string>* existing_workers, tensorflow::uint64 context_id,
    tensorflow::uint64 context_view_id, const tensorflow::ServerDef& server_def,
    tensorflow::eager::EagerClientCache* client_cache,
    std::vector<string>* replaced_workers) {
  tensorflow::BlockingCounter counter(existing_workers->size());
  std::vector<tensorflow::Status> statuses(existing_workers->size());
  tensorflow::eager::KeepAliveRequest request;
  request.set_context_id(context_id);
  std::vector<tensorflow::eager::KeepAliveResponse> responses(
      existing_workers->size());
  for (int i = 0; i < existing_workers->size(); i++) {
    tensorflow::core::RefCountPtr<tensorflow::eager::EagerClient> eager_client;
    statuses[i] =
        client_cache->GetClient(existing_workers->at(i), &eager_client);
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }
    eager_client->KeepAliveAsync(
        &request, &responses[i],
        [i, &statuses, &counter](const tensorflow::Status& s) {
          statuses[i] = s;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < existing_workers->size(); i++) {
    // If the RPC fails (indicating that the requested ID doesn't exist on
    // remote), or the returned view ID is not equal to the local one
    // (indicating that the remote worker has a stale view of cluster), treat
    // the worker as replaced.
    if (!statuses[i].ok() ||
        responses[i].context_view_id() != context_view_id) {
      replaced_workers->emplace_back(existing_workers->at(i));
    }
  }
  return tensorflow::Status::OK();
}

tensorflow::Status CreateRemoteContexts(
    TFE_Context* ctx, const std::vector<string>& remote_workers,
    tensorflow::uint64 context_id, tensorflow::uint64 context_view_id,
    int keep_alive_secs, const tensorflow::ServerDef& server_def,
    tensorflow::eager::EagerClientCache* remote_eager_workers, bool async,
    const bool lazy_copy_remote_function_inputs,
    const tensorflow::eager::CreateContextRequest& base_request) {
  int num_remote_workers = remote_workers.size();
  tensorflow::BlockingCounter counter(num_remote_workers);
  std::vector<tensorflow::Status> statuses(num_remote_workers);
  for (int i = 0; i < num_remote_workers; i++) {
    const string& remote_worker = remote_workers[i];
    tensorflow::DeviceNameUtils::ParsedName parsed_name;
    if (!tensorflow::DeviceNameUtils::ParseFullName(remote_worker,
                                                    &parsed_name)) {
      statuses[i] = tensorflow::errors::InvalidArgument(
          "Unable to parse ", remote_worker, " as a device name");
      counter.DecrementCount();
      continue;
    }

    tensorflow::core::RefCountPtr<tensorflow::eager::EagerClient> eager_client;
    statuses[i] = remote_eager_workers->GetClient(remote_worker, &eager_client);
    if (eager_client == nullptr) {
      statuses[i] = tensorflow::errors::Internal(
          "Cannot find a client for the given target:", remote_worker);
    }
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }

    tensorflow::eager::CreateContextRequest request;
    tensorflow::eager::CreateContextResponse* response =
        new tensorflow::eager::CreateContextResponse();
    request.set_context_id(context_id);
    request.set_context_view_id(context_view_id);
    *request.mutable_server_def() = server_def;
    request.mutable_server_def()->set_job_name(parsed_name.job);
    request.mutable_server_def()->set_task_index(parsed_name.task);
    request.mutable_server_def()->mutable_default_session_config()->MergeFrom(
        server_def.default_session_config());

    std::vector<bool> filtered_device_mask;
    ctx->context->FilterDevicesForRemoteWorkers(
        remote_worker, base_request.cluster_device_attributes(),
        &filtered_device_mask);
    DCHECK_EQ(filtered_device_mask.size(),
              base_request.cluster_device_attributes_size());
    for (int i = 0; i < filtered_device_mask.size(); i++) {
      if (filtered_device_mask[i]) {
        const auto& da = base_request.cluster_device_attributes(i);
        *request.add_cluster_device_attributes() = da;
      }
    }
    request.set_async(async);
    request.set_keep_alive_secs(keep_alive_secs);
    request.set_lazy_copy_remote_function_inputs(
        lazy_copy_remote_function_inputs);

    eager_client->CreateContextAsync(
        &request, response,
        [i, &statuses, &counter, response](const tensorflow::Status& s) {
          statuses[i] = s;
          delete response;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < num_remote_workers; i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status UpdateRemoteContexts(
    TFE_Context* ctx, const std::vector<string>& remote_workers,
    const std::vector<string>& added_workers,
    const std::vector<string>& removed_workers, tensorflow::uint64 context_id,
    tensorflow::uint64 context_view_id, const tensorflow::ServerDef& server_def,
    tensorflow::eager::EagerClientCache* remote_eager_workers,
    const tensorflow::eager::CreateContextRequest& base_request) {
  int num_remote_workers = remote_workers.size();
  tensorflow::BlockingCounter counter(num_remote_workers);
  std::vector<tensorflow::Status> statuses(num_remote_workers);

  int cluster_device_count = base_request.cluster_device_attributes_size();
  std::unordered_set<string> added_or_removed(added_workers.begin(),
                                              added_workers.end());
  std::copy(removed_workers.begin(), removed_workers.end(),
            std::inserter(added_or_removed, added_or_removed.end()));
  // Whether each device is in the updated (added or removed) workers
  std::vector<bool> device_added_or_removed(cluster_device_count);
  for (int i = 0; i < base_request.cluster_device_attributes_size(); i++) {
    const auto& da = base_request.cluster_device_attributes().at(i);
    tensorflow::DeviceNameUtils::ParsedName pn;
    tensorflow::DeviceNameUtils::ParseFullName(da.name(), &pn);
    string task_name;
    tensorflow::DeviceNameUtils::GetTaskName(pn, &task_name);
    if (added_or_removed.find(task_name) != added_or_removed.end()) {
      device_added_or_removed[i] = true;
    }
  }

  for (int i = 0; i < num_remote_workers; i++) {
    const string& remote_worker = remote_workers[i];
    tensorflow::DeviceNameUtils::ParsedName parsed_name;
    if (!tensorflow::DeviceNameUtils::ParseFullName(remote_worker,
                                                    &parsed_name)) {
      statuses[i] = tensorflow::errors::InvalidArgument(
          "Unable to parse ", remote_worker, " as a device name");
      counter.DecrementCount();
      continue;
    }

    tensorflow::core::RefCountPtr<tensorflow::eager::EagerClient> eager_client;
    statuses[i] = remote_eager_workers->GetClient(remote_worker, &eager_client);
    if (eager_client == nullptr) {
      statuses[i] = tensorflow::errors::Internal(
          "Cannot find a client for the given target:", remote_worker);
    }
    if (!statuses[i].ok()) {
      counter.DecrementCount();
      continue;
    }

    std::vector<bool> filtered_device_mask;
    ctx->context->FilterDevicesForRemoteWorkers(
        remote_worker, base_request.cluster_device_attributes(),
        &filtered_device_mask);
    DCHECK_EQ(filtered_device_mask.size(), cluster_device_count);

    // If any of the devices that match the device filters are in the set of
    // added or removed workers, we must send a complete UpdateContextRequest.
    // Otherwise, only send a simple request to increment context view ID.
    std::vector<bool> added_or_removed_filtered_devices(cluster_device_count);
    std::transform(device_added_or_removed.begin(),
                   device_added_or_removed.end(), filtered_device_mask.begin(),
                   added_or_removed_filtered_devices.begin(),
                   std::logical_and<bool>());
    const bool full_update_request =
        std::accumulate(added_or_removed_filtered_devices.begin(),
                        added_or_removed_filtered_devices.end(), false,
                        std::logical_or<bool>());

    tensorflow::eager::UpdateContextRequest request;
    auto* response = new tensorflow::eager::UpdateContextResponse();
    request.set_context_id(context_id);
    request.set_context_view_id(context_view_id);
    if (full_update_request) {
      *request.mutable_server_def() = server_def;
      request.mutable_server_def()->set_job_name(parsed_name.job);
      request.mutable_server_def()->set_task_index(parsed_name.task);
      request.mutable_server_def()->mutable_default_session_config()->MergeFrom(
          server_def.default_session_config());
      for (int i = 0; i < cluster_device_count; i++) {
        if (filtered_device_mask[i]) {
          const auto& da = base_request.cluster_device_attributes(i);
          *request.add_cluster_device_attributes() = da;
        }
      }
    }

    eager_client->UpdateContextAsync(
        &request, response,
        [i, &statuses, &counter, response](const tensorflow::Status& s) {
          statuses[i] = s;
          delete response;
          counter.DecrementCount();
        });
  }
  counter.Wait();
  for (int i = 0; i < num_remote_workers; i++) {
    TF_RETURN_IF_ERROR(statuses[i]);
  }
  return tensorflow::Status::OK();
}

tensorflow::Status UpdateTFE_ContextWithServerDef(
    int keep_alive_secs, const tensorflow::ServerDef& server_def,
    TFE_Context* ctx, bool reset_context) {
  // We don't use the TF_RETURN_IF_ERROR macro directly since that destroys the
  // server object (which currently CHECK-fails) and we miss the error, instead,
  // we log the error, and then return to allow the user to see the error
  // message.
#define LOG_AND_RETURN_IF_ERROR(...)                    \
  do {                                                  \
    const ::tensorflow::Status _status = (__VA_ARGS__); \
    if (TF_PREDICT_FALSE(!_status.ok())) {              \
      LOG(ERROR) << _status.error_message();            \
      return _status;                                   \
    }                                                   \
  } while (0);

  string worker_name =
      tensorflow::strings::StrCat("/job:", server_def.job_name(),
                                  "/replica:0/task:", server_def.task_index());

  // List of current remote workers before updating server_def. Unused if
  // resetting the server_def.
  std::vector<string> curr_remote_workers;
  // List of updated remote workers.
  std::vector<string> remote_workers;

  // New server created for new server_def. Unused if updating server_def.
  std::unique_ptr<tensorflow::ServerInterface> new_server;
  tensorflow::EagerContext* context = ctx->context;
  tensorflow::GrpcServer* grpc_server;
  if (reset_context) {
    LOG_AND_RETURN_IF_ERROR(tensorflow::NewServer(server_def, &new_server));
    grpc_server = dynamic_cast<tensorflow::GrpcServer*>(new_server.get());
    LOG_AND_RETURN_IF_ERROR(
        ListRemoteWorkers(grpc_server, worker_name, &remote_workers));
  } else {
    LOG_AND_RETURN_IF_ERROR(ListRemoteWorkers(context->GetServer(), worker_name,
                                              &curr_remote_workers));
    // No need to check the cast here, since `ListRemoteWorkers` already checks
    // if the server is a GRPC server or not.
    grpc_server = dynamic_cast<tensorflow::GrpcServer*>(context->GetServer());
    LOG_AND_RETURN_IF_ERROR(grpc_server->UpdateServerDef(server_def));
    LOG_AND_RETURN_IF_ERROR(
        ListRemoteWorkers(grpc_server, worker_name, &remote_workers));
  }

  tensorflow::uint64 context_id = context->GetContextId();
  tensorflow::uint64 context_view_id = context->GetContextViewId();
  if (reset_context) {
    context_id = tensorflow::EagerContext::NewContextId();
    context_view_id = 0;
    // Make master eager context accessible by local eager service, which might
    // receive send tensor requests from remote workers.
    LOG_AND_RETURN_IF_ERROR(
        grpc_server->AddMasterEagerContextToEagerService(context_id, context));
  }

  std::unique_ptr<tensorflow::eager::EagerClientCache> remote_eager_workers;
  LOG_AND_RETURN_IF_ERROR(
      grpc_server->master_env()->worker_cache->GetEagerClientCache(
          &remote_eager_workers));

  // When updating an existing context, populate the following lists with:
  // * added_workers: set(remote_workers) - set(curr_remote_workers)
  // * removed_workers: set(curr_remote_workers) - set(remote_workers)
  // * existing_workers: set(curr_remote_workers) intersect set(remote_workers)
  // * replaced_workers: workers with the same task names and potentially the
  //     same `hostname:port`s, but replaced by different processes
  std::vector<string> added_workers;
  std::vector<string> removed_workers;
  std::vector<string> existing_workers;
  std::vector<string> replaced_workers;

  // New remote device manager created for new server_def. Unused if updating
  // server_def.
  std::unique_ptr<tensorflow::DynamicDeviceMgr> new_remote_device_mgr;
  tensorflow::DynamicDeviceMgr* remote_device_mgr = nullptr;
  if (reset_context) {
    LOG_AND_RETURN_IF_ERROR(GetAllRemoteDevices(
        remote_workers, grpc_server->master_env()->worker_cache,
        &new_remote_device_mgr));
    remote_device_mgr = new_remote_device_mgr.get();
  } else {
    context->ClearCachesAndDefaultExecutor();
    // TODO(b/143914772): Potential memory leak if rendezvous has pending
    // tensors for removed / replaced workers.

    remote_device_mgr = context->GetOwnedRemoteDeviceMgr();
    if (remote_device_mgr == nullptr) {
      LOG_AND_RETURN_IF_ERROR(tensorflow::errors::InvalidArgument(
          "Updating context with an invalid set of remote devices."));
    }
    std::sort(curr_remote_workers.begin(), curr_remote_workers.end());
    std::sort(remote_workers.begin(), remote_workers.end());
    DifferentiateWorkerLists(&curr_remote_workers, &remote_workers,
                             &added_workers, &removed_workers,
                             &existing_workers);
    LOG_AND_RETURN_IF_ERROR(GetReplacedFromExistingWorkers(
        &existing_workers, context_id, context->GetContextViewId(), server_def,
        remote_eager_workers.get(), &replaced_workers));
    if (VLOG_IS_ON(1)) {
      VLOG(1) << "Updating cluster with following changes";
      for (const string& w : added_workers) VLOG(1) << "  Added worker " << w;
      for (const string& w : removed_workers)
        VLOG(1) << "  Removed worker " << w;
      for (const string& w : replaced_workers)
        VLOG(1) << "  Replaced worker " << w;
    }
    if (!replaced_workers.empty()) {
      // Treat replaced workers as removed then added back, so that we recreate
      // remote devices and contexts, and re-register functions on those workers
      removed_workers.insert(removed_workers.end(), replaced_workers.begin(),
                             replaced_workers.end());
      added_workers.insert(added_workers.end(), replaced_workers.begin(),
                           replaced_workers.end());
      for (const string& w : replaced_workers) {
        existing_workers.erase(
            std::remove(existing_workers.begin(), existing_workers.end(), w),
            existing_workers.end());
      }
    }
    LOG_AND_RETURN_IF_ERROR(
        RemoveRemoteDevicesFromMgr(removed_workers, remote_device_mgr));
    LOG_AND_RETURN_IF_ERROR(AddRemoteDevicesToMgr(
        added_workers, grpc_server->master_env()->worker_cache,
        remote_device_mgr));
  }

  std::vector<tensorflow::DeviceAttributes> cluster_device_attributes;
  remote_device_mgr->ListDeviceAttributes(&cluster_device_attributes);

  std::vector<tensorflow::DeviceAttributes> local_device_attributes;
  grpc_server->worker_env()->device_mgr->ListDeviceAttributes(
      &local_device_attributes);

  // This request make sure that we can create Rendezvous properly between
  // Local and Remote context.
  tensorflow::eager::CreateContextRequest base_request;
  for (const auto& da : cluster_device_attributes) {
    *base_request.add_cluster_device_attributes() = da;
  }
  for (const auto& da : local_device_attributes) {
    *base_request.add_cluster_device_attributes() = da;
  }

  // Initialize remote eager workers.
  // TODO(b/138847548) Create remote eager contexts in async mode by default.
  if (reset_context) {
    LOG_AND_RETURN_IF_ERROR(CreateRemoteContexts(
        ctx, remote_workers, context_id, context_view_id, keep_alive_secs,
        server_def, remote_eager_workers.get(), context->Executor().Async(),
        context->LazyCopyFunctionRemoteInputs(), base_request));
  } else {
    // The master's context_view_id will be incremented by one
    // the UpdateRemoteMaster call later. We want all new workers and
    // existing workers to also have the updated context_view_id, so
    // we must set their context_view_id to the existing master's
    // context_view_id + 1.
    LOG_AND_RETURN_IF_ERROR(CreateRemoteContexts(
        ctx, added_workers, context_id, context_view_id + 1, keep_alive_secs,
        server_def, remote_eager_workers.get(), context->Executor().Async(),
        context->LazyCopyFunctionRemoteInputs(), base_request));
    if (!existing_workers.empty()) {
      if (VLOG_IS_ON(1)) {
        for (const string& w : existing_workers) {
          VLOG(1) << "Updating cluster with existing worker " << w;
        }
      }
      LOG_AND_RETURN_IF_ERROR(UpdateRemoteContexts(
          ctx, existing_workers, added_workers, removed_workers, context_id,
          context_view_id + 1, server_def, remote_eager_workers.get(),
          base_request));
    }
  }

  tensorflow::RemoteRendezvous* r =
      grpc_server->worker_env()->rendezvous_mgr->Find(context_id);
  auto session_name = tensorflow::strings::StrCat("eager_", context_id);
  auto* device_mgr = grpc_server->worker_env()->device_mgr;
  std::shared_ptr<tensorflow::WorkerSession> worker_session;

  if (reset_context) {
    TF_RETURN_IF_ERROR(grpc_server->worker_env()->session_mgr->CreateSession(
        session_name, server_def, base_request.cluster_device_attributes(),
        true));
    TF_RETURN_IF_ERROR(
        grpc_server->worker_env()->session_mgr->WorkerSessionForSession(
            session_name, &worker_session));

    // Initialize remote tensor communication based on worker session.
    TF_RETURN_IF_ERROR(r->Initialize(worker_session.get()));

    tensorflow::DistributedFunctionLibraryRuntime* cluster_flr =
        tensorflow::eager::CreateClusterFLR(context_id, context,
                                            worker_session.get());
    auto remote_mgr = absl::make_unique<tensorflow::eager::RemoteMgr>(
        /*is_master=*/true, context);

    LOG_AND_RETURN_IF_ERROR(context->InitializeRemoteMaster(
        std::move(new_server), grpc_server->worker_env(), worker_session,
        std::move(remote_eager_workers), std::move(new_remote_device_mgr),
        remote_workers, context_id, r, device_mgr, keep_alive_secs, cluster_flr,
        std::move(remote_mgr)));

    // NOTE: We start the server after all other initialization, because the
    // GrpcServer cannot be destroyed after it is started.
    LOG_AND_RETURN_IF_ERROR(grpc_server->Start());
  } else {
    LOG_AND_RETURN_IF_ERROR(
        grpc_server->worker_env()->session_mgr->UpdateSession(
            session_name, server_def, base_request.cluster_device_attributes(),
            true));
    TF_RETURN_IF_ERROR(
        grpc_server->worker_env()->session_mgr->WorkerSessionForSession(
            session_name, &worker_session));
    tensorflow::DistributedFunctionLibraryRuntime* cluster_flr =
        tensorflow::eager::CreateClusterFLR(context_id, context,
                                            worker_session.get());
    LOG_AND_RETURN_IF_ERROR(context->UpdateRemoteMaster(
        grpc_server->worker_env(), std::move(remote_eager_workers),
        added_workers, removed_workers, context_id, r, device_mgr,
        keep_alive_secs, cluster_flr));
  }
#undef LOG_AND_RETURN_IF_ERROR

  return tensorflow::Status::OK();
}
#endif  // !IS_MOBILE_PLATFORM

}  // namespace

extern "C" {

TFE_ContextOptions* TFE_NewContextOptions() { return new TFE_ContextOptions; }

void TFE_ContextOptionsSetConfig(TFE_ContextOptions* options, const void* proto,
                                 size_t proto_len, TF_Status* status) {
  TF_SetConfig(&options->session_options, proto, proto_len, status);
}

void TFE_ContextOptionsSetAsync(TFE_ContextOptions* options,
                                unsigned char enable) {
  options->async = enable;
}

void TFE_ContextOptionsSetDevicePlacementPolicy(
    TFE_ContextOptions* options, TFE_ContextDevicePlacementPolicy policy) {
  options->device_placement_policy = policy;
}

void TFE_DeleteContextOptions(TFE_ContextOptions* options) { delete options; }

TFE_Context* TFE_NewContext(const TFE_ContextOptions* opts, TF_Status* status) {
  std::vector<std::unique_ptr<tensorflow::Device>> devices;
  status->status = tensorflow::DeviceFactory::AddDevices(
      opts->session_options.options, "/job:localhost/replica:0/task:0",
      &devices);
  if (!status->status.ok()) return nullptr;
  std::unique_ptr<tensorflow::DeviceMgr> device_mgr(
      new tensorflow::StaticDeviceMgr(std::move(devices)));

  tensorflow::Rendezvous* r =
      new tensorflow::IntraProcessRendezvous(device_mgr.get());

  return new TFE_Context{new tensorflow::EagerContext(
      opts->session_options.options,
      static_cast<tensorflow::ContextDevicePlacementPolicy>(
          opts->device_placement_policy),
      static_cast<tensorflow::ContextMirroringPolicy>(opts->mirroring_policy),
      opts->async, opts->lazy_remote_inputs_copy, device_mgr.release(),
      /*device_mgr_owned*/ true, r,
      tensorflow::GetDefaultCustomKernelCreator())};
}

TFE_Context* TFE_NewContextFromSession(const TFE_ContextOptions* opts,
                                       TF_Session* sess, TF_Status* status) {
  const tensorflow::DeviceMgr* device_mgr = nullptr;
  status->status = sess->session->LocalDeviceManager(&device_mgr);
  if (!status->status.ok()) return nullptr;
  tensorflow::Rendezvous* r =
      new tensorflow::IntraProcessRendezvous(device_mgr);

  return new TFE_Context{new tensorflow::EagerContext(
      opts->session_options.options,
      static_cast<tensorflow::ContextDevicePlacementPolicy>(
          opts->device_placement_policy),
      static_cast<tensorflow::ContextMirroringPolicy>(opts->mirroring_policy),
      opts->async, opts->lazy_remote_inputs_copy, device_mgr,
      /*device_mgr_owned*/ false, r,
      tensorflow::GetDefaultCustomKernelCreator())};
}

void TFE_DeleteContext(TFE_Context* ctx) {
  // context->RefCountIsOne() should be true here.
  // TODO(iga): Remove EagerContext refcounting.
  ctx->context->Unref();

  delete ctx;
}

TF_DeviceList* TFE_ContextListDevices(TFE_Context* ctx, TF_Status* status) {
  TF_DeviceList* l = new TF_DeviceList;
  ctx->context->ListDevices(&l->response);
  return l;
}

void TFE_ContextClearCaches(TFE_Context* ctx) {
  ctx->context->ClearCachesAndThreadExecutors();
}

// Set server_def on the context, possibly updating it.
TF_CAPI_EXPORT extern void TFE_ContextSetServerDef(TFE_Context* ctx,
                                                   int keep_alive_secs,
                                                   const void* proto,
                                                   size_t proto_len,
                                                   TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  }
  if (server_def.has_cluster_device_filters()) {
    const auto& cdf = server_def.cluster_device_filters();
    for (const auto& jdf : cdf.jobs()) {
      const string& remote_prefix = "/job:" + jdf.name() + "/task:";
      for (const auto& tdf : jdf.tasks()) {
        const int32_t task_index = tdf.first;
        std::vector<string> device_filters(tdf.second.device_filters_size());
        for (int i = 0; i < tdf.second.device_filters_size(); i++) {
          device_filters[i] = tdf.second.device_filters(i);
        }
        const string remote_worker = remote_prefix + std::to_string(task_index);
        status->status =
            ctx->context->SetRemoteDeviceFilters(remote_worker, device_filters);
      }
    }
  }
  status->status = UpdateTFE_ContextWithServerDef(keep_alive_secs, server_def,
                                                  ctx, /*reset_context=*/true);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextUpdateServerDef(TFE_Context* ctx,
                                                      int keep_alive_secs,
                                                      const void* proto,
                                                      size_t proto_len,
                                                      TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::ServerDef server_def;
  if (!server_def.ParseFromArray(proto, proto_len)) {
    status->status = tensorflow::errors::InvalidArgument(
        "Invalid tensorflow.ServerDef protocol buffer");
    return;
  } else if (ctx->context->GetContextId() ==
             tensorflow::EagerContext::kInvalidContextId) {
    status->status = tensorflow::errors::InvalidArgument(
        "Trying to update a context with invalid context id.");
  }
  if (server_def.has_cluster_device_filters()) {
    LOG(WARNING) << "Device filters can only be specified when initializing "
                    "the cluster. Any changes in device filters are ignored "
                    "when updating the server def.";
  }
  // TODO(haoyuzhang): Check server_def compatibility before the update
  status->status = UpdateTFE_ContextWithServerDef(keep_alive_secs, server_def,
                                                  ctx, /*reset_context=*/false);
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern bool TFE_ContextCheckAlive(TFE_Context* ctx,
                                                 const char* worker_name,
                                                 TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::errors::Unimplemented(
      "TFE_ContextSetServerDef not supported on mobile");
  return false;
#else   // !defined(IS_MOBILE_PLATFORM)
  tensorflow::EagerContext* context = ctx->context;
  tensorflow::GrpcServer* grpc_server =
      static_cast<tensorflow::GrpcServer*>(context->GetServer());

  std::unique_ptr<tensorflow::eager::EagerClientCache> remote_eager_workers;
  status->status = grpc_server->master_env()->worker_cache->GetEagerClientCache(
      &remote_eager_workers);
  if (!status->status.ok()) {
    LOG(ERROR) << "Failed to get client cache for remote workers.";
    return false;
  }

  // TODO(yuefengz): support partially specified `worker_name`.
  tensorflow::core::RefCountPtr<tensorflow::eager::EagerClient> eager_client;
  status->status = remote_eager_workers->GetClient(worker_name, &eager_client);
  if (!status->status.ok()) {
    return false;
  }

  // Send a rpc request to the worker to check aliveness.
  tensorflow::eager::KeepAliveRequest request;
  request.set_context_id(context->GetContextId());
  tensorflow::eager::KeepAliveResponse response;

  tensorflow::Status keep_alive_status;
  tensorflow::Notification done;
  eager_client->KeepAliveAsync(
      &request, &response,
      [&keep_alive_status, &done](const tensorflow::Status& s) {
        keep_alive_status = s;
        done.Notify();
      });
  done.WaitForNotification();

  status->status = tensorflow::Status::OK();

  // If `context_id` doesn't exist on the remote worker, an InvalidArgument
  // error will return. But this still indicates that the remote worker is
  // alive.
  if (keep_alive_status.ok() ||
      keep_alive_status.code() == tensorflow::error::INVALID_ARGUMENT) {
    return true;
  } else {
    LOG(INFO) << "Remote worker " << worker_name
              << " is not alive: " << keep_alive_status.error_message();
    return false;
  }
#endif  // !IS_MOBILE_PLATFORM
}

TF_CAPI_EXPORT extern void TFE_ContextClearRemoteExecutors(TFE_Context* ctx,
                                                           TF_Status* status) {
#if defined(IS_MOBILE_PLATFORM)
  status->status = tensorflow::Status::OK();
#else   // !defined(IS_MOBILE_PLATFORM)
  status->status = ctx->context->ClearRemoteExecutors();
#endif  // !IS_MOBILE_PLATFORM
}

void TFE_ContextSetThreadLocalDevicePlacementPolicy(
    TFE_Context* ctx, TFE_ContextDevicePlacementPolicy policy) {
  ctx->context->SetThreadLocalDevicePlacementPolicy(
      static_cast<tensorflow::ContextDevicePlacementPolicy>(policy));
}

// Note: this function looks up a thread local policy. So it should be called in
// the appropriate client thread. In particular, in async mode, it may not be
// safe to call this function from the async EagerExecutor threads.
extern TFE_ContextDevicePlacementPolicy TFE_ContextGetDevicePlacementPolicy(
    TFE_Context* ctx) {
  return static_cast<TFE_ContextDevicePlacementPolicy>(
      ctx->context->GetDevicePlacementPolicy());
}

TFE_TensorHandle* TFE_NewTensorHandle(TF_Tensor* t, TF_Status* status) {
  tensorflow::Tensor tensor;
  status->status = tensorflow::TF_TensorToTensor(t, &tensor);
  if (!status->status.ok()) return nullptr;
  return TFE_TensorHandle::CreateLocalHandle(tensor, status);
}

void TFE_DeleteTensorHandle(TFE_TensorHandle* h) {
  if (h == nullptr) return;
  tensorflow::profiler::TraceMe activity(
      "TFE_DeleteTensorHandle", tensorflow::profiler::TraceMeLevel::kInfo);
  delete h;
}

tensorflow::TensorHandleInterface::~TensorHandleInterface() {
  VLOG(1) << "Deleting tensor handle " << this << " with internal handle "
          << handle_;
  if (handle_) {
    handle_->Unref();
  }
}

bool tensorflow::TensorHandleInterface::IsValid(Status* status) const {
  if (handle_ == nullptr) {
    *status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return false;
  }

  return true;
}

TF_DataType TFE_TensorHandleDataType(TFE_TensorHandle* h) {
  return h->handle->DataType();
}

TF_DataType tensorflow::TensorHandleInterface::DataType() const {
  return static_cast<TF_DataType>(handle_->dtype);
}

int TFE_TensorHandleNumDims(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return -1;
  }

  return h->handle->NumDims(&status->status);
}

int tensorflow::TensorHandleInterface::NumDims(Status* status) const {
  if (!IsValid(status)) {
    return -1;
  }

  int result;
  *status = handle_->NumDims(&result);
  return result;
}

int64_t TFE_TensorHandleNumElements(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return -1;
  }

  return h->handle->NumElements(&status->status);
}

int64_t tensorflow::TensorHandleInterface::NumElements(Status* status) const {
  if (!IsValid(status)) {
    return -1;
  }

  tensorflow::int64 result;
  *status = handle_->NumElements(&result);
  return result;
}

int64_t TFE_TensorHandleDim(TFE_TensorHandle* h, int dim_index,
                            TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return -1;
  }

  return h->handle->Dim(dim_index, &status->status);
}

int64_t tensorflow::TensorHandleInterface::Dim(int dim_index,
                                               Status* status) const {
  if (!IsValid(status)) {
    return -1;
  }

  tensorflow::int64 result;
  *status = handle_->Dim(dim_index, &result);
  return result;
}

const char* TFE_TensorHandleDeviceName(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return nullptr;
  }
  return h->handle->DeviceName(&status->status);
}

const char* tensorflow::TensorHandleInterface::DeviceName(
    Status* status) const {
  if (!IsValid(status)) {
    return nullptr;
  }
  if (VariantDeviceIsCustom(handle_->device())) {
    return absl::get<CustomDevice*>(handle_->device())->name().c_str();
  }
  tensorflow::Device* d = handle_->op_device();
  return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                        : d->name().c_str();
}

const char* TFE_TensorHandleBackingDeviceName(TFE_TensorHandle* h,
                                              TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return nullptr;
  }
  return h->handle->BackingDeviceName(&status->status);
}

const char* tensorflow::TensorHandleInterface::BackingDeviceName(
    Status* status) const {
  if (!IsValid(status)) {
    return nullptr;
  }
  if (VariantDeviceIsCustom(handle_->device())) {
    return absl::get<tensorflow::CustomDevice*>(handle_->device())
        ->name()
        .c_str();
  } else {
    tensorflow::Device* d = absl::get<tensorflow::Device*>(handle_->device());
    return (d == nullptr) ? "/job:localhost/replica:0/task:0/device:CPU:0"
                          : d->name().c_str();
  }
}

TF_CAPI_EXPORT extern TFE_TensorHandle* TFE_TensorHandleCopySharingTensor(
    TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr || !h->handle->IsValid(&status->status)) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return nullptr;
  }

  return new TFE_TensorHandle{
      std::unique_ptr<AbstractTensorHandleInterface>(h->handle->Copy())};
}

AbstractTensorHandleInterface* tensorflow::TensorHandleInterface::Copy() {
  handle_->Ref();
  return new TensorHandleInterface(handle_);
}

void tensorflow::TensorHandleInterface::EnableImplicitMirroring() {
  handle_->EnableImplicitMirroring();
}

TF_Tensor* TFE_TensorHandleResolve(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return nullptr;
  }

  return h->handle->Resolve(&status->status);
}

TF_Tensor* tensorflow::TensorHandleInterface::Resolve(Status* status) {
  if (!IsValid(status)) {
    return nullptr;
  }
  if (VariantDeviceIsCustom(handle_->device())) {
    tensorflow::CustomDevice* custom_device =
        absl::get<tensorflow::CustomDevice*>(handle_->device());
    tensorflow::TensorHandle* copy;
    *status = custom_device->CopyTensorFromDevice(
        handle_, "/job:localhost/task:0/replica:0/device:CPU:0", &copy);
    if (status->ok()) {
      return TensorHandleInterface(copy).Resolve(status);
    } else {
      return nullptr;
    }
  }

  // TODO(agarwal): move this implementation inside TFE_TensorHandle.
  if (handle_->IsRemote()) {
    const tensorflow::Tensor* t = nullptr;
    tensorflow::TensorHandle* h_cpu = nullptr;
    *status = EagerCopyToDevice(handle_, handle_->Context(),
                                &handle_->Context()->Executor(),
                                handle_->Context()->HostCPU(), false, &h_cpu);
    if (!status->ok()) {
      return nullptr;
    }
    *status = h_cpu->Tensor(&t);
    if (!status->ok()) {
      h_cpu->Unref();
      return nullptr;
    }
    TF_Tensor* retval = tensorflow::TF_TensorFromTensor(*t, status);
    h_cpu->Unref();
    return retval;
  } else {
    tensorflow::Tensor tensor;
    if (IsCPU(handle_->device())) {
      const tensorflow::Tensor* src = nullptr;
      *status = handle_->Tensor(&src);
      if (!status->ok()) return nullptr;
      tensor = *src;
    } else {
      tensorflow::EagerContext* ctx = handle_->Context();
      CHECK_NE(ctx, nullptr);
      *status = handle_->CopyToDevice(*ctx, ctx->HostCPU(), &tensor);
      if (!status->ok()) return nullptr;
    }
    return tensorflow::TF_TensorFromTensor(tensor, status);
  }
}

void* TFE_TensorHandleDevicePointer(TFE_TensorHandle* h, TF_Status* status) {
  if (h == nullptr || !h->handle->IsValid(&status->status)) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return nullptr;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::down_cast<tensorflow::TensorHandleInterface*>(h->handle.get())
          ->Handle();
  if (VariantDeviceIsCustom(handle->device())) {
    const tensorflow::Tensor* t;
    status->status = handle->Tensor(&t);
    return t->data();
  }

  if (handle->IsRemote()) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDevicePointer may not be called on a remote tensor "
        "handle.");
    return nullptr;
  }
  tensorflow::Device* device(absl::get<tensorflow::Device*>(handle->device()));
  if (device != nullptr) {
    status->status = device->Sync();
    if (!status->status.ok()) {
      return nullptr;
    }
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return nullptr;
  }
  return const_cast<void*>(
      static_cast<const void*>(tensor->tensor_data().data()));
}

TFE_TensorHandle* TFE_NewTensorHandleFromDeviceMemory(
    TFE_Context* ctx, const char* device_name, TF_DataType dtype,
    const int64_t* dims, int num_dims, void* data, size_t len,
    void (*deallocator)(void* data, size_t len, void* arg),
    void* deallocator_arg, TF_Status* status) {
  tensorflow::Device* device = nullptr;
  tensorflow::EagerContext* context = ctx->context;
  status->status = context->FindDeviceFromName(device_name, &device);
  tensorflow::CustomDevice* custom_device = nullptr;
  if (!status->status.ok()) {
    status->status =
        context->FindCustomDeviceFromName(device_name, &custom_device);
    if (!status->status.ok()) {
      deallocator(data, len, deallocator_arg);
      return nullptr;
    }
  }
  std::vector<tensorflow::int64> dimvec(num_dims);
  for (int i = 0; i < num_dims; ++i) {
    dimvec[i] = static_cast<tensorflow::int64>(dims[i]);
  }

  if (dtype == TF_STRING || dtype == TF_RESOURCE ||
      !tensorflow::DataTypeCanUseMemcpy(
          static_cast<tensorflow::DataType>(dtype))) {
    status->status = tensorflow::errors::InvalidArgument(
        "Trying to create a tensor with a pointer to non-pod memory.");
    deallocator(data, len, deallocator_arg);
    return nullptr;
  }
  // TODO(apassos) do we need to wrap the deallocator here to make sure to sync
  // the device?
  TF_ManagedBuffer* buf =
      new TF_ManagedBuffer(data, len, deallocator, deallocator_arg);

  tensorflow::Tensor t(static_cast<tensorflow::DataType>(dtype),
                       tensorflow::TensorShape(dimvec), buf);
  buf->Unref();
  tensorflow::TensorHandle* ret_handle;
  if (custom_device == nullptr) {
    status->status = tensorflow::TensorHandle::CreateLocalHandle(
        t, device, context, &ret_handle);
  } else {
    status->status = tensorflow::TensorHandle::CreateLocalHandle(
        t, custom_device, context, &ret_handle);
  }
  if (!status->status.ok()) {
    return nullptr;
  }
  return new TFE_TensorHandle{
      std::make_unique<tensorflow::TensorHandleInterface>(ret_handle)};
}

// This function will block till the operation that produces `h` has
// completed. This is only valid on local TFE_TensorHandles. Returns the size in
// bytes of the memory pointed to by the device pointer returned above.
size_t TFE_TensorHandleDeviceMemorySize(TFE_TensorHandle* h,
                                        TF_Status* status) {
  if (h == nullptr || !h->handle->IsValid(&status->status)) {
    status->status = tensorflow::errors::InvalidArgument(
        "The passed in handle is a nullptr");
    return 0;
  }
  tensorflow::TensorHandle* handle =
      tensorflow::down_cast<tensorflow::TensorHandleInterface*>(h->handle.get())
          ->Handle();

  if (handle->IsRemote()) {
    status->status = tensorflow::errors::InvalidArgument(
        "TFE_TensorHandleDeviceMemorySize may not be called on a remote tensor "
        "handle.");
    return 0;
  }
  const tensorflow::Tensor* tensor;
  status->status = handle->Tensor(&tensor);
  if (!status->status.ok()) {
    return 0;
  }
  return tensor->TotalBytes();
}

TFE_Op* TFE_NewOp(TFE_Context* ctx, const char* op_or_function_name,
                  TF_Status* status) {
  std::unique_ptr<TFE_Op> new_op(
      new TFE_Op{tensorflow::EagerOperation(ctx->context)});
  status->status =
      new_op->operation.Reset(op_or_function_name, nullptr, false, nullptr);
  if (!status->status.ok()) {
    new_op.reset();
  }
  return new_op.release();
}

void TFE_DeleteOp(TFE_Op* op) { delete op; }

void TFE_OpSetDevice(TFE_Op* op, const char* device_name, TF_Status* status) {
  status->status = op->operation.SetDeviceName(device_name);
}

const char* TFE_OpGetDevice(TFE_Op* op, TF_Status* status) {
  tensorflow::Device* device = (op->operation.Device() == nullptr)
                                   ? op->operation.EagerContext().HostCPU()
                                   : op->operation.Device();
  return device->name().c_str();
}

void TFE_OpSetXLACompilation(TFE_Op* op, unsigned char enable) {
  op->operation.SetUseXla(enable);
#ifndef TENSORFLOW_EAGER_USE_XLA
  LOG(WARNING) << "This call is a no-op, as the TensorFlow library is not "
                  "built with XLA support.";
#endif  // TENSORFLOW_EAGER_USE_XLA
}

void TFE_OpAddInput(TFE_Op* op, TFE_TensorHandle* input, TF_Status* status) {
  tensorflow::TensorHandle* h =
      tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
          input->handle.get())
          ->Handle();
  op->operation.AddInput(h);
  status->status = op->operation.MaybeInferSingleInputAttrs(h);
}

void TFE_OpAddInputList(TFE_Op* op, TFE_TensorHandle** inputs, int num_inputs,
                        TF_Status* status) {
  for (int i = 0; i < num_inputs; ++i) {
    op->operation.AddInput(
        tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
            inputs[i]->handle.get())
            ->Handle());
  }
  status->status = op->operation.InferInputListAttrs(num_inputs);
}

TF_AttrType TFE_OpGetAttrType(TFE_Op* op, const char* attr_name,
                              unsigned char* is_list, TF_Status* status) {
  TF_AttrType ret = TF_ATTR_INT;
  status->status = tensorflow::AttrTypeByName(*op->operation.AttrTypes(),
                                              attr_name, &ret, is_list);
  return ret;
}

TF_AttrType TFE_OpNameGetAttrType(TFE_Context* ctx,
                                  const char* op_or_function_name,
                                  const char* attr_name, unsigned char* is_list,
                                  TF_Status* status) {
  TF_AttrType ret;
  TFE_Op* op = TFE_NewOp(ctx, op_or_function_name, status);
  if (status->status.ok()) {
    ret = TFE_OpGetAttrType(op, attr_name, is_list, status);
  } else {
    ret = TF_ATTR_INT;  // Same dummy return as TFE_OpGetAttrType.
  }
  TFE_DeleteOp(op);
  return ret;
}

void TFE_OpSetAttrString(TFE_Op* op, const char* attr_name, const void* value,
                         size_t length) {
  op->operation.MutableAttrs()->Set(
      attr_name,
      tensorflow::StringPiece(static_cast<const char*>(value), length));
}

void TFE_OpSetAttrInt(TFE_Op* op, const char* attr_name, int64_t value) {
  op->operation.MutableAttrs()->Set(attr_name, static_cast<int64>(value));
}

void TFE_OpSetAttrFloat(TFE_Op* op, const char* attr_name, float value) {
  op->operation.MutableAttrs()->Set(attr_name, value);
}

void TFE_OpSetAttrBool(TFE_Op* op, const char* attr_name, unsigned char value) {
  op->operation.MutableAttrs()->Set(attr_name, (value == 0) ? false : true);
}

void TFE_OpSetAttrType(TFE_Op* op, const char* attr_name, TF_DataType value) {
  op->operation.MutableAttrs()->Set(attr_name,
                                    static_cast<tensorflow::DataType>(value));
}

void TFE_OpSetAttrShape(TFE_Op* op, const char* attr_name, const int64_t* dims,
                        const int num_dims, TF_Status* out_status) {
  if (num_dims > tensorflow::TensorShape::MaxDimensions()) {
    TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                 tensorflow::strings::StrCat(
                     "Value specified for `", attr_name, "` has ", num_dims,
                     " dimensions which is over the limit of ",
                     tensorflow::TensorShape::MaxDimensions(), ".")
                     .c_str());
    return;
  }
  tensorflow::TensorShapeProto proto;
  if (num_dims < 0) {
    proto.set_unknown_rank(true);
  } else {
    for (int d = 0; d < num_dims; ++d) {
      proto.add_dim()->set_size(dims[d]);
    }
  }
  op->operation.MutableAttrs()->Set(attr_name, proto);
}

void TFE_OpSetAttrFunction(TFE_Op* op, const char* attr_name,
                           const TFE_Op* value) {
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(value->operation.Name());
  value->operation.Attrs().FillAttrValueMap(func->mutable_attr());
  op->operation.MutableAttrs()->Set(attr_name, attr_value);
}

void TFE_OpSetAttrFunctionName(TFE_Op* op, const char* attr_name,
                               const char* data, size_t length) {
  tensorflow::AttrValue attr_value;
  tensorflow::NameAttrList* func = attr_value.mutable_func();
  func->set_name(data, length);
  op->operation.MutableAttrs()->Set(attr_name, attr_value);
}

void TFE_OpSetAttrTensor(TFE_Op* op, const char* attr_name, TF_Tensor* tensor,
                         TF_Status* status) {
  tensorflow::Tensor t;
  status->status = TF_TensorToTensor(tensor, &t);
  if (status->status.ok()) op->operation.MutableAttrs()->Set(attr_name, t);
}

void TFE_OpSetAttrStringList(TFE_Op* op, const char* attr_name,
                             const void* const* values, const size_t* lengths,
                             int num_values) {
  std::vector<tensorflow::StringPiece> v(num_values);
  for (int i = 0; i < num_values; ++i) {
    v[i] = tensorflow::StringPiece(static_cast<const char*>(values[i]),
                                   lengths[i]);
  }
  op->operation.MutableAttrs()->Set(attr_name, v);
}

void TFE_OpSetAttrFloatList(TFE_Op* op, const char* attr_name,
                            const float* values, int num_values) {
  op->operation.MutableAttrs()->Set(
      attr_name, tensorflow::gtl::ArraySlice<const float>(values, num_values));
}

void TFE_OpSetAttrIntList(TFE_Op* op, const char* attr_name,
                          const int64_t* values, int num_values) {
  op->operation.MutableAttrs()->Set(
      attr_name, tensorflow::gtl::ArraySlice<const int64>(
                     reinterpret_cast<const int64*>(values), num_values));
}

void TFE_OpSetAttrTypeList(TFE_Op* op, const char* attr_name,
                           const TF_DataType* values, int num_values) {
  op->operation.MutableAttrs()->Set(
      attr_name,
      tensorflow::gtl::ArraySlice<const tensorflow::DataType>(
          reinterpret_cast<const tensorflow::DataType*>(values), num_values));
}

void TFE_OpSetAttrBoolList(TFE_Op* op, const char* attr_name,
                           const unsigned char* values, int num_values) {
  std::unique_ptr<bool[]> b(new bool[num_values]);
  for (int i = 0; i < num_values; ++i) {
    b[i] = values[i];
  }
  op->operation.MutableAttrs()->Set(
      attr_name, tensorflow::gtl::ArraySlice<const bool>(b.get(), num_values));
}

void TFE_OpSetAttrShapeList(TFE_Op* op, const char* attr_name,
                            const int64_t** dims, const int* num_dims,
                            int num_values, TF_Status* out_status) {
  std::unique_ptr<tensorflow::TensorShapeProto[]> proto(
      new tensorflow::TensorShapeProto[num_values]);
  for (int i = 0; i < num_values; ++i) {
    const auto num_dims_i = num_dims[i];

    if (num_dims_i > tensorflow::TensorShape::MaxDimensions()) {
      TF_SetStatus(out_status, TF_INVALID_ARGUMENT,
                   tensorflow::strings::StrCat(
                       "Value specified for `", attr_name, "` has ", num_dims_i,
                       " dimensions which is over the limit of ",
                       tensorflow::TensorShape::MaxDimensions(), ".")
                       .c_str());
      return;
    }
    if (num_dims_i < 0) {
      proto[i].set_unknown_rank(true);
    } else {
      const int64_t* dims_i = dims[i];
      auto proto_i = &proto[i];
      for (int d = 0; d < num_dims_i; ++d) {
        proto_i->add_dim()->set_size(dims_i[d]);
      }
    }
  }
  op->operation.MutableAttrs()->Set(
      attr_name, tensorflow::gtl::ArraySlice<tensorflow::TensorShapeProto>(
                     proto.get(), num_values));
}

void TFE_OpSetAttrFunctionList(TFE_Op* op, const char* attr_name,
                               const TFE_Op** value, int num_values) {
  std::unique_ptr<tensorflow::NameAttrList[]> funcs(
      new tensorflow::NameAttrList[num_values]);
  for (int i = 0; i < num_values; i++) {
    funcs[i].set_name(value[i]->operation.Name());
    value[i]->operation.Attrs().FillAttrValueMap(funcs[i].mutable_attr());
  }
  op->operation.MutableAttrs()->Set(
      attr_name, tensorflow::gtl::ArraySlice<const tensorflow::NameAttrList>(
                     funcs.get(), num_values));
}

TF_CAPI_EXPORT extern int TFE_OpGetInputLength(TFE_Op* op,
                                               const char* input_name,
                                               TF_Status* status) {
  const tensorflow::OpDef* op_def = GetOpDef(op, status);
  if (!status->status.ok()) {
    return -1;
  }
  tensorflow::AttrValueMap attrs;
  op->operation.Attrs().FillAttrValueMap(&attrs);
  tensorflow::NameRangeMap name_ranges;
  status->status = tensorflow::NameRangesForNode(
      tensorflow::AttrSlice(&attrs), *op_def, &name_ranges, nullptr);
  if (!status->status.ok()) {
    return -1;
  }
  auto iter = name_ranges.find(input_name);
  if (iter == name_ranges.end()) {
    status->status = tensorflow::errors::InvalidArgument("Input '", input_name,
                                                         "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

TF_CAPI_EXPORT extern int TFE_OpGetOutputLength(TFE_Op* op,
                                                const char* output_name,
                                                TF_Status* status) {
  const tensorflow::OpDef* op_def = GetOpDef(op, status);
  if (!status->status.ok()) {
    return -1;
  }
  tensorflow::AttrValueMap attrs;
  op->operation.Attrs().FillAttrValueMap(&attrs);
  tensorflow::NameRangeMap name_ranges;
  status->status = tensorflow::NameRangesForNode(
      tensorflow::AttrSlice(&attrs), *op_def, nullptr, &name_ranges);
  if (!status->status.ok()) {
    return -1;
  }
  auto iter = name_ranges.find(output_name);
  if (iter == name_ranges.end()) {
    status->status = tensorflow::errors::InvalidArgument(
        "Output '", output_name, "' not found");
    return -1;
  }
  return iter->second.second - iter->second.first;
}

void TFE_Execute(TFE_Op* op, TFE_TensorHandle** retvals, int* num_retvals,
                 TF_Status* status) {
  absl::FixedArray<tensorflow::TensorHandle*> handle_retvals(*num_retvals);
  VLOG(1) << "Calling TFE_Execute() on op " << op;
  status->status = tensorflow::EagerExecute(&op->operation,
                                            handle_retvals.data(), num_retvals);
  if (!status->status.ok()) {
    return;
  }
  for (int i = 0; i < *num_retvals; ++i) {
    retvals[i] = new TFE_TensorHandle{
        std::make_unique<tensorflow::TensorHandleInterface>(handle_retvals[i])};
  }
}

TFE_TensorHandle* TFE_TensorHandleCopyToDevice(TFE_TensorHandle* h,
                                               TFE_Context* ctx,
                                               const char* device_name,
                                               TF_Status* status) {
  tensorflow::TensorHandle* handle = nullptr;
  tensorflow::Device* device;
  tensorflow::EagerContext* context = ctx->context;
  status->status = context->FindDeviceFromName(device_name, &device);
  if (!status->status.ok()) {
    tensorflow::CustomDevice* dev;
    status->status = context->FindCustomDeviceFromName(device_name, &dev);
    if (status->status.ok()) {
      status->status = dev->CopyTensorToDevice(
          tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
              h->handle.get())
              ->Handle(),
          &handle);
      if (status->status.ok()) {
        return new TFE_TensorHandle{
            std::make_unique<tensorflow::TensorHandleInterface>(handle)};
      }
    }
    return nullptr;
  }
  // Handle tensor handles currently in custom devices
  const char* handle_device_name = h->handle->DeviceName(&status->status);
  if (!status->status.ok()) {
    return nullptr;
  }
  tensorflow::CustomDevice* dev;
  status->status = context->FindCustomDeviceFromName(handle_device_name, &dev);
  if (status->status.ok()) {
    status->status = dev->CopyTensorFromDevice(
        tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
            h->handle.get())
            ->Handle(),
        device_name, &handle);
    if (status->status.ok()) {
      return new TFE_TensorHandle{
          std::make_unique<tensorflow::TensorHandleInterface>(handle)};
    }
    return nullptr;
  }

  // Handle regular case.
  status->status = tensorflow::EagerCopyToDevice(
      tensorflow::down_cast<tensorflow::TensorHandleInterface*>(h->handle.get())
          ->Handle(),
      context, &context->Executor(), device, false, &handle);
  if (status->status.ok()) {
    return new TFE_TensorHandle{
        std::make_unique<tensorflow::TensorHandleInterface>(handle)};
  }
  return nullptr;
}

void TFE_ContextAddFunctionDef(TFE_Context* ctx,
                               const char* serialized_function_def, size_t size,
                               TF_Status* status) {
  tensorflow::FunctionDef function_def;
  if (!function_def.ParseFromArray(serialized_function_def, size)) {
    status->status =
        tensorflow::errors::InvalidArgument("Invalid FunctionDef proto");
    return;
  }
  status->status = ctx->context->AddFunctionDef(function_def);
}

void TFE_ContextAddFunction(TFE_Context* ctx, TF_Function* function,
                            TF_Status* status) {
  status->status = ctx->context->AddFunctionDef(function->fdef);
}

void TFE_ContextRemoveFunction(TFE_Context* ctx, const char* name,
                               TF_Status* status) {
  status->status = ctx->context->RemoveFunction(name);
}

unsigned char TFE_ContextHasFunction(TFE_Context* ctx, const char* name) {
  return ctx->context->FindFunctionDef(name) != nullptr;
}

void TFE_ContextEnableRunMetadata(TFE_Context* ctx) {
  ctx->context->SetShouldStoreGraphs(true);
}

void TFE_ContextDisableRunMetadata(TFE_Context* ctx) {
  ctx->context->SetShouldStoreGraphs(false);
}

}  // extern "C"

TFE_TensorHandle* TFE_NewTensorHandle(const tensorflow::Tensor& t,
                                      TF_Status* status) {
  return TFE_TensorHandle::CreateLocalHandle(t, status);
}

void TFE_ContextExportRunMetadata(TFE_Context* ctx, TF_Buffer* buf,
                                  TF_Status* status) {
  tensorflow::EagerContext* context = ctx->context;
  status->status = context->Executor().WaitForAllPendingNodes();
  if (!status->status.ok()) return;
  tensorflow::mutex_lock ml(*context->MetadataMu());
  status->status = MessageToBuffer(*context->RunMetadataProto(), buf);
  context->ClearRunMetadata();
}

namespace {
TFE_Op* GetFunc(TFE_Context* ctx, const tensorflow::NameAttrList& func,
                TF_Status* status) {
  TFE_Op* func_op = TFE_NewOp(ctx, func.name().data(), status);
  for (const auto& attr : func.attr()) {
    if (!status->status.ok()) return nullptr;
    SetOpAttrValueScalar(ctx, func_op, attr.second, attr.first.data(), status);
    if (!status->status.ok()) return nullptr;
  }
  return func_op;
}
}  // namespace

void TFE_ContextStartStep(TFE_Context* ctx) { ctx->context->StartStep(); }

void TFE_ContextEndStep(TFE_Context* ctx) { ctx->context->EndStep(); }

namespace tensorflow {
void SetOpAttrValueScalar(TFE_Context* ctx, TFE_Op* op,
                          const tensorflow::AttrValue& default_value,
                          const char* attr_name, TF_Status* status) {
  switch (default_value.value_case()) {
    case tensorflow::AttrValue::kS: {
      const string& v = default_value.s();
      TFE_OpSetAttrString(op, attr_name, v.data(), v.size());
      break;
    }
    case tensorflow::AttrValue::kI:
      TFE_OpSetAttrInt(op, attr_name, static_cast<int64_t>(default_value.i()));
      break;
    case tensorflow::AttrValue::kF:
      TFE_OpSetAttrFloat(op, attr_name, default_value.f());
      break;
    case tensorflow::AttrValue::kB:
      TFE_OpSetAttrBool(op, attr_name, default_value.b());
      break;
    case tensorflow::AttrValue::kType:
      TFE_OpSetAttrType(op, attr_name,
                        static_cast<TF_DataType>(default_value.type()));
      break;
    case tensorflow::AttrValue::kShape: {
      const auto& tensor_shape = default_value.shape();
      if (tensor_shape.unknown_rank()) {
        TFE_OpSetAttrShape(op, attr_name, nullptr, -1, status);
      } else {
        const auto num_dims = tensor_shape.dim_size();
        std::unique_ptr<int64_t[]> dims(new int64_t[num_dims]);
        for (int i = 0; i < num_dims; ++i) {
          dims[i] = tensor_shape.dim(i).size();
        }
        TFE_OpSetAttrShape(op, attr_name, dims.get(), num_dims, status);
      }
    } break;
    case tensorflow::AttrValue::kFunc: {
      const auto func_op = GetFunc(ctx, default_value.func(), status);
      if (!status->status.ok()) return;
      // TODO(nareshmodi): TFE_OpSetAttrFunction and TFE_OpSetAttrFunctionList
      // require TFE_Op* and just convert it internally a NameAttrValue, so
      // consider adding an overload to the C API to make this case easier.
      TFE_OpSetAttrFunction(op, attr_name, func_op);
    } break;
    case tensorflow::AttrValue::kList:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::kTensor:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::kPlaceholder:
      TF_FALLTHROUGH_INTENDED;
    case tensorflow::AttrValue::VALUE_NOT_SET:
      TF_SetStatus(
          status, TF_UNIMPLEMENTED,
          tensorflow::strings::StrCat("Unable to get setfor default value: ",
                                      default_value.DebugString())
              .data());
  }
}
}  // namespace tensorflow

namespace {
class CustomDeviceAPI : public tensorflow::CustomDevice {
 public:
  CustomDeviceAPI(TFE_CustomDevice device, void* info, string name)
      : device_(device), info_(info), name_(name) {}

  ~CustomDeviceAPI() override { device_.delete_device(info_); }

  const string& name() override { return name_; }

  tensorflow::Status CopyTensorToDevice(
      tensorflow::TensorHandle* tensor,
      tensorflow::TensorHandle** result) override {
    tensor->Ref();
    TFE_TensorHandle tensor_handle{
        std::make_unique<tensorflow::TensorHandleInterface>(tensor)};
    TF_Status status;
    TFE_TensorHandle* result_handle =
        device_.copy_tensor_to_device(&tensor_handle, &status, info_);
    if (!status.status.ok()) return status.status;
    *result = tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
                  result_handle->handle.get())
                  ->Handle();
    (*result)->Ref();
    delete result_handle;
    return status.status;
  }

  tensorflow::Status CopyTensorFromDevice(
      tensorflow::TensorHandle* tensor,
      const tensorflow::string& target_device_name,
      tensorflow::TensorHandle** result) override {
    TF_Status status;
    tensor->Ref();
    TFE_TensorHandle tensor_handle{
        std::make_unique<tensorflow::TensorHandleInterface>(tensor)};
    TFE_TensorHandle* result_handle = device_.copy_tensor_from_device(
        &tensor_handle, target_device_name.c_str(), &status, info_);
    if (!status.status.ok()) return status.status;
    *result = tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
                  result_handle->handle.get())
                  ->Handle();
    (*result)->Ref();
    delete result_handle;
    return status.status;
  }

  tensorflow::Status Execute(tensorflow::EagerOperation* op,
                             tensorflow::TensorHandle** retvals,
                             int* num_retvals) override {
    std::vector<TFE_TensorHandle*> inputs;
    inputs.reserve(op->Inputs().size());
    for (int i = 0; i < op->Inputs().size(); ++i) {
      op->Inputs()[i]->Ref();
      inputs.push_back(new TFE_TensorHandle{
          std::make_unique<tensorflow::TensorHandleInterface>(
              op->Inputs()[i])});
    }
    std::vector<TFE_TensorHandle*> outputs(*num_retvals);
    // TODO(allenl): figure out how to get attrs from EagerOperation
    TF_Status status;
    device_.execute(inputs.size(), inputs.data(), op->Name().c_str(),
                    num_retvals, outputs.data(), &status, info_);
    if (status.status.ok()) {
      for (int i = 0; i < *num_retvals; ++i) {
        retvals[i] = tensorflow::down_cast<tensorflow::TensorHandleInterface*>(
                         outputs[i]->handle.get())
                         ->Handle();
        retvals[i]->Ref();
        delete outputs[i];
      }
    }

    for (auto inp : inputs) {
      delete inp;
    }
    return status.status;
  }

 private:
  TFE_CustomDevice device_;
  void* info_;
  string name_;
};
}  // namespace

void TFE_RegisterCustomDevice(TFE_Context* ctx, TFE_CustomDevice device,
                              const char* device_name, void* device_info) {
  auto custom_device =
      std::make_unique<CustomDeviceAPI>(device, device_info, device_name);
  ctx->context->RegisterCustomDevice(device_name, std::move(custom_device));
}
