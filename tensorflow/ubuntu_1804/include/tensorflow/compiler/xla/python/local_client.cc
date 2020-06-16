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

// Implementation notes:
//
// Asynchronous execution:
// -----------------------
//
// Computations and host-to-device transfers do not need to block the host
// waiting for the operation to complete but instead return control to the host
// immediately. This allows Python logic to overlap with device-side
// computation.
//
// For a good user experience, we must be careful only to enqueue operations
// that are unlikely to fail; as a rule error checking must be done eagerly
// before returning control to the client.
//
// The degree to which the client can enqueue operations ahead of the client
// is limited by a semaphore. There are at two modes: asynchronous, where we
// allow the client to enqueue up to 32 executions ahead of the device, and
// synchronous, where we limit the client to having one enqueued operation at
// a time. The value of 32 is arbitrary.
//
// Even in asynchronous mode, it is important that we do not permit
// unbounded queue-ahead. Firstly it is problematic when the user does something
// like the following in Python:
// %timeit run_computation()
// To the timeit logic, op() appears to be extremely cheap since it is deferring
// all of its real work and not blocking, and so the %timeit will run op() many
// (e.g., 10000) times to get better timing resolution, even though in reality
// it may be expensive. Secondly, on CPU the allocator is synchronized with the
// head of the compute stream, and we allocate buffers for all of the enqueued
// programs without any reuse (unlike GPU). This means that the memory usage
// is proportional to the queue size.
//
// Multi-stream execution:
// -----------------------
//
// We use a multistream execution design, where different Streams are used for
// host-to-device transfers, device-to-host transfers, and compute. This allows
// us to overlap transfers on and off the device with computation.
//
// Synchronization between streams occurs via BufferDefinitionEvents that
// describe when the contents of a logical buffer are known to be valid on
// a particular stream.
//
// Synchronous vs asynchronous deallocation:
// -----------------------------------------
//
// In asynchronous deallocation mode (currently only enabled on TPU), the client
// need only keep buffers alive from its perspective until all operations that
// touch those buffers have been enqueued.
// The allocator and lower-level runtime is responsible for keeping buffers
// alive (if that is needed) from the perspective of the device until any
// device-side work actually completes. The client's use of the device allocator
// thereby corresponds to a view of the tail of the compute stream instead of
// its head.
//
// In synchronous deallocation mode the client is responsible for keeping
// buffers alive until all device-side activity that consumes those buffers has
// ceased. This is the case for CPU since HostExecutor performs allocation
// and deallocation eagerly. In this mode, the client's use of the device
// allocator is logically synchronized to the head of the compute stream, not
// the tail.

#include "tensorflow/compiler/xla/python/local_client.h"

#include <memory>
#include <string>
#include <vector>

#include "absl/base/casts.h"
#include "absl/memory/memory.h"
#include "absl/strings/str_format.h"
#include "absl/synchronization/mutex.h"
#include "absl/time/time.h"
#include "absl/types/span.h"
#include "tensorflow/compiler/xla/client/xla_computation.h"
#include "tensorflow/compiler/xla/executable_run_options.h"
#include "tensorflow/compiler/xla/literal.h"
#include "tensorflow/compiler/xla/literal_util.h"
#include "tensorflow/compiler/xla/python/shared_device_buffer.h"
#include "tensorflow/compiler/xla/shape_util.h"
#include "tensorflow/compiler/xla/util.h"
#include "tensorflow/compiler/xla/xla_data.pb.h"
#include "tensorflow/core/platform/types.h"
#include "tensorflow/core/profiler/lib/traceme.h"

namespace xla {

StatusOr<LocalDeviceState*> Device::GetLocalDeviceState() const {
  if (local_device_state_) {
    return local_device_state_.get();
  }
  return InvalidArgument("Device %s is not a local device.", DebugString());
}

std::string Device::DebugString() const {
  return absl::StrCat(platform_name(), ":", id());
}

PyLocalClient::PyLocalClient(
    std::string platform_name, LocalClient* client,
    std::vector<std::shared_ptr<Device>> devices, int host_id,
    std::unique_ptr<se::DeviceMemoryAllocator> allocator,
    std::unique_ptr<tensorflow::Allocator> host_memory_allocator)
    : platform_name_(std::move(platform_name)),
      client_(client),
      devices_(std::move(devices)),
      host_id_(host_id),
      owned_allocator_(std::move(allocator)),
      host_memory_allocator_(std::move(host_memory_allocator)),
      h2d_transfer_pool_(tensorflow::Env::Default(), "py_xla_h2d_transfer",
                         client->device_count()) {
  if (owned_allocator_ != nullptr) {
    allocator_ = owned_allocator_.get();
  } else {
    allocator_ = client_->backend().memory_allocator();
  }

  for (const std::shared_ptr<Device>& device : devices_) {
    CHECK(id_to_device_.insert({device->id(), device}).second)
        << "Duplicate device id: " << device->id();

    if (device->local_device_state()) {
      int idx = device->local_device_state()->device_ordinal();
      if (idx >= local_devices_.size()) {
        local_devices_.resize(idx + 1);
      }
      CHECK(local_devices_[idx] == nullptr) << idx;
      local_devices_[idx] = device;
    }
  }
  for (int idx = 0; idx < local_devices_.size(); ++idx) {
    CHECK(local_devices_[idx] != nullptr) << idx;
  }
}

StatusOr<DeviceAssignment> PyLocalClient::GetDefaultDeviceAssignment(
    int num_replicas, int num_partitions) const {
  return client_->backend().computation_placer()->AssignDevices(num_replicas,
                                                                num_partitions);
}

/* static */
StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::FromHostBuffer(
    const void* data, const Shape& shape, bool force_copy,
    std::shared_ptr<void> buffer_reference,
    std::shared_ptr<PyLocalClient> client, std::shared_ptr<Device> device) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::FromLiterals");
  VLOG(2) << "PyLocalBuffer::FromLiterals: shape: " << shape.ToString()
          << " device: " << device->DebugString();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      device->GetLocalDeviceState());

  // If we are on the host platform and the input buffer is sufficiently
  // aligned, we can simply point to the NumPy array's data without any further
  // copies. We require a 64-byte alignment because XLA may generate AVX512
  // code which requires it. Unfortunately NumPy's allocator doesn't align
  // quite as aggressively, so there's a high chance this test will fail.
  static constexpr int kMinimumAlignment = 64;
  if (!force_copy &&
      ((absl::bit_cast<std::uintptr_t>(data) & (kMinimumAlignment - 1)) == 0) &&
      local_device->executor()->platform_kind() == se::PlatformKind::kHost) {
    std::function<void()> on_delete_callback =
        [buffer_reference{std::move(buffer_reference)}]() {
          // Frees buffer_reference.
        };
    se::DeviceMemoryBase buffer(const_cast<void*>(data),
                                ShapeUtil::ByteSizeOf(shape));
    auto device_buffer = std::make_shared<SharedDeviceBuffer>(
        /*allocator=*/nullptr, local_device->device_ordinal(),
        std::initializer_list<se::DeviceMemoryBase>{buffer},
        /*children=*/std::vector<std::shared_ptr<SharedDeviceBuffer>>{},
        /*definition_event=*/nullptr, std::move(on_delete_callback));
    return absl::make_unique<PyLocalBuffer>(
        shape, shape, std::move(device_buffer), std::move(client),
        std::move(device));
  }

  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TF_ASSIGN_OR_RETURN(Shape compact_shape,
                      transfer_manager->ChooseCompactLayoutForShape(shape));
  TF_ASSIGN_OR_RETURN(
      ScopedShapedBuffer scoped_buffer,
      transfer_manager->AllocateScopedShapedBuffer(
          compact_shape, allocator, local_device->device_ordinal()));

  // Make the host to device stream wait for the newly allocated buffer to be
  // available on the compute stream. We schedule this wait synchronously; while
  // not strictly necessary, we must not create stream dependency cycles, and
  // adding the wait synchronously avoids any chance of any dependent
  // computations that depend on this transfer being enqueued on the compute
  // stream.
  if (!transfer_manager->CanShapedBufferBeAccessedNow(
          local_device->host_to_device_stream()->parent(), scoped_buffer)) {
    local_device->host_to_device_stream()->ThenWaitFor(
        local_device->compute_stream());
  }

  std::shared_ptr<BufferDefinitionEvent> definition_event =
      std::make_shared<BufferDefinitionEvent>();
  std::shared_ptr<SharedDeviceBuffer> device_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(&scoped_buffer,
                                                 definition_event);
  Shape on_device_shape = scoped_buffer.on_device_shape();

  auto transfer_h2d = [client, transfer_manager, local_device, device_buffer,
                       shape, compact_shape, on_device_shape, data,
                       buffer_reference{std::move(buffer_reference)}]() {
    // This function uses TF_CHECK_OK and ValueOrDie() since we have no way to
    // report failures from a callback. However, the operations here are
    // unlikely to fail and not recoverable even if we were to fail: DMAs to
    // memory that has already been allocated, and a possible Event allocation.
    ShapedBuffer buffer = device_buffer->AsShapedBuffer(
        compact_shape, on_device_shape, client->client()->platform());
    TF_CHECK_OK(transfer_manager->WriteTupleIndexTablesAsync(
        local_device->host_to_device_stream(), buffer));
    std::shared_ptr<void> staging_buffer;

    // If applicable on the backend, stage the transfer via host memory
    // allocated via the host_memory_allocator. On GPU, this is pinned memory.
    if (client->host_memory_allocator()) {
      int64 size = ShapeUtil::ByteSizeOf(shape);
      void* ptr = client->host_memory_allocator()->AllocateRaw(
          tensorflow::Allocator::kAllocatorAlignment, size);
      staging_buffer = std::shared_ptr<void>(ptr, [client](void* ptr) {
        client->host_memory_allocator()->DeallocateRaw(ptr);
      });
      std::memcpy(ptr, data, size);
      BorrowingLiteral literal(static_cast<const char*>(staging_buffer.get()),
                               shape);
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    } else {
      BorrowingLiteral literal(static_cast<const char*>(data), shape);
      // Otherwise, just transfer the literal.
      TF_CHECK_OK(transfer_manager->TransferLiteralToDeviceAsync(
          local_device->host_to_device_stream(), literal, buffer));
    }

    EventPool::Handle event =
        local_device->event_pool()
            .ThenAllocateAndRecordEvent(local_device->host_to_device_stream())
            .ValueOrDie();

    // Sets the buffer definition event. Note: this has the side effect of
    // unblocking any host threads that may have been waiting to consume the
    // buffer.
    device_buffer->definition_event()->SetDefinitionEvent(
        std::move(event), local_device->host_to_device_stream());

    if (local_device->synchronous_deallocation()) {
      local_device->ThenRelease(local_device->host_to_device_stream(),
                                device_buffer);
    }

    local_device->ThenRelease(
        local_device->host_to_device_stream(),
        std::make_pair(buffer_reference, std::move(staging_buffer)));
  };
  client->h2d_transfer_pool()->Schedule(transfer_h2d);
  return absl::make_unique<PyLocalBuffer>(
      compact_shape, std::move(on_device_shape), std::move(device_buffer),
      std::move(client), std::move(device));
}

/* static */ StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::MakeTuple(
    const std::vector<PyLocalBuffer*> buffers,
    std::shared_ptr<PyLocalClient> client, std::shared_ptr<Device> device) {
  TF_ASSIGN_OR_RETURN(LocalDeviceState * local_device,
                      device->GetLocalDeviceState());
  std::vector<Shape> host_shapes;
  std::vector<Shape> device_shapes;
  std::vector<std::shared_ptr<SharedDeviceBuffer>> device_buffers;
  host_shapes.reserve(buffers.size());
  device_shapes.reserve(buffers.size());
  device_buffers.reserve(buffers.size());
  for (const PyLocalBuffer* buffer : buffers) {
    if (buffer->device().get() != device.get()) {
      return InvalidArgument(
          "Tuple elements must be on the same device; %s vs %s",
          buffer->device()->DebugString(), device->DebugString());
    }
    std::shared_ptr<SharedDeviceBuffer> device_buffer = buffer->DeviceBuffer();
    if (!device_buffer) {
      return InvalidArgument(
          "Invalid buffer passed to MakeTuple() as argument %d.",
          device_buffers.size());
    }
    host_shapes.push_back(buffer->on_host_shape());
    device_shapes.push_back(buffer->on_device_shape());
    device_buffers.push_back(std::move(device_buffer));
  }
  se::DeviceMemoryAllocator* allocator = client->allocator();
  TransferManager* transfer_manager =
      client->client()->backend().transfer_manager();

  Shape on_host_shape = ShapeUtil::MakeTupleShape(host_shapes);
  auto definition_event = std::make_shared<BufferDefinitionEvent>();
  TF_ASSIGN_OR_RETURN(
      std::shared_ptr<SharedDeviceBuffer> tuple_buffer,
      SharedDeviceBuffer::MakeTuple(
          device_buffers, on_host_shape, transfer_manager, allocator,
          local_device->device_ordinal(), definition_event));
  auto buffer = absl::make_unique<PyLocalBuffer>(
      std::move(on_host_shape), ShapeUtil::MakeTupleShape(device_shapes),
      tuple_buffer, std::move(client), std::move(device));

  // TODO(phawkins): extend TransferManager so we do not need to form a full
  // ShapedBuffer just to write the root tuple index table.
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, buffer->AsShapedBuffer());
  if (!transfer_manager->CanShapedBufferBeAccessedNow(
          local_device->host_to_device_stream()->parent(), shaped_buffer)) {
    // Wait for the compute stream so that memory allocations are synchronized.
    local_device->host_to_device_stream()->ThenWaitFor(
        local_device->compute_stream());
  }
  TF_RETURN_IF_ERROR(transfer_manager->WriteRootTupleIndexTable(
      local_device->host_to_device_stream(), shaped_buffer));

  TF_ASSIGN_OR_RETURN(EventPool::Handle event,
                      local_device->event_pool().ThenAllocateAndRecordEvent(
                          local_device->host_to_device_stream()));
  definition_event->SetDefinitionEvent(std::move(event),
                                       local_device->host_to_device_stream());

  if (local_device->synchronous_deallocation()) {
    local_device->ThenRelease(local_device->host_to_device_stream(),
                              std::move(tuple_buffer));
  }
  return buffer;
}

PyLocalBuffer::PyLocalBuffer(Shape on_host_shape, Shape on_device_shape,
                             std::shared_ptr<SharedDeviceBuffer> device_buffer,
                             std::shared_ptr<PyLocalClient> client,
                             std::shared_ptr<Device> device)
    : client_(std::move(client)),
      on_host_shape_(std::move(on_host_shape)),
      on_device_shape_(std::move(on_device_shape)),
      device_(std::move(device)),
      device_buffer_(std::move(device_buffer)) {}

void PyLocalBuffer::Delete() {
  absl::MutexLock lock(&mu_);
  device_buffer_ = nullptr;
  host_value_ = nullptr;
}

Status PyLocalBuffer::CopyToHostAsync() {
  std::shared_ptr<SharedDeviceBuffer> device_buffer;
  std::shared_ptr<HostValue> host_value;
  {
    absl::MutexLock lock(&mu_);
    if (!device_buffer_) {
      return InvalidArgument("CopyToHostAsync() called on invalid buffer.");
    }
    device_buffer = device_buffer_;

    if (host_value_) {
      // The host value has already been requested or is available.
      return Status::OK();
    }
    host_value = host_value_ = std::make_shared<HostValue>();
  }
  se::Stream* stream = device_->local_device_state()->GetDeviceToHostStream();
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  host_value->value = std::make_shared<Literal>(on_host_shape_);
  TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, AsShapedBuffer());
  client_->client()->backend().transfer_manager()->TransferLiteralFromDevice(
      stream, shaped_buffer, host_value->value.get(),
      [host_value](Status done_status) {
        host_value->status = done_status;
        host_value->ready.Notify();
      });
  return Status::OK();
}

StatusOr<std::shared_ptr<Literal>> PyLocalBuffer::ToLiteral() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::ToLiteral");
  std::shared_ptr<SharedDeviceBuffer> device_buffer = DeviceBuffer();
  if (!device_buffer) {
    return InvalidArgument("ToLiteral() called on invalid buffer.");
  }

  TF_RETURN_IF_ERROR(CopyToHostAsync());
  std::shared_ptr<HostValue> host_value;
  {
    absl::MutexLock lock(&mu_);
    host_value = host_value_;
  }
  host_value->ready.WaitForNotification();
  TF_RETURN_IF_ERROR(host_value->status);
  return host_value->value;
}

std::shared_ptr<SharedDeviceBuffer> PyLocalBuffer::DeviceBuffer() const {
  absl::MutexLock lock(&mu_);
  return device_buffer_;
}

StatusOr<ShapedBuffer> PyLocalBuffer::AsShapedBuffer() const {
  absl::MutexLock lock(&mu_);
  if (!device_buffer_) {
    return InvalidArgument(
        "Attempted to fetch value of invalid/deleted buffer.");
  }
  return device_buffer_->AsShapedBuffer(on_host_shape_, on_device_shape_,
                                        client_->client()->platform());
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalBuffer::DestructureTuple() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::DestructureTuple");
  absl::MutexLock lock(&mu_);
  if (!on_host_shape_.IsTuple()) {
    return InvalidArgument(
        "Attempted to destructure a PyLocalBuffer that did not have a tuple "
        "shape; shape: %s",
        ShapeUtil::HumanString(on_host_shape_));
  }
  if (!device_buffer_) {
    return InvalidArgument("Attempted to destructure a deleted buffer.");
  }
  int num_children = ShapeUtil::TupleElementCount(on_host_shape_);
  std::vector<std::unique_ptr<PyLocalBuffer>> results;
  results.reserve(num_children);
  for (int64 i = 0; i < num_children; ++i) {
    results.push_back(absl::make_unique<PyLocalBuffer>(
        on_host_shape_.tuple_shapes(i), on_device_shape_.tuple_shapes(i),
        device_buffer_->children().at(i), client_, device_));
  }
  return results;
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalBuffer::CopyToDevice(
    std::shared_ptr<Device> dst_device) {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::CopyToDevice");
  std::shared_ptr<SharedDeviceBuffer> src_device_buffer = DeviceBuffer();
  TF_ASSIGN_OR_RETURN(LocalDeviceState * dst_local_device,
                      dst_device->GetLocalDeviceState());

  if (dst_device.get() == device_.get()) {
    return absl::make_unique<PyLocalBuffer>(
        on_host_shape_, on_device_shape_, src_device_buffer, client_, device_);
  }
  LocalDeviceState* transfer_local_device =
      client_->EnqueueD2DTransfersOnSrcStream() ? device_->local_device_state()
                                                : dst_local_device;

  se::Stream* transfer_stream =
      transfer_local_device->GetDeviceToDeviceStream();

  TransferManager* transfer_manager =
      client_->client()->backend().transfer_manager();
  TF_ASSIGN_OR_RETURN(ScopedShapedBuffer dst_buffer,
                      transfer_manager->AllocateScopedShapedBuffer(
                          on_host_shape_, client_->allocator(),
                          dst_local_device->device_ordinal()));
  if (!transfer_manager->CanShapedBufferBeAccessedNow(
          dst_local_device->compute_stream()->parent(), dst_buffer)) {
    transfer_stream->ThenWaitFor(dst_local_device->compute_stream());
  }
  TF_ASSIGN_OR_RETURN(ShapedBuffer src_buffer, AsShapedBuffer());

  WaitForBufferDefinitionEventsOnStream(*src_device_buffer, transfer_stream);

  // Copy the leaf buffers.
  for (const auto& leaf : src_buffer.buffers().leaves()) {
    const ShapeIndex& index = leaf.first;
    const se::DeviceMemoryBase& input_buffer = leaf.second;
    const se::DeviceMemoryBase& output_buffer = dst_buffer.buffer(index);
    TF_RET_CHECK(input_buffer.size() == output_buffer.size())
        << "input: " << input_buffer.size()
        << " output: " << output_buffer.size();
    TF_RETURN_IF_ERROR(transfer_local_device->ThenMemcpyDeviceToDevice(
        transfer_stream, dst_local_device->compute_stream(), input_buffer,
        output_buffer));
  }

  // We hold on to the `src_device_buffer` until the transfer is finished.
  transfer_local_device->ThenRelease(transfer_stream,
                                     std::move(src_device_buffer));

  // Write new tuple buffers. The destination buffers have different addresses,
  // so we must construct tuple buffers from scratch instead of copying them.
  if (dst_buffer.on_device_shape().IsTuple()) {
    TF_RETURN_IF_ERROR(transfer_manager->WriteTupleIndexTablesAsync(
        dst_local_device->host_to_device_stream(), dst_buffer));

    // We need a single definition event, so make the device to device stream
    // wait for the stream that wrote the tuple index tables on the destination
    // device.
    transfer_stream->ThenWaitFor(dst_local_device->host_to_device_stream());
  }

  auto definition_event = std::make_shared<BufferDefinitionEvent>();
  TF_ASSIGN_OR_RETURN(
      EventPool::Handle event,
      transfer_local_device->event_pool().ThenAllocateAndRecordEvent(
          transfer_stream));
  definition_event->SetDefinitionEvent(std::move(event), transfer_stream);

  std::shared_ptr<SharedDeviceBuffer> dst_device_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(&dst_buffer, definition_event);
  return absl::make_unique<PyLocalBuffer>(
      dst_buffer.on_host_shape(), dst_buffer.on_device_shape(),
      std::move(dst_device_buffer), client_, dst_device);
}

Status PyLocalBuffer::BlockHostUntilReady() {
  tensorflow::profiler::TraceMe traceme("PyLocalBuffer::BlockHostUntilReady");
  std::shared_ptr<SharedDeviceBuffer> device_buffer = DeviceBuffer();
  if (!device_buffer) {
    return InvalidArgument("BlockHostUntilReady() called on invalid buffer.");
  }

  // This code waits at least until the buffer is ready, but it may wait longer
  // if there are other device to host transfers scheduled. If this proves to
  // be an issue, we could either use a separate stream for this purpose, or
  // poll for the buffer definition events.
  se::Stream* stream =
      client_->device_state(device_->local_device_state()->device_ordinal())
          .GetDeviceToHostStream();
  WaitForBufferDefinitionEventsOnStream(*device_buffer, stream);
  return stream->BlockHostUntilDone();
}

static std::shared_ptr<Device> LookupDevice(const PyLocalClient& client,
                                            int device_id) {
  auto it = client.id_to_device().find(device_id);
  CHECK(it != client.id_to_device().end())
      << "Unknown device id: " << device_id;
  return it->second;
}

PyLocalExecutable::PyLocalExecutable(
    std::vector<std::unique_ptr<LocalExecutable>> executables,
    DeviceAssignment device_assignment, std::shared_ptr<PyLocalClient> client)
    : client_(std::move(client)),
      device_assignment_(
          std::make_shared<DeviceAssignment>(device_assignment)) {
  executables_.reserve(executables.size());
  for (auto& executable : executables) {
    executables_.emplace_back(std::move(executable));
  }

  // This must go after `executables_` is initialized.
  VLOG(1) << "PyLocalExecutable " << name() << " device_assignment:\n"
          << device_assignment_->ToString();

  const int num_replicas = device_assignment_->replica_count();
  const int num_partitions = device_assignment_->computation_count();

  // SPMD sharding produces a single executable for multiple partitions.
  if (executables_.size() > 1) {
    CHECK_EQ(num_partitions, executables_.size())
        << "Number of executables " << executables_.size()
        << " did not match number of partitions " << num_partitions;
  }

  for (int replica = 0; replica < num_replicas; ++replica) {
    for (int partition = 0; partition < num_partitions; ++partition) {
      int device_id = (*device_assignment_)(replica, partition);
      std::shared_ptr<Device> device = LookupDevice(*client_, device_id);
      if (device->host_id() != client_->host_id()) {
        VLOG(3) << "Non-local device: " << device_id;
        continue;
      }
      local_logical_device_ids_.emplace_back(replica, partition);
      local_devices_.push_back(device);
    }
  }
  CHECK_GE(local_devices_.size(), 1) << device_assignment_->ToString();
  CHECK_LE(local_devices_.size(), client_->local_device_count())
      << "Inconsistent local device count.";
}

const std::string& PyLocalExecutable::name() const {
  Executable* executable = executables_[0]->executable();
  if (executable->has_module()) {
    return executable->module().name();
  } else {
    static const std::string* unknown_name =
        new std::string("<unknown executable>");
    return *unknown_name;
  }
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalExecutable::ExecuteHelper(
    absl::Span<PyLocalBuffer* const> argument_handles, int replica,
    int partition, const RunId& run_id) {
  const int device_id = (*device_assignment_)(replica, partition);
  std::shared_ptr<Device> device = LookupDevice(*client_, device_id);
  CHECK_EQ(device->host_id(), client_->host_id());
  int device_ordinal = device->local_device_state()->device_ordinal();
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Execute");
  VLOG(3) << "Replica " << replica << ", partition " << partition
          << " mapped to device ordinal for execution: " << device_ordinal;

  absl::flat_hash_set<BufferDefinitionEvent*> events;
  std::vector<std::shared_ptr<SharedDeviceBuffer>> device_buffers;
  std::vector<ShapedBuffer> argument_buffers;
  std::vector<const ShapedBuffer*> argument_buffer_ptrs;
  device_buffers.reserve(argument_handles.size() + 1);
  argument_buffers.reserve(argument_handles.size());
  argument_buffer_ptrs.reserve(argument_handles.size());
  for (int i = 0; i < argument_handles.size(); ++i) {
    PyLocalBuffer* handle = argument_handles[i];
    std::shared_ptr<SharedDeviceBuffer> device_buffer = handle->DeviceBuffer();
    if (!device_buffer) {
      return InvalidArgument(
          "Deleted buffer passed to Execute() as argument %d to replica %d", i,
          replica);
    }
    if (handle->device().get() != device.get()) {
      return InvalidArgument(
          "Buffer passed to Execute() as argument %d to replica %d is on "
          "device %s, but replica is assigned to device %s.",
          i, replica, handle->device()->DebugString(), device->DebugString());
    }
    TF_ASSIGN_OR_RETURN(ShapedBuffer shaped_buffer, handle->AsShapedBuffer());
    argument_buffers.push_back(std::move(shaped_buffer));
    argument_buffer_ptrs.push_back(&argument_buffers.back());
    GetDeviceBufferDefinitionEvents(*device_buffer, &events);
    device_buffers.push_back(std::move(device_buffer));
    VLOG(4) << "Argument " << i
            << " buffer: " << argument_buffers.back().ToString();
  }

  LocalDeviceState* device_state = &client_->device_state(device_ordinal);

  for (BufferDefinitionEvent* event : events) {
    event->WaitForEventOnStream(device_state->compute_stream());
  }

  ExecutableRunOptions options;
  options.set_stream(device_state->compute_stream());
  options.set_host_to_device_stream(device_state->host_to_device_stream());
  options.set_allocator(client_->allocator());
  options.set_intra_op_thread_pool(
      client_->client()->backend().eigen_intra_op_thread_pool_device());
  options.set_device_assignment(device_assignment_.get());
  options.set_run_id(run_id);
  options.set_rng_seed(device_state->GetNewPrngSeed());

  // The choice of where we wait is arbitrary; the reason for the wait is pacing
  // to avoid problems such as memory fragmentation and running ahead too far,
  // not for correctness. Placing it before the executable launch allows the
  // inputs for the next executable to be fetched even if the launch is delayed.
  auto compute_reservation = std::make_shared<Semaphore::ScopedReservation>(
      device_state->compute_semaphore().ScopedAcquire(1));

  // SPMD sharding produces a single executable for multiple partitions.
  int executable_idx = executables_.size() > 1 ? partition : 0;

  StatusOr<ScopedShapedBuffer> result_buffer_or_status =
      executables_[executable_idx]->RunAsync(argument_buffer_ptrs, options);

  VLOG(1) << "Replica " << replica << " partition " << partition
          << " completed; ok=" << result_buffer_or_status.ok();
  if (!result_buffer_or_status.ok()) {
    LOG(ERROR) << "Execution of replica " << replica
               << " failed: " << result_buffer_or_status.status();
    return result_buffer_or_status.status();
  }
  ScopedShapedBuffer& result_buffer = result_buffer_or_status.ValueOrDie();

  auto definition_event = std::make_shared<BufferDefinitionEvent>();
  TF_ASSIGN_OR_RETURN(EventPool::Handle event,
                      device_state->event_pool().ThenAllocateAndRecordEvent(
                          device_state->compute_stream()));
  definition_event->SetDefinitionEvent(std::move(event),
                                       device_state->compute_stream());

  std::shared_ptr<SharedDeviceBuffer> out_buffer =
      SharedDeviceBuffer::FromScopedShapedBuffer(&result_buffer,
                                                 definition_event);

  if (device_state->synchronous_deallocation()) {
    device_buffers.push_back(out_buffer);
    device_state->ThenRelease(device_state->compute_stream(),
                              std::move(device_buffers));
  }

  device_state->ThenRelease(
      device_state->compute_stream(),
      std::make_tuple(executables_[executable_idx], compute_reservation,
                      device_assignment_));
  return absl::make_unique<PyLocalBuffer>(
      result_buffer.on_host_shape(), result_buffer.on_device_shape(),
      std::move(out_buffer), client_, device);
}

StatusOr<std::unique_ptr<PyLocalBuffer>> PyLocalExecutable::Execute(
    absl::Span<PyLocalBuffer* const> argument_handles) {
  if (num_replicas() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d replicas using Execute()",
        num_replicas());
  }
  if (num_partitions() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d partitions using Execute()",
        num_partitions());
  }
  VLOG(1) << "Executing computation " << name();
  return ExecuteHelper(argument_handles, /*replica=*/0, /*partition=*/0,
                       RunId());
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalExecutable::ExecutePerReplica(
    absl::Span<const std::vector<PyLocalBuffer*>> argument_handles) {
  tensorflow::profiler::TraceMe traceme("LocalExecutable::ExecutePerReplica");
  if (num_partitions() != 1) {
    return InvalidArgument(
        "Attempted to execute computation with %d partitions using "
        "ExecutePerReplica()",
        num_partitions());
  }
  return ExecuteOnLocalDevices(argument_handles);
}

StatusOr<std::vector<std::unique_ptr<PyLocalBuffer>>>
PyLocalExecutable::ExecuteOnLocalDevices(
    absl::Span<const std::vector<PyLocalBuffer*>> argument_handles) {
  tensorflow::profiler::TraceMe traceme(
      "LocalExecutable::ExecuteOnLocalDevices");

  const int num_local_devices = local_devices_.size();

  if (argument_handles.size() != num_local_devices) {
    return InvalidArgument(
        "Attempted to execute with %d argument lists when local device "
        "count is %d (total replica count: %d, partition count: %d)",
        argument_handles.size(), num_local_devices, num_replicas(),
        num_partitions());
  }

  VLOG(1) << "Executing computation " << name()
          << "; num_replicas=" << num_replicas()
          << " num_partitions=" << num_partitions()
          << " num_local_devices=" << num_local_devices;
  std::vector<StatusOr<std::unique_ptr<PyLocalBuffer>>> results(
      num_local_devices);
  if (num_local_devices == 1) {
    // Fast-path if there is only one device — run the computation on the
    // current thread.
    const int replica = local_logical_device_ids_[0].first;
    const int partition = local_logical_device_ids_[0].second;
    results[0] =
        ExecuteHelper(argument_handles[0], replica, partition, RunId());
  } else {
    RunId run_id;
    absl::Mutex mu;
    int running = num_local_devices;
    int failed = 0;
    Status first_failure_status;

    for (int i = 0; i < num_local_devices; ++i) {
      const int replica = local_logical_device_ids_[i].first;
      const int partition = local_logical_device_ids_[i].second;
      std::shared_ptr<Device> device = local_devices_[i];
      const LocalDeviceState& device_state = *device->local_device_state();
      device_state.execute_thread()->Schedule([&, replica, partition, i] {
        results[i] =
            ExecuteHelper(argument_handles[i], replica, partition, run_id);

        absl::MutexLock lock(&mu);
        --running;
        if (!results[i].ok()) {
          if (failed == 0) {
            first_failure_status = results[i].status();
          }
          ++failed;
        }
      });
    }

    auto done_running_or_failed = [&]() {
      mu.AssertHeld();
      return running == 0 || failed > 0;
    };
    absl::MutexLock lock(&mu);
    mu.Await(absl::Condition(&done_running_or_failed));
    if (failed > 0) {
      auto done_running = [&]() {
        mu.AssertHeld();
        return running == 0;
      };
      // If execution does not terminate within a reasonable amount of time, we
      // may be stuck at a cross-replica barrier on-device. Terminate the
      // process since that's the only way we can escape this situation at the
      // moment (b/130629719).
      if (!mu.AwaitWithTimeout(absl::Condition(&done_running),
                               absl::Seconds(10))) {
        LOG(FATAL)
            << "Replicated computation launch failed, but not all replicas "
               "terminated. Aborting process to work around deadlock. Failure "
               "message (there may have been multiple failures, see the "
               "error log for all failures): \n\n"
            << first_failure_status.error_message();
      }
    }
  }
  VLOG(1) << "Replicated execution complete.";

  std::vector<std::unique_ptr<PyLocalBuffer>> wrapped_results(
      num_local_devices);
  for (int i = 0; i < num_local_devices; ++i) {
    const int replica = local_logical_device_ids_[i].first;
    const int partition = local_logical_device_ids_[i].second;
    auto& statusor = results[i];
    if (!statusor.ok()) {
      return AppendStatus(
          statusor.status(),
          absl::StrFormat("while running replica %d and partition %d of a"
                          "replicated computation (other "
                          "replicas may have failed as well).",
                          replica, partition));
    }
    wrapped_results[i] = std::move(statusor.ValueOrDie());
  }
  return wrapped_results;
}

/*static*/ StatusOr<std::unique_ptr<PyLocalExecutable>>
PyLocalExecutable::CompileForDevices(
    const XlaComputation& computation,
    absl::optional<std::vector<Shape>> argument_layouts,
    const ExecutableBuildOptions* build_options,
    std::shared_ptr<PyLocalClient> client,
    const std::vector<std::vector<std::shared_ptr<Device>>>&
        device_assignment) {
  if (device_assignment.empty()) {
    return InvalidArgument(
        "Device assignment passed to Compile() must be non-empty.");
  }
  if (device_assignment[0].empty()) {
    return InvalidArgument(
        "Device assignment passed to Compile() must have a nonzero number of "
        "partitions per replica; replica 0 had 0 partitions.");
  }
  DeviceAssignment xla_assignment(device_assignment.size(),
                                  device_assignment[0].size());
  for (int replica = 0; replica < device_assignment.size(); ++replica) {
    if (device_assignment[replica].size() != device_assignment[0].size()) {
      return InvalidArgument(
          "Device assignment passed to Compile() has different numbers of "
          "partitions between replicas; %d partitions for replica %d versus %d "
          "partitions for replica 0.",
          device_assignment[replica].size(), replica,
          device_assignment[0].size());
    }
    for (int partition = 0; partition < device_assignment[replica].size();
         ++partition) {
      if (device_assignment[0][0]->platform_name() !=
          device_assignment[replica][partition]->platform_name()) {
        return InvalidArgument(
            "Device assignment passed to Compile() must have devices of a "
            "single kind, got %s for replica 0 partition 0 and %s for replica "
            "%d partition %d.",
            device_assignment[0][0]->platform_name(),
            device_assignment[replica][partition]->platform_name(), replica,
            partition);
      }
      xla_assignment(replica, partition) =
          device_assignment[replica][partition]->id();
    }
  }
  return Compile(computation, std::move(argument_layouts), build_options,
                 std::move(client), xla_assignment);
}

/*static*/ StatusOr<std::unique_ptr<PyLocalExecutable>>
PyLocalExecutable::Compile(const XlaComputation& computation,
                           absl::optional<std::vector<Shape>> argument_layouts,
                           const ExecutableBuildOptions* build_options,
                           std::shared_ptr<PyLocalClient> client,
                           absl::optional<DeviceAssignment> device_assignment) {
  tensorflow::profiler::TraceMe traceme("LocalExecutable::Compile");

  ExecutableBuildOptions options;
  if (build_options != nullptr) {
    options = *build_options;
  }

  if (!options.device_allocator()) {
    options.set_device_allocator(client->allocator());
  }

  if (device_assignment) {
    VLOG(2) << "PyLocalExecutable::Compile got device_assignment:\n"
            << device_assignment->ToString();
    if (device_assignment->replica_count() != options.num_replicas()) {
      return InvalidArgument(
          "Mismatched number of replicas for device "
          "assignment and computation (%d vs %d).\n%s",
          device_assignment->replica_count(), options.num_replicas(),
          device_assignment->ToString());
    }
    if (device_assignment->computation_count() != options.num_partitions()) {
      return InvalidArgument(
          "Mismatched number of partitions for device "
          "assignment and computation (%d vs %d).\n%s",
          device_assignment->computation_count(), options.num_partitions(),
          device_assignment->ToString());
    }
  } else {
    TF_ASSIGN_OR_RETURN(device_assignment,
                        client->GetDefaultDeviceAssignment(
                            options.num_replicas(), options.num_partitions()));
    VLOG(2) << "PyLocalExecutable::Compile using default device_assignment:\n"
            << device_assignment->ToString();
  }

  if (!argument_layouts) {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    argument_layouts = program_shape.parameters();
    for (Shape& shape : *argument_layouts) {
      LayoutUtil::ClearLayout(&shape);
    }
  }
  std::vector<const Shape*> argument_layout_pointers;
  argument_layout_pointers.reserve(argument_layouts->size());

  // Assign a default layout to any array subshapes that are missing layouts.
  auto assign_layouts = [client](Shape* shape) {
    return ShapeUtil::ForEachMutableSubshapeWithStatus(
        shape, [&](Shape* subshape, const ShapeIndex&) {
          if (subshape->IsArray() && !subshape->has_layout()) {
            LayoutUtil::SetToDefaultLayout(subshape);
            TF_ASSIGN_OR_RETURN(*subshape,
                                client->client()
                                    ->backend()
                                    .transfer_manager()
                                    ->ChooseCompactLayoutForShape(*subshape));
          }
          return Status::OK();
        });
  };

  for (Shape& layout : *argument_layouts) {
    argument_layout_pointers.push_back(&layout);
    TF_RETURN_IF_ERROR(assign_layouts(&layout));
  }

  Shape result_layout;
  if (options.result_layout()) {
    result_layout = *options.result_layout();
  } else {
    TF_ASSIGN_OR_RETURN(ProgramShape program_shape,
                        computation.GetProgramShape());
    result_layout = program_shape.result();
    LayoutUtil::ClearLayout(&result_layout);
  }
  TF_RETURN_IF_ERROR(assign_layouts(&result_layout));
  options.set_result_layout(result_layout);

  TF_ASSIGN_OR_RETURN(
      std::vector<std::unique_ptr<LocalExecutable>> local_executables,
      client->client()->Compile(computation, argument_layout_pointers,
                                options));

  return absl::make_unique<PyLocalExecutable>(std::move(local_executables),
                                              std::move(*device_assignment),
                                              std::move(client));
}

}  // namespace xla
