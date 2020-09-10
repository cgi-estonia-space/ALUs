// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/queue_runner.proto

#include "tensorflow/core/protobuf/queue_runner.pb.h"

#include <algorithm>

#include <google/protobuf/stubs/common.h>
#include <google/protobuf/io/coded_stream.h>
#include <google/protobuf/extension_set.h>
#include <google/protobuf/wire_format_lite.h>
#include <google/protobuf/descriptor.h>
#include <google/protobuf/generated_message_reflection.h>
#include <google/protobuf/reflection_ops.h>
#include <google/protobuf/wire_format.h>
// @@protoc_insertion_point(includes)
#include <google/protobuf/port_def.inc>
namespace tensorflow {
class QueueRunnerDefDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<QueueRunnerDef> _instance;
} _QueueRunnerDef_default_instance_;
}  // namespace tensorflow
static void InitDefaultsscc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_QueueRunnerDef_default_instance_;
    new (ptr) ::tensorflow::QueueRunnerDef();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::QueueRunnerDef::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsscc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, queue_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, enqueue_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, close_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, cancel_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::QueueRunnerDef, queue_closed_exception_types_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::QueueRunnerDef)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_QueueRunnerDef_default_instance_),
};

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto[] =
  "\n+tensorflow/core/protobuf/queue_runner."
  "proto\022\ntensorflow\032*tensorflow/core/proto"
  "buf/error_codes.proto\"\252\001\n\016QueueRunnerDef"
  "\022\022\n\nqueue_name\030\001 \001(\t\022\027\n\017enqueue_op_name\030"
  "\002 \003(\t\022\025\n\rclose_op_name\030\003 \001(\t\022\026\n\016cancel_o"
  "p_name\030\004 \001(\t\022<\n\034queue_closed_exception_t"
  "ypes\030\005 \003(\0162\026.tensorflow.error.CodeB|\n\030or"
  "g.tensorflow.frameworkB\021QueueRunnerProto"
  "sP\001ZHgithub.com/tensorflow/tensorflow/te"
  "nsorflow/go/core/core_protos_go_proto\370\001\001"
  "b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_deps[1] = {
  &::descriptor_table_tensorflow_2fcore_2fprotobuf_2ferror_5fcodes_2eproto,
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_sccs[1] = {
  &scc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_once;
static bool descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = {
  &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_initialized, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto, "tensorflow/core/protobuf/queue_runner.proto", 408,
  &descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_once, descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_sccs, descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto_deps, 1, 1,
  schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto::offsets,
  file_level_metadata_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto, 1, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto, file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto), true);
namespace tensorflow {

// ===================================================================

void QueueRunnerDef::InitAsDefaultInstance() {
}
class QueueRunnerDef::HasBitSetters {
 public:
};

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int QueueRunnerDef::kQueueNameFieldNumber;
const int QueueRunnerDef::kEnqueueOpNameFieldNumber;
const int QueueRunnerDef::kCloseOpNameFieldNumber;
const int QueueRunnerDef::kCancelOpNameFieldNumber;
const int QueueRunnerDef::kQueueClosedExceptionTypesFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

QueueRunnerDef::QueueRunnerDef()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.QueueRunnerDef)
}
QueueRunnerDef::QueueRunnerDef(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena),
  enqueue_op_name_(arena),
  queue_closed_exception_types_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.QueueRunnerDef)
}
QueueRunnerDef::QueueRunnerDef(const QueueRunnerDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      enqueue_op_name_(from.enqueue_op_name_),
      queue_closed_exception_types_(from.queue_closed_exception_types_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  queue_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.queue_name().size() > 0) {
    queue_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.queue_name(),
      GetArenaNoVirtual());
  }
  close_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.close_op_name().size() > 0) {
    close_op_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.close_op_name(),
      GetArenaNoVirtual());
  }
  cancel_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.cancel_op_name().size() > 0) {
    cancel_op_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.cancel_op_name(),
      GetArenaNoVirtual());
  }
  // @@protoc_insertion_point(copy_constructor:tensorflow.QueueRunnerDef)
}

void QueueRunnerDef::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto.base);
  queue_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  close_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  cancel_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

QueueRunnerDef::~QueueRunnerDef() {
  // @@protoc_insertion_point(destructor:tensorflow.QueueRunnerDef)
  SharedDtor();
}

void QueueRunnerDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  queue_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  close_op_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  cancel_op_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void QueueRunnerDef::ArenaDtor(void* object) {
  QueueRunnerDef* _this = reinterpret_cast< QueueRunnerDef* >(object);
  (void)_this;
}
void QueueRunnerDef::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void QueueRunnerDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const QueueRunnerDef& QueueRunnerDef::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_QueueRunnerDef_tensorflow_2fcore_2fprotobuf_2fqueue_5frunner_2eproto.base);
  return *internal_default_instance();
}


void QueueRunnerDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.QueueRunnerDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  enqueue_op_name_.Clear();
  queue_closed_exception_types_.Clear();
  queue_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  close_op_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  cancel_op_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* QueueRunnerDef::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // string queue_name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_queue_name(), ptr, ctx, "tensorflow.QueueRunnerDef.queue_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated string enqueue_op_name = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(add_enqueue_op_name(), ptr, ctx, "tensorflow.QueueRunnerDef.enqueue_op_name");
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<::PROTOBUF_NAMESPACE_ID::uint8>(ptr) == 18);
        } else goto handle_unusual;
        continue;
      // string close_op_name = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_close_op_name(), ptr, ctx, "tensorflow.QueueRunnerDef.close_op_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // string cancel_op_name = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 34)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_cancel_op_name(), ptr, ctx, "tensorflow.QueueRunnerDef.cancel_op_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 42)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::PackedEnumParser(mutable_queue_closed_exception_types(), ptr, ctx);
          CHK_(ptr);
        } else if (static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
          add_queue_closed_exception_types(static_cast<::tensorflow::error::Code>(val));
        } else goto handle_unusual;
        continue;
      default: {
      handle_unusual:
        if ((tag & 7) == 4 || tag == 0) {
          ctx->SetLastTag(tag);
          goto success;
        }
        ptr = UnknownFieldParse(tag, &_internal_metadata_, ptr, ctx);
        CHK_(ptr != nullptr);
        continue;
      }
    }  // switch
  }  // while
success:
  return ptr;
failure:
  ptr = nullptr;
  goto success;
#undef CHK_
}
#else  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
bool QueueRunnerDef::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.QueueRunnerDef)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string queue_name = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_queue_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->queue_name().data(), static_cast<int>(this->queue_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.QueueRunnerDef.queue_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated string enqueue_op_name = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->add_enqueue_op_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->enqueue_op_name(this->enqueue_op_name_size() - 1).data(),
            static_cast<int>(this->enqueue_op_name(this->enqueue_op_name_size() - 1).length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.QueueRunnerDef.enqueue_op_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string close_op_name = 3;
      case 3: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (26 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_close_op_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->close_op_name().data(), static_cast<int>(this->close_op_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.QueueRunnerDef.close_op_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string cancel_op_name = 4;
      case 4: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (34 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_cancel_op_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->cancel_op_name().data(), static_cast<int>(this->cancel_op_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.QueueRunnerDef.cancel_op_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
      case 5: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (42 & 0xFF)) {
          ::PROTOBUF_NAMESPACE_ID::uint32 length;
          DO_(input->ReadVarint32(&length));
          ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream::Limit limit = input->PushLimit(static_cast<int>(length));
          while (input->BytesUntilLimit() > 0) {
            int value = 0;
            DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   int, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
            add_queue_closed_exception_types(static_cast< ::tensorflow::error::Code >(value));
          }
          input->PopLimit(limit);
        } else if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (40 & 0xFF)) {
          int value = 0;
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   int, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          add_queue_closed_exception_types(static_cast< ::tensorflow::error::Code >(value));
        } else {
          goto handle_unusual;
        }
        break;
      }

      default: {
      handle_unusual:
        if (tag == 0) {
          goto success;
        }
        DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SkipField(
              input, tag, _internal_metadata_.mutable_unknown_fields()));
        break;
      }
    }
  }
success:
  // @@protoc_insertion_point(parse_success:tensorflow.QueueRunnerDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.QueueRunnerDef)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void QueueRunnerDef::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.QueueRunnerDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string queue_name = 1;
  if (this->queue_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->queue_name().data(), static_cast<int>(this->queue_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.queue_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->queue_name(), output);
  }

  // repeated string enqueue_op_name = 2;
  for (int i = 0, n = this->enqueue_op_name_size(); i < n; i++) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->enqueue_op_name(i).data(), static_cast<int>(this->enqueue_op_name(i).length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.enqueue_op_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteString(
      2, this->enqueue_op_name(i), output);
  }

  // string close_op_name = 3;
  if (this->close_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->close_op_name().data(), static_cast<int>(this->close_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.close_op_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      3, this->close_op_name(), output);
  }

  // string cancel_op_name = 4;
  if (this->cancel_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->cancel_op_name().data(), static_cast<int>(this->cancel_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.cancel_op_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      4, this->cancel_op_name(), output);
  }

  // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
  if (this->queue_closed_exception_types_size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteTag(
      5,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      output);
    output->WriteVarint32(_queue_closed_exception_types_cached_byte_size_.load(
        std::memory_order_relaxed));
  }
  for (int i = 0, n = this->queue_closed_exception_types_size(); i < n; i++) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumNoTag(
      this->queue_closed_exception_types(i), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.QueueRunnerDef)
}

::PROTOBUF_NAMESPACE_ID::uint8* QueueRunnerDef::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.QueueRunnerDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string queue_name = 1;
  if (this->queue_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->queue_name().data(), static_cast<int>(this->queue_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.queue_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        1, this->queue_name(), target);
  }

  // repeated string enqueue_op_name = 2;
  for (int i = 0, n = this->enqueue_op_name_size(); i < n; i++) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->enqueue_op_name(i).data(), static_cast<int>(this->enqueue_op_name(i).length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.enqueue_op_name");
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      WriteStringToArray(2, this->enqueue_op_name(i), target);
  }

  // string close_op_name = 3;
  if (this->close_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->close_op_name().data(), static_cast<int>(this->close_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.close_op_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        3, this->close_op_name(), target);
  }

  // string cancel_op_name = 4;
  if (this->cancel_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->cancel_op_name().data(), static_cast<int>(this->cancel_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.QueueRunnerDef.cancel_op_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        4, this->cancel_op_name(), target);
  }

  // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
  if (this->queue_closed_exception_types_size() > 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteTagToArray(
      5,
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WIRETYPE_LENGTH_DELIMITED,
      target);
    target = ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream::WriteVarint32ToArray(      _queue_closed_exception_types_cached_byte_size_.load(std::memory_order_relaxed),
        target);
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumNoTagToArray(
      this->queue_closed_exception_types_, target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.QueueRunnerDef)
  return target;
}

size_t QueueRunnerDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.QueueRunnerDef)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated string enqueue_op_name = 2;
  total_size += 1 *
      ::PROTOBUF_NAMESPACE_ID::internal::FromIntSize(this->enqueue_op_name_size());
  for (int i = 0, n = this->enqueue_op_name_size(); i < n; i++) {
    total_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
      this->enqueue_op_name(i));
  }

  // repeated .tensorflow.error.Code queue_closed_exception_types = 5;
  {
    size_t data_size = 0;
    unsigned int count = static_cast<unsigned int>(this->queue_closed_exception_types_size());for (unsigned int i = 0; i < count; i++) {
      data_size += ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(
        this->queue_closed_exception_types(static_cast<int>(i)));
    }
    if (data_size > 0) {
      total_size += 1 +
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
            static_cast<::PROTOBUF_NAMESPACE_ID::int32>(data_size));
    }
    int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(data_size);
    _queue_closed_exception_types_cached_byte_size_.store(cached_size,
                                    std::memory_order_relaxed);
    total_size += data_size;
  }

  // string queue_name = 1;
  if (this->queue_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->queue_name());
  }

  // string close_op_name = 3;
  if (this->close_op_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->close_op_name());
  }

  // string cancel_op_name = 4;
  if (this->cancel_op_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->cancel_op_name());
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void QueueRunnerDef::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.QueueRunnerDef)
  GOOGLE_DCHECK_NE(&from, this);
  const QueueRunnerDef* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<QueueRunnerDef>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.QueueRunnerDef)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.QueueRunnerDef)
    MergeFrom(*source);
  }
}

void QueueRunnerDef::MergeFrom(const QueueRunnerDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.QueueRunnerDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  enqueue_op_name_.MergeFrom(from.enqueue_op_name_);
  queue_closed_exception_types_.MergeFrom(from.queue_closed_exception_types_);
  if (from.queue_name().size() > 0) {
    set_queue_name(from.queue_name());
  }
  if (from.close_op_name().size() > 0) {
    set_close_op_name(from.close_op_name());
  }
  if (from.cancel_op_name().size() > 0) {
    set_cancel_op_name(from.cancel_op_name());
  }
}

void QueueRunnerDef::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.QueueRunnerDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void QueueRunnerDef::CopyFrom(const QueueRunnerDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.QueueRunnerDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool QueueRunnerDef::IsInitialized() const {
  return true;
}

void QueueRunnerDef::Swap(QueueRunnerDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    QueueRunnerDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void QueueRunnerDef::UnsafeArenaSwap(QueueRunnerDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void QueueRunnerDef::InternalSwap(QueueRunnerDef* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  enqueue_op_name_.InternalSwap(CastToBase(&other->enqueue_op_name_));
  queue_closed_exception_types_.InternalSwap(&other->queue_closed_exception_types_);
  queue_name_.Swap(&other->queue_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  close_op_name_.Swap(&other->close_op_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  cancel_op_name_.Swap(&other->cancel_op_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
}

::PROTOBUF_NAMESPACE_ID::Metadata QueueRunnerDef::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::QueueRunnerDef* Arena::CreateMaybeMessage< ::tensorflow::QueueRunnerDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::QueueRunnerDef >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>