// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/protobuf/saver.proto

#include "tensorflow/core/protobuf/saver.pb.h"

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
class SaverDefDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<SaverDef> _instance;
} _SaverDef_default_instance_;
}  // namespace tensorflow
static void InitDefaultsscc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_SaverDef_default_instance_;
    new (ptr) ::tensorflow::SaverDef();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::SaverDef::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsscc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto[1];
static const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto[1];
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, filename_tensor_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, save_tensor_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, restore_op_name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, max_to_keep_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, sharded_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, keep_checkpoint_every_n_hours_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::SaverDef, version_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::SaverDef)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_SaverDef_default_instance_),
};

const char descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto[] =
  "\n$tensorflow/core/protobuf/saver.proto\022\n"
  "tensorflow\"\236\002\n\010SaverDef\022\034\n\024filename_tens"
  "or_name\030\001 \001(\t\022\030\n\020save_tensor_name\030\002 \001(\t\022"
  "\027\n\017restore_op_name\030\003 \001(\t\022\023\n\013max_to_keep\030"
  "\004 \001(\005\022\017\n\007sharded\030\005 \001(\010\022%\n\035keep_checkpoin"
  "t_every_n_hours\030\006 \001(\002\022=\n\007version\030\007 \001(\0162,"
  ".tensorflow.SaverDef.CheckpointFormatVer"
  "sion\"5\n\027CheckpointFormatVersion\022\n\n\006LEGAC"
  "Y\020\000\022\006\n\002V1\020\001\022\006\n\002V2\020\002Bq\n\023org.tensorflow.ut"
  "ilB\013SaverProtosP\001ZHgithub.com/tensorflow"
  "/tensorflow/tensorflow/go/core/core_prot"
  "os_go_proto\370\001\001b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_sccs[1] = {
  &scc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_once;
static bool descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto = {
  &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_initialized, descriptor_table_protodef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto, "tensorflow/core/protobuf/saver.proto", 462,
  &descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_once, descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_sccs, descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto_deps, 1, 0,
  schemas, file_default_instances, TableStruct_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto::offsets,
  file_level_metadata_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto, 1, file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto, file_level_service_descriptors_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto), true);
namespace tensorflow {
const ::PROTOBUF_NAMESPACE_ID::EnumDescriptor* SaverDef_CheckpointFormatVersion_descriptor() {
  ::PROTOBUF_NAMESPACE_ID::internal::AssignDescriptors(&descriptor_table_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto);
  return file_level_enum_descriptors_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto[0];
}
bool SaverDef_CheckpointFormatVersion_IsValid(int value) {
  switch (value) {
    case 0:
    case 1:
    case 2:
      return true;
    default:
      return false;
  }
}

#if (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)
constexpr SaverDef_CheckpointFormatVersion SaverDef::LEGACY;
constexpr SaverDef_CheckpointFormatVersion SaverDef::V1;
constexpr SaverDef_CheckpointFormatVersion SaverDef::V2;
constexpr SaverDef_CheckpointFormatVersion SaverDef::CheckpointFormatVersion_MIN;
constexpr SaverDef_CheckpointFormatVersion SaverDef::CheckpointFormatVersion_MAX;
constexpr int SaverDef::CheckpointFormatVersion_ARRAYSIZE;
#endif  // (__cplusplus < 201703) && (!defined(_MSC_VER) || _MSC_VER >= 1900)

// ===================================================================

void SaverDef::InitAsDefaultInstance() {
}
class SaverDef::HasBitSetters {
 public:
};

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int SaverDef::kFilenameTensorNameFieldNumber;
const int SaverDef::kSaveTensorNameFieldNumber;
const int SaverDef::kRestoreOpNameFieldNumber;
const int SaverDef::kMaxToKeepFieldNumber;
const int SaverDef::kShardedFieldNumber;
const int SaverDef::kKeepCheckpointEveryNHoursFieldNumber;
const int SaverDef::kVersionFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

SaverDef::SaverDef()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.SaverDef)
}
SaverDef::SaverDef(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.SaverDef)
}
SaverDef::SaverDef(const SaverDef& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  filename_tensor_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.filename_tensor_name().size() > 0) {
    filename_tensor_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.filename_tensor_name(),
      GetArenaNoVirtual());
  }
  save_tensor_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.save_tensor_name().size() > 0) {
    save_tensor_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.save_tensor_name(),
      GetArenaNoVirtual());
  }
  restore_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.restore_op_name().size() > 0) {
    restore_op_name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.restore_op_name(),
      GetArenaNoVirtual());
  }
  ::memcpy(&max_to_keep_, &from.max_to_keep_,
    static_cast<size_t>(reinterpret_cast<char*>(&version_) -
    reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
  // @@protoc_insertion_point(copy_constructor:tensorflow.SaverDef)
}

void SaverDef::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto.base);
  filename_tensor_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  save_tensor_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  restore_op_name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(&max_to_keep_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&version_) -
      reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
}

SaverDef::~SaverDef() {
  // @@protoc_insertion_point(destructor:tensorflow.SaverDef)
  SharedDtor();
}

void SaverDef::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  filename_tensor_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  save_tensor_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  restore_op_name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void SaverDef::ArenaDtor(void* object) {
  SaverDef* _this = reinterpret_cast< SaverDef* >(object);
  (void)_this;
}
void SaverDef::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void SaverDef::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const SaverDef& SaverDef::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_SaverDef_tensorflow_2fcore_2fprotobuf_2fsaver_2eproto.base);
  return *internal_default_instance();
}


void SaverDef::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.SaverDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  filename_tensor_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  save_tensor_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  restore_op_name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  ::memset(&max_to_keep_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&version_) -
      reinterpret_cast<char*>(&max_to_keep_)) + sizeof(version_));
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* SaverDef::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // string filename_tensor_name = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_filename_tensor_name(), ptr, ctx, "tensorflow.SaverDef.filename_tensor_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // string save_tensor_name = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_save_tensor_name(), ptr, ctx, "tensorflow.SaverDef.save_tensor_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // string restore_op_name = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 26)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_restore_op_name(), ptr, ctx, "tensorflow.SaverDef.restore_op_name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // int32 max_to_keep = 4;
      case 4:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 32)) {
          max_to_keep_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // bool sharded = 5;
      case 5:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 40)) {
          sharded_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // float keep_checkpoint_every_n_hours = 6;
      case 6:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 53)) {
          keep_checkpoint_every_n_hours_ = ::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<float>(ptr);
          ptr += sizeof(float);
        } else goto handle_unusual;
        continue;
      // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
      case 7:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 56)) {
          ::PROTOBUF_NAMESPACE_ID::uint64 val = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
          set_version(static_cast<::tensorflow::SaverDef_CheckpointFormatVersion>(val));
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
bool SaverDef::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.SaverDef)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // string filename_tensor_name = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_filename_tensor_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.filename_tensor_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string save_tensor_name = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_save_tensor_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.save_tensor_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string restore_op_name = 3;
      case 3: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (26 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_restore_op_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.SaverDef.restore_op_name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // int32 max_to_keep = 4;
      case 4: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (32 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::int32, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_INT32>(
                 input, &max_to_keep_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // bool sharded = 5;
      case 5: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (40 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   bool, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_BOOL>(
                 input, &sharded_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // float keep_checkpoint_every_n_hours = 6;
      case 6: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (53 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   float, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_FLOAT>(
                 input, &keep_checkpoint_every_n_hours_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
      case 7: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (56 & 0xFF)) {
          int value = 0;
          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   int, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_ENUM>(
                 input, &value)));
          set_version(static_cast< ::tensorflow::SaverDef_CheckpointFormatVersion >(value));
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
  // @@protoc_insertion_point(parse_success:tensorflow.SaverDef)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.SaverDef)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void SaverDef::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.SaverDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.filename_tensor_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      1, this->filename_tensor_name(), output);
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.save_tensor_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->save_tensor_name(), output);
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.restore_op_name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      3, this->restore_op_name(), output);
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32(4, this->max_to_keep(), output);
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBool(5, this->sharded(), output);
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (!(this->keep_checkpoint_every_n_hours() <= 0 && this->keep_checkpoint_every_n_hours() >= 0)) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloat(6, this->keep_checkpoint_every_n_hours(), output);
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnum(
      7, this->version(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.SaverDef)
}

::PROTOBUF_NAMESPACE_ID::uint8* SaverDef::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.SaverDef)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->filename_tensor_name().data(), static_cast<int>(this->filename_tensor_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.filename_tensor_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        1, this->filename_tensor_name(), target);
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->save_tensor_name().data(), static_cast<int>(this->save_tensor_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.save_tensor_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        2, this->save_tensor_name(), target);
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->restore_op_name().data(), static_cast<int>(this->restore_op_name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.SaverDef.restore_op_name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        3, this->restore_op_name(), target);
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteInt32ToArray(4, this->max_to_keep(), target);
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteBoolToArray(5, this->sharded(), target);
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (!(this->keep_checkpoint_every_n_hours() <= 0 && this->keep_checkpoint_every_n_hours() >= 0)) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteFloatToArray(6, this->keep_checkpoint_every_n_hours(), target);
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteEnumToArray(
      7, this->version(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.SaverDef)
  return target;
}

size_t SaverDef::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.SaverDef)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string filename_tensor_name = 1;
  if (this->filename_tensor_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->filename_tensor_name());
  }

  // string save_tensor_name = 2;
  if (this->save_tensor_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->save_tensor_name());
  }

  // string restore_op_name = 3;
  if (this->restore_op_name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->restore_op_name());
  }

  // int32 max_to_keep = 4;
  if (this->max_to_keep() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::Int32Size(
        this->max_to_keep());
  }

  // bool sharded = 5;
  if (this->sharded() != 0) {
    total_size += 1 + 1;
  }

  // float keep_checkpoint_every_n_hours = 6;
  if (!(this->keep_checkpoint_every_n_hours() <= 0 && this->keep_checkpoint_every_n_hours() >= 0)) {
    total_size += 1 + 4;
  }

  // .tensorflow.SaverDef.CheckpointFormatVersion version = 7;
  if (this->version() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::EnumSize(this->version());
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void SaverDef::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.SaverDef)
  GOOGLE_DCHECK_NE(&from, this);
  const SaverDef* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<SaverDef>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.SaverDef)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.SaverDef)
    MergeFrom(*source);
  }
}

void SaverDef::MergeFrom(const SaverDef& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.SaverDef)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.filename_tensor_name().size() > 0) {
    set_filename_tensor_name(from.filename_tensor_name());
  }
  if (from.save_tensor_name().size() > 0) {
    set_save_tensor_name(from.save_tensor_name());
  }
  if (from.restore_op_name().size() > 0) {
    set_restore_op_name(from.restore_op_name());
  }
  if (from.max_to_keep() != 0) {
    set_max_to_keep(from.max_to_keep());
  }
  if (from.sharded() != 0) {
    set_sharded(from.sharded());
  }
  if (!(from.keep_checkpoint_every_n_hours() <= 0 && from.keep_checkpoint_every_n_hours() >= 0)) {
    set_keep_checkpoint_every_n_hours(from.keep_checkpoint_every_n_hours());
  }
  if (from.version() != 0) {
    set_version(from.version());
  }
}

void SaverDef::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.SaverDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void SaverDef::CopyFrom(const SaverDef& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.SaverDef)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool SaverDef::IsInitialized() const {
  return true;
}

void SaverDef::Swap(SaverDef* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    SaverDef* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void SaverDef::UnsafeArenaSwap(SaverDef* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void SaverDef::InternalSwap(SaverDef* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  filename_tensor_name_.Swap(&other->filename_tensor_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  save_tensor_name_.Swap(&other->save_tensor_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  restore_op_name_.Swap(&other->restore_op_name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(max_to_keep_, other->max_to_keep_);
  swap(sharded_, other->sharded_);
  swap(keep_checkpoint_every_n_hours_, other->keep_checkpoint_every_n_hours_);
  swap(version_, other->version_);
}

::PROTOBUF_NAMESPACE_ID::Metadata SaverDef::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::SaverDef* Arena::CreateMaybeMessage< ::tensorflow::SaverDef >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::SaverDef >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>
