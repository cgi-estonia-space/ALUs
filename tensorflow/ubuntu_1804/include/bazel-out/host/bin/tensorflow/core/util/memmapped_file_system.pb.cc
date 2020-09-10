// Generated by the protocol buffer compiler.  DO NOT EDIT!
// source: tensorflow/core/util/memmapped_file_system.proto

#include "tensorflow/core/util/memmapped_file_system.pb.h"

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
extern PROTOBUF_INTERNAL_EXPORT_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto ::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto;
namespace tensorflow {
class MemmappedFileSystemDirectoryElementDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<MemmappedFileSystemDirectoryElement> _instance;
} _MemmappedFileSystemDirectoryElement_default_instance_;
class MemmappedFileSystemDirectoryDefaultTypeInternal {
 public:
  ::PROTOBUF_NAMESPACE_ID::internal::ExplicitlyConstructed<MemmappedFileSystemDirectory> _instance;
} _MemmappedFileSystemDirectory_default_instance_;
}  // namespace tensorflow
static void InitDefaultsscc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_MemmappedFileSystemDirectory_default_instance_;
    new (ptr) ::tensorflow::MemmappedFileSystemDirectory();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::MemmappedFileSystemDirectory::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<1> scc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 1, InitDefaultsscc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto}, {
      &scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base,}};

static void InitDefaultsscc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto() {
  GOOGLE_PROTOBUF_VERIFY_VERSION;

  {
    void* ptr = &::tensorflow::_MemmappedFileSystemDirectoryElement_default_instance_;
    new (ptr) ::tensorflow::MemmappedFileSystemDirectoryElement();
    ::PROTOBUF_NAMESPACE_ID::internal::OnShutdownDestroyMessage(ptr);
  }
  ::tensorflow::MemmappedFileSystemDirectoryElement::InitAsDefaultInstance();
}

::PROTOBUF_NAMESPACE_ID::internal::SCCInfo<0> scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto =
    {{ATOMIC_VAR_INIT(::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase::kUninitialized), 0, InitDefaultsscc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto}, {}};

static ::PROTOBUF_NAMESPACE_ID::Metadata file_level_metadata_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto[2];
static constexpr ::PROTOBUF_NAMESPACE_ID::EnumDescriptor const** file_level_enum_descriptors_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto = nullptr;
static constexpr ::PROTOBUF_NAMESPACE_ID::ServiceDescriptor const** file_level_service_descriptors_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto = nullptr;

const ::PROTOBUF_NAMESPACE_ID::uint32 TableStruct_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::offsets[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, offset_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, name_),
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectoryElement, length_),
  ~0u,  // no _has_bits_
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectory, _internal_metadata_),
  ~0u,  // no _extensions_
  ~0u,  // no _oneof_case_
  ~0u,  // no _weak_field_map_
  PROTOBUF_FIELD_OFFSET(::tensorflow::MemmappedFileSystemDirectory, element_),
};
static const ::PROTOBUF_NAMESPACE_ID::internal::MigrationSchema schemas[] PROTOBUF_SECTION_VARIABLE(protodesc_cold) = {
  { 0, -1, sizeof(::tensorflow::MemmappedFileSystemDirectoryElement)},
  { 8, -1, sizeof(::tensorflow::MemmappedFileSystemDirectory)},
};

static ::PROTOBUF_NAMESPACE_ID::Message const * const file_default_instances[] = {
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_MemmappedFileSystemDirectoryElement_default_instance_),
  reinterpret_cast<const ::PROTOBUF_NAMESPACE_ID::Message*>(&::tensorflow::_MemmappedFileSystemDirectory_default_instance_),
};

const char descriptor_table_protodef_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto[] =
  "\n0tensorflow/core/util/memmapped_file_sy"
  "stem.proto\022\ntensorflow\"S\n#MemmappedFileS"
  "ystemDirectoryElement\022\016\n\006offset\030\001 \001(\004\022\014\n"
  "\004name\030\002 \001(\t\022\016\n\006length\030\003 \001(\004\"`\n\034Memmapped"
  "FileSystemDirectory\022@\n\007element\030\001 \003(\0132/.t"
  "ensorflow.MemmappedFileSystemDirectoryEl"
  "ementB\003\370\001\001b\006proto3"
  ;
static const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable*const descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_deps[1] = {
};
static ::PROTOBUF_NAMESPACE_ID::internal::SCCInfoBase*const descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_sccs[2] = {
  &scc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base,
  &scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base,
};
static ::PROTOBUF_NAMESPACE_ID::internal::once_flag descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_once;
static bool descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_initialized = false;
const ::PROTOBUF_NAMESPACE_ID::internal::DescriptorTable descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto = {
  &descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_initialized, descriptor_table_protodef_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto, "tensorflow/core/util/memmapped_file_system.proto", 258,
  &descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_once, descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_sccs, descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto_deps, 2, 0,
  schemas, file_default_instances, TableStruct_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto::offsets,
  file_level_metadata_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto, 2, file_level_enum_descriptors_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto, file_level_service_descriptors_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto,
};

// Force running AddDescriptors() at dynamic initialization time.
static bool dynamic_init_dummy_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto = (  ::PROTOBUF_NAMESPACE_ID::internal::AddDescriptors(&descriptor_table_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto), true);
namespace tensorflow {

// ===================================================================

void MemmappedFileSystemDirectoryElement::InitAsDefaultInstance() {
}
class MemmappedFileSystemDirectoryElement::HasBitSetters {
 public:
};

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MemmappedFileSystemDirectoryElement::kOffsetFieldNumber;
const int MemmappedFileSystemDirectoryElement::kNameFieldNumber;
const int MemmappedFileSystemDirectoryElement::kLengthFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}
MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}
MemmappedFileSystemDirectoryElement::MemmappedFileSystemDirectoryElement(const MemmappedFileSystemDirectoryElement& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  if (from.name().size() > 0) {
    name_.Set(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), from.name(),
      GetArenaNoVirtual());
  }
  ::memcpy(&offset_, &from.offset_,
    static_cast<size_t>(reinterpret_cast<char*>(&length_) -
    reinterpret_cast<char*>(&offset_)) + sizeof(length_));
  // @@protoc_insertion_point(copy_constructor:tensorflow.MemmappedFileSystemDirectoryElement)
}

void MemmappedFileSystemDirectoryElement::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base);
  name_.UnsafeSetDefault(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
  ::memset(&offset_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&length_) -
      reinterpret_cast<char*>(&offset_)) + sizeof(length_));
}

MemmappedFileSystemDirectoryElement::~MemmappedFileSystemDirectoryElement() {
  // @@protoc_insertion_point(destructor:tensorflow.MemmappedFileSystemDirectoryElement)
  SharedDtor();
}

void MemmappedFileSystemDirectoryElement::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
  name_.DestroyNoArena(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited());
}

void MemmappedFileSystemDirectoryElement::ArenaDtor(void* object) {
  MemmappedFileSystemDirectoryElement* _this = reinterpret_cast< MemmappedFileSystemDirectoryElement* >(object);
  (void)_this;
}
void MemmappedFileSystemDirectoryElement::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void MemmappedFileSystemDirectoryElement::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const MemmappedFileSystemDirectoryElement& MemmappedFileSystemDirectoryElement::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_MemmappedFileSystemDirectoryElement_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base);
  return *internal_default_instance();
}


void MemmappedFileSystemDirectoryElement::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  name_.ClearToEmpty(&::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(), GetArenaNoVirtual());
  ::memset(&offset_, 0, static_cast<size_t>(
      reinterpret_cast<char*>(&length_) -
      reinterpret_cast<char*>(&offset_)) + sizeof(length_));
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* MemmappedFileSystemDirectoryElement::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // uint64 offset = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 8)) {
          offset_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // string name = 2;
      case 2:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 18)) {
          ptr = ::PROTOBUF_NAMESPACE_ID::internal::InlineGreedyStringParserUTF8(mutable_name(), ptr, ctx, "tensorflow.MemmappedFileSystemDirectoryElement.name");
          CHK_(ptr);
        } else goto handle_unusual;
        continue;
      // uint64 length = 3;
      case 3:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 24)) {
          length_ = ::PROTOBUF_NAMESPACE_ID::internal::ReadVarint(&ptr);
          CHK_(ptr);
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
bool MemmappedFileSystemDirectoryElement::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.MemmappedFileSystemDirectoryElement)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // uint64 offset = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (8 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::uint64, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_UINT64>(
                 input, &offset_)));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // string name = 2;
      case 2: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (18 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadString(
                input, this->mutable_name()));
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
            this->name().data(), static_cast<int>(this->name().length()),
            ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::PARSE,
            "tensorflow.MemmappedFileSystemDirectoryElement.name"));
        } else {
          goto handle_unusual;
        }
        break;
      }

      // uint64 length = 3;
      case 3: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (24 & 0xFF)) {

          DO_((::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadPrimitive<
                   ::PROTOBUF_NAMESPACE_ID::uint64, ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::TYPE_UINT64>(
                 input, &length_)));
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
  // @@protoc_insertion_point(parse_success:tensorflow.MemmappedFileSystemDirectoryElement)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.MemmappedFileSystemDirectoryElement)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void MemmappedFileSystemDirectoryElement::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 offset = 1;
  if (this->offset() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64(1, this->offset(), output);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.MemmappedFileSystemDirectoryElement.name");
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringMaybeAliased(
      2, this->name(), output);
  }

  // uint64 length = 3;
  if (this->length() != 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64(3, this->length(), output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.MemmappedFileSystemDirectoryElement)
}

::PROTOBUF_NAMESPACE_ID::uint8* MemmappedFileSystemDirectoryElement::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.MemmappedFileSystemDirectoryElement)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // uint64 offset = 1;
  if (this->offset() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(1, this->offset(), target);
  }

  // string name = 2;
  if (this->name().size() > 0) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::VerifyUtf8String(
      this->name().data(), static_cast<int>(this->name().length()),
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::SERIALIZE,
      "tensorflow.MemmappedFileSystemDirectoryElement.name");
    target =
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteStringToArray(
        2, this->name(), target);
  }

  // uint64 length = 3;
  if (this->length() != 0) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteUInt64ToArray(3, this->length(), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.MemmappedFileSystemDirectoryElement)
  return target;
}

size_t MemmappedFileSystemDirectoryElement::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.MemmappedFileSystemDirectoryElement)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // string name = 2;
  if (this->name().size() > 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::StringSize(
        this->name());
  }

  // uint64 offset = 1;
  if (this->offset() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->offset());
  }

  // uint64 length = 3;
  if (this->length() != 0) {
    total_size += 1 +
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::UInt64Size(
        this->length());
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void MemmappedFileSystemDirectoryElement::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  GOOGLE_DCHECK_NE(&from, this);
  const MemmappedFileSystemDirectoryElement* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<MemmappedFileSystemDirectoryElement>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.MemmappedFileSystemDirectoryElement)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.MemmappedFileSystemDirectoryElement)
    MergeFrom(*source);
  }
}

void MemmappedFileSystemDirectoryElement::MergeFrom(const MemmappedFileSystemDirectoryElement& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  if (from.name().size() > 0) {
    set_name(from.name());
  }
  if (from.offset() != 0) {
    set_offset(from.offset());
  }
  if (from.length() != 0) {
    set_length(from.length());
  }
}

void MemmappedFileSystemDirectoryElement::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MemmappedFileSystemDirectoryElement::CopyFrom(const MemmappedFileSystemDirectoryElement& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.MemmappedFileSystemDirectoryElement)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MemmappedFileSystemDirectoryElement::IsInitialized() const {
  return true;
}

void MemmappedFileSystemDirectoryElement::Swap(MemmappedFileSystemDirectoryElement* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    MemmappedFileSystemDirectoryElement* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void MemmappedFileSystemDirectoryElement::UnsafeArenaSwap(MemmappedFileSystemDirectoryElement* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void MemmappedFileSystemDirectoryElement::InternalSwap(MemmappedFileSystemDirectoryElement* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  name_.Swap(&other->name_, &::PROTOBUF_NAMESPACE_ID::internal::GetEmptyStringAlreadyInited(),
    GetArenaNoVirtual());
  swap(offset_, other->offset_);
  swap(length_, other->length_);
}

::PROTOBUF_NAMESPACE_ID::Metadata MemmappedFileSystemDirectoryElement::GetMetadata() const {
  return GetMetadataStatic();
}


// ===================================================================

void MemmappedFileSystemDirectory::InitAsDefaultInstance() {
}
class MemmappedFileSystemDirectory::HasBitSetters {
 public:
};

#if !defined(_MSC_VER) || _MSC_VER >= 1900
const int MemmappedFileSystemDirectory::kElementFieldNumber;
#endif  // !defined(_MSC_VER) || _MSC_VER >= 1900

MemmappedFileSystemDirectory::MemmappedFileSystemDirectory()
  : ::PROTOBUF_NAMESPACE_ID::Message(), _internal_metadata_(nullptr) {
  SharedCtor();
  // @@protoc_insertion_point(constructor:tensorflow.MemmappedFileSystemDirectory)
}
MemmappedFileSystemDirectory::MemmappedFileSystemDirectory(::PROTOBUF_NAMESPACE_ID::Arena* arena)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
  _internal_metadata_(arena),
  element_(arena) {
  SharedCtor();
  RegisterArenaDtor(arena);
  // @@protoc_insertion_point(arena_constructor:tensorflow.MemmappedFileSystemDirectory)
}
MemmappedFileSystemDirectory::MemmappedFileSystemDirectory(const MemmappedFileSystemDirectory& from)
  : ::PROTOBUF_NAMESPACE_ID::Message(),
      _internal_metadata_(nullptr),
      element_(from.element_) {
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  // @@protoc_insertion_point(copy_constructor:tensorflow.MemmappedFileSystemDirectory)
}

void MemmappedFileSystemDirectory::SharedCtor() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&scc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base);
}

MemmappedFileSystemDirectory::~MemmappedFileSystemDirectory() {
  // @@protoc_insertion_point(destructor:tensorflow.MemmappedFileSystemDirectory)
  SharedDtor();
}

void MemmappedFileSystemDirectory::SharedDtor() {
  GOOGLE_DCHECK(GetArenaNoVirtual() == nullptr);
}

void MemmappedFileSystemDirectory::ArenaDtor(void* object) {
  MemmappedFileSystemDirectory* _this = reinterpret_cast< MemmappedFileSystemDirectory* >(object);
  (void)_this;
}
void MemmappedFileSystemDirectory::RegisterArenaDtor(::PROTOBUF_NAMESPACE_ID::Arena*) {
}
void MemmappedFileSystemDirectory::SetCachedSize(int size) const {
  _cached_size_.Set(size);
}
const MemmappedFileSystemDirectory& MemmappedFileSystemDirectory::default_instance() {
  ::PROTOBUF_NAMESPACE_ID::internal::InitSCC(&::scc_info_MemmappedFileSystemDirectory_tensorflow_2fcore_2futil_2fmemmapped_5ffile_5fsystem_2eproto.base);
  return *internal_default_instance();
}


void MemmappedFileSystemDirectory::Clear() {
// @@protoc_insertion_point(message_clear_start:tensorflow.MemmappedFileSystemDirectory)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  element_.Clear();
  _internal_metadata_.Clear();
}

#if GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER
const char* MemmappedFileSystemDirectory::_InternalParse(const char* ptr, ::PROTOBUF_NAMESPACE_ID::internal::ParseContext* ctx) {
#define CHK_(x) if (PROTOBUF_PREDICT_FALSE(!(x))) goto failure
  ::PROTOBUF_NAMESPACE_ID::Arena* arena = GetArenaNoVirtual(); (void)arena;
  while (!ctx->Done(&ptr)) {
    ::PROTOBUF_NAMESPACE_ID::uint32 tag;
    ptr = ::PROTOBUF_NAMESPACE_ID::internal::ReadTag(ptr, &tag);
    CHK_(ptr);
    switch (tag >> 3) {
      // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
      case 1:
        if (PROTOBUF_PREDICT_TRUE(static_cast<::PROTOBUF_NAMESPACE_ID::uint8>(tag) == 10)) {
          ptr -= 1;
          do {
            ptr += 1;
            ptr = ctx->ParseMessage(add_element(), ptr);
            CHK_(ptr);
            if (!ctx->DataAvailable(ptr)) break;
          } while (::PROTOBUF_NAMESPACE_ID::internal::UnalignedLoad<::PROTOBUF_NAMESPACE_ID::uint8>(ptr) == 10);
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
bool MemmappedFileSystemDirectory::MergePartialFromCodedStream(
    ::PROTOBUF_NAMESPACE_ID::io::CodedInputStream* input) {
#define DO_(EXPRESSION) if (!PROTOBUF_PREDICT_TRUE(EXPRESSION)) goto failure
  ::PROTOBUF_NAMESPACE_ID::uint32 tag;
  // @@protoc_insertion_point(parse_start:tensorflow.MemmappedFileSystemDirectory)
  for (;;) {
    ::std::pair<::PROTOBUF_NAMESPACE_ID::uint32, bool> p = input->ReadTagWithCutoffNoLastTag(127u);
    tag = p.first;
    if (!p.second) goto handle_unusual;
    switch (::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::GetTagFieldNumber(tag)) {
      // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
      case 1: {
        if (static_cast< ::PROTOBUF_NAMESPACE_ID::uint8>(tag) == (10 & 0xFF)) {
          DO_(::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::ReadMessage(
                input, add_element()));
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
  // @@protoc_insertion_point(parse_success:tensorflow.MemmappedFileSystemDirectory)
  return true;
failure:
  // @@protoc_insertion_point(parse_failure:tensorflow.MemmappedFileSystemDirectory)
  return false;
#undef DO_
}
#endif  // GOOGLE_PROTOBUF_ENABLE_EXPERIMENTAL_PARSER

void MemmappedFileSystemDirectory::SerializeWithCachedSizes(
    ::PROTOBUF_NAMESPACE_ID::io::CodedOutputStream* output) const {
  // @@protoc_insertion_point(serialize_start:tensorflow.MemmappedFileSystemDirectory)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->element_size()); i < n; i++) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::WriteMessageMaybeToArray(
      1,
      this->element(static_cast<int>(i)),
      output);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFields(
        _internal_metadata_.unknown_fields(), output);
  }
  // @@protoc_insertion_point(serialize_end:tensorflow.MemmappedFileSystemDirectory)
}

::PROTOBUF_NAMESPACE_ID::uint8* MemmappedFileSystemDirectory::InternalSerializeWithCachedSizesToArray(
    ::PROTOBUF_NAMESPACE_ID::uint8* target) const {
  // @@protoc_insertion_point(serialize_to_array_start:tensorflow.MemmappedFileSystemDirectory)
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  for (unsigned int i = 0,
      n = static_cast<unsigned int>(this->element_size()); i < n; i++) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::
      InternalWriteMessageToArray(
        1, this->element(static_cast<int>(i)), target);
  }

  if (_internal_metadata_.have_unknown_fields()) {
    target = ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::SerializeUnknownFieldsToArray(
        _internal_metadata_.unknown_fields(), target);
  }
  // @@protoc_insertion_point(serialize_to_array_end:tensorflow.MemmappedFileSystemDirectory)
  return target;
}

size_t MemmappedFileSystemDirectory::ByteSizeLong() const {
// @@protoc_insertion_point(message_byte_size_start:tensorflow.MemmappedFileSystemDirectory)
  size_t total_size = 0;

  if (_internal_metadata_.have_unknown_fields()) {
    total_size +=
      ::PROTOBUF_NAMESPACE_ID::internal::WireFormat::ComputeUnknownFieldsSize(
        _internal_metadata_.unknown_fields());
  }
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  // Prevent compiler warnings about cached_has_bits being unused
  (void) cached_has_bits;

  // repeated .tensorflow.MemmappedFileSystemDirectoryElement element = 1;
  {
    unsigned int count = static_cast<unsigned int>(this->element_size());
    total_size += 1UL * count;
    for (unsigned int i = 0; i < count; i++) {
      total_size +=
        ::PROTOBUF_NAMESPACE_ID::internal::WireFormatLite::MessageSize(
          this->element(static_cast<int>(i)));
    }
  }

  int cached_size = ::PROTOBUF_NAMESPACE_ID::internal::ToCachedSize(total_size);
  SetCachedSize(cached_size);
  return total_size;
}

void MemmappedFileSystemDirectory::MergeFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_merge_from_start:tensorflow.MemmappedFileSystemDirectory)
  GOOGLE_DCHECK_NE(&from, this);
  const MemmappedFileSystemDirectory* source =
      ::PROTOBUF_NAMESPACE_ID::DynamicCastToGenerated<MemmappedFileSystemDirectory>(
          &from);
  if (source == nullptr) {
  // @@protoc_insertion_point(generalized_merge_from_cast_fail:tensorflow.MemmappedFileSystemDirectory)
    ::PROTOBUF_NAMESPACE_ID::internal::ReflectionOps::Merge(from, this);
  } else {
  // @@protoc_insertion_point(generalized_merge_from_cast_success:tensorflow.MemmappedFileSystemDirectory)
    MergeFrom(*source);
  }
}

void MemmappedFileSystemDirectory::MergeFrom(const MemmappedFileSystemDirectory& from) {
// @@protoc_insertion_point(class_specific_merge_from_start:tensorflow.MemmappedFileSystemDirectory)
  GOOGLE_DCHECK_NE(&from, this);
  _internal_metadata_.MergeFrom(from._internal_metadata_);
  ::PROTOBUF_NAMESPACE_ID::uint32 cached_has_bits = 0;
  (void) cached_has_bits;

  element_.MergeFrom(from.element_);
}

void MemmappedFileSystemDirectory::CopyFrom(const ::PROTOBUF_NAMESPACE_ID::Message& from) {
// @@protoc_insertion_point(generalized_copy_from_start:tensorflow.MemmappedFileSystemDirectory)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

void MemmappedFileSystemDirectory::CopyFrom(const MemmappedFileSystemDirectory& from) {
// @@protoc_insertion_point(class_specific_copy_from_start:tensorflow.MemmappedFileSystemDirectory)
  if (&from == this) return;
  Clear();
  MergeFrom(from);
}

bool MemmappedFileSystemDirectory::IsInitialized() const {
  return true;
}

void MemmappedFileSystemDirectory::Swap(MemmappedFileSystemDirectory* other) {
  if (other == this) return;
  if (GetArenaNoVirtual() == other->GetArenaNoVirtual()) {
    InternalSwap(other);
  } else {
    MemmappedFileSystemDirectory* temp = New(GetArenaNoVirtual());
    temp->MergeFrom(*other);
    other->CopyFrom(*this);
    InternalSwap(temp);
    if (GetArenaNoVirtual() == nullptr) {
      delete temp;
    }
  }
}
void MemmappedFileSystemDirectory::UnsafeArenaSwap(MemmappedFileSystemDirectory* other) {
  if (other == this) return;
  GOOGLE_DCHECK(GetArenaNoVirtual() == other->GetArenaNoVirtual());
  InternalSwap(other);
}
void MemmappedFileSystemDirectory::InternalSwap(MemmappedFileSystemDirectory* other) {
  using std::swap;
  _internal_metadata_.Swap(&other->_internal_metadata_);
  CastToBase(&element_)->InternalSwap(CastToBase(&other->element_));
}

::PROTOBUF_NAMESPACE_ID::Metadata MemmappedFileSystemDirectory::GetMetadata() const {
  return GetMetadataStatic();
}


// @@protoc_insertion_point(namespace_scope)
}  // namespace tensorflow
PROTOBUF_NAMESPACE_OPEN
template<> PROTOBUF_NOINLINE ::tensorflow::MemmappedFileSystemDirectoryElement* Arena::CreateMaybeMessage< ::tensorflow::MemmappedFileSystemDirectoryElement >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::MemmappedFileSystemDirectoryElement >(arena);
}
template<> PROTOBUF_NOINLINE ::tensorflow::MemmappedFileSystemDirectory* Arena::CreateMaybeMessage< ::tensorflow::MemmappedFileSystemDirectory >(Arena* arena) {
  return Arena::CreateMessageInternal< ::tensorflow::MemmappedFileSystemDirectory >(arena);
}
PROTOBUF_NAMESPACE_CLOSE

// @@protoc_insertion_point(global_scope)
#include <google/protobuf/port_undef.inc>