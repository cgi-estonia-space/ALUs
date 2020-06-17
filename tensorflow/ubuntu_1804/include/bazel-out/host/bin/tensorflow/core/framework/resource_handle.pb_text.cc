// GENERATED FILE - DO NOT MODIFY

#include <algorithm>

#include "tensorflow/core/framework/resource_handle.pb_text-impl.h"

using ::tensorflow::strings::Scanner;
using ::tensorflow::strings::StrCat;

namespace tensorflow {

string ProtoDebugString(
    const ::tensorflow::ResourceHandleProto_DtypeAndShape& msg) {
  string s;
  ::tensorflow::strings::ProtoTextOutput o(&s, false);
  internal::AppendProtoDebugString(&o, msg);
  o.CloseTopMessage();
  return s;
}

string ProtoShortDebugString(
    const ::tensorflow::ResourceHandleProto_DtypeAndShape& msg) {
  string s;
  ::tensorflow::strings::ProtoTextOutput o(&s, true);
  internal::AppendProtoDebugString(&o, msg);
  o.CloseTopMessage();
  return s;
}

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::ResourceHandleProto_DtypeAndShape& msg) {
  if (msg.dtype() != 0) {
    const char* enum_name = ::tensorflow::EnumName_DataType(msg.dtype());
    if (enum_name[0]) {
      o->AppendEnumName("dtype", enum_name);
    } else {
      o->AppendNumeric("dtype", msg.dtype());
    }
  }
  if (msg.has_shape()) {
    o->OpenNestedMessage("shape");
    ::tensorflow::internal::AppendProtoDebugString(o, msg.shape());
    o->CloseNestedMessage();
  }
}

}  // namespace internal

bool ProtoParseFromString(
    const string& s,
    ::tensorflow::ResourceHandleProto_DtypeAndShape* msg) {
  msg->Clear();
  Scanner scanner(s);
  if (!internal::ProtoParseFromScanner(&scanner, false, false, msg)) return false;
  scanner.Eos();
  return scanner.GetResult();
}

namespace internal {

bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::ResourceHandleProto_DtypeAndShape* msg) {
  std::vector<bool> has_seen(2, false);
  while(true) {
    ProtoSpaceAndComments(scanner);
    if (nested && (scanner->Peek() == (close_curly ? '}' : '>'))) {
      scanner->One(Scanner::ALL);
      ProtoSpaceAndComments(scanner);
      return true;
    }
    if (!nested && scanner->empty()) { return true; }
    scanner->RestartCapture()
        .Many(Scanner::LETTER_DIGIT_UNDERSCORE)
        .StopCapture();
    StringPiece identifier;
    if (!scanner->GetResult(nullptr, &identifier)) return false;
    bool parsed_colon = false;
    (void)parsed_colon;
    ProtoSpaceAndComments(scanner);
    if (scanner->Peek() == ':') {
      parsed_colon = true;
      scanner->One(Scanner::ALL);
      ProtoSpaceAndComments(scanner);
    }
    if (identifier == "dtype") {
      if (has_seen[0]) return false;
      has_seen[0] = true;
      StringPiece value;
      if (!parsed_colon || !scanner->RestartCapture().Many(Scanner::LETTER_DIGIT_DASH_UNDERSCORE).GetResult(nullptr, &value)) return false;
      if (value == "DT_INVALID") {
        msg->set_dtype(::tensorflow::DT_INVALID);
      } else if (value == "DT_FLOAT") {
        msg->set_dtype(::tensorflow::DT_FLOAT);
      } else if (value == "DT_DOUBLE") {
        msg->set_dtype(::tensorflow::DT_DOUBLE);
      } else if (value == "DT_INT32") {
        msg->set_dtype(::tensorflow::DT_INT32);
      } else if (value == "DT_UINT8") {
        msg->set_dtype(::tensorflow::DT_UINT8);
      } else if (value == "DT_INT16") {
        msg->set_dtype(::tensorflow::DT_INT16);
      } else if (value == "DT_INT8") {
        msg->set_dtype(::tensorflow::DT_INT8);
      } else if (value == "DT_STRING") {
        msg->set_dtype(::tensorflow::DT_STRING);
      } else if (value == "DT_COMPLEX64") {
        msg->set_dtype(::tensorflow::DT_COMPLEX64);
      } else if (value == "DT_INT64") {
        msg->set_dtype(::tensorflow::DT_INT64);
      } else if (value == "DT_BOOL") {
        msg->set_dtype(::tensorflow::DT_BOOL);
      } else if (value == "DT_QINT8") {
        msg->set_dtype(::tensorflow::DT_QINT8);
      } else if (value == "DT_QUINT8") {
        msg->set_dtype(::tensorflow::DT_QUINT8);
      } else if (value == "DT_QINT32") {
        msg->set_dtype(::tensorflow::DT_QINT32);
      } else if (value == "DT_BFLOAT16") {
        msg->set_dtype(::tensorflow::DT_BFLOAT16);
      } else if (value == "DT_QINT16") {
        msg->set_dtype(::tensorflow::DT_QINT16);
      } else if (value == "DT_QUINT16") {
        msg->set_dtype(::tensorflow::DT_QUINT16);
      } else if (value == "DT_UINT16") {
        msg->set_dtype(::tensorflow::DT_UINT16);
      } else if (value == "DT_COMPLEX128") {
        msg->set_dtype(::tensorflow::DT_COMPLEX128);
      } else if (value == "DT_HALF") {
        msg->set_dtype(::tensorflow::DT_HALF);
      } else if (value == "DT_RESOURCE") {
        msg->set_dtype(::tensorflow::DT_RESOURCE);
      } else if (value == "DT_VARIANT") {
        msg->set_dtype(::tensorflow::DT_VARIANT);
      } else if (value == "DT_UINT32") {
        msg->set_dtype(::tensorflow::DT_UINT32);
      } else if (value == "DT_UINT64") {
        msg->set_dtype(::tensorflow::DT_UINT64);
      } else if (value == "DT_FLOAT_REF") {
        msg->set_dtype(::tensorflow::DT_FLOAT_REF);
      } else if (value == "DT_DOUBLE_REF") {
        msg->set_dtype(::tensorflow::DT_DOUBLE_REF);
      } else if (value == "DT_INT32_REF") {
        msg->set_dtype(::tensorflow::DT_INT32_REF);
      } else if (value == "DT_UINT8_REF") {
        msg->set_dtype(::tensorflow::DT_UINT8_REF);
      } else if (value == "DT_INT16_REF") {
        msg->set_dtype(::tensorflow::DT_INT16_REF);
      } else if (value == "DT_INT8_REF") {
        msg->set_dtype(::tensorflow::DT_INT8_REF);
      } else if (value == "DT_STRING_REF") {
        msg->set_dtype(::tensorflow::DT_STRING_REF);
      } else if (value == "DT_COMPLEX64_REF") {
        msg->set_dtype(::tensorflow::DT_COMPLEX64_REF);
      } else if (value == "DT_INT64_REF") {
        msg->set_dtype(::tensorflow::DT_INT64_REF);
      } else if (value == "DT_BOOL_REF") {
        msg->set_dtype(::tensorflow::DT_BOOL_REF);
      } else if (value == "DT_QINT8_REF") {
        msg->set_dtype(::tensorflow::DT_QINT8_REF);
      } else if (value == "DT_QUINT8_REF") {
        msg->set_dtype(::tensorflow::DT_QUINT8_REF);
      } else if (value == "DT_QINT32_REF") {
        msg->set_dtype(::tensorflow::DT_QINT32_REF);
      } else if (value == "DT_BFLOAT16_REF") {
        msg->set_dtype(::tensorflow::DT_BFLOAT16_REF);
      } else if (value == "DT_QINT16_REF") {
        msg->set_dtype(::tensorflow::DT_QINT16_REF);
      } else if (value == "DT_QUINT16_REF") {
        msg->set_dtype(::tensorflow::DT_QUINT16_REF);
      } else if (value == "DT_UINT16_REF") {
        msg->set_dtype(::tensorflow::DT_UINT16_REF);
      } else if (value == "DT_COMPLEX128_REF") {
        msg->set_dtype(::tensorflow::DT_COMPLEX128_REF);
      } else if (value == "DT_HALF_REF") {
        msg->set_dtype(::tensorflow::DT_HALF_REF);
      } else if (value == "DT_RESOURCE_REF") {
        msg->set_dtype(::tensorflow::DT_RESOURCE_REF);
      } else if (value == "DT_VARIANT_REF") {
        msg->set_dtype(::tensorflow::DT_VARIANT_REF);
      } else if (value == "DT_UINT32_REF") {
        msg->set_dtype(::tensorflow::DT_UINT32_REF);
      } else if (value == "DT_UINT64_REF") {
        msg->set_dtype(::tensorflow::DT_UINT64_REF);
      } else {
        int32 int_value;
        if (strings::SafeStringToNumeric(value, &int_value)) {
          msg->set_dtype(static_cast<::tensorflow::DataType>(int_value));
        } else {
          return false;
        }
      }
    }
    else if (identifier == "shape") {
      if (has_seen[1]) return false;
      has_seen[1] = true;
      const char open_char = scanner->Peek();
      if (open_char != '{' && open_char != '<') return false;
      scanner->One(Scanner::ALL);
      ProtoSpaceAndComments(scanner);
      if (!::tensorflow::internal::ProtoParseFromScanner(
          scanner, true, open_char == '{', msg->mutable_shape())) return false;
    }
  }
}

}  // namespace internal

string ProtoDebugString(
    const ::tensorflow::ResourceHandleProto& msg) {
  string s;
  ::tensorflow::strings::ProtoTextOutput o(&s, false);
  internal::AppendProtoDebugString(&o, msg);
  o.CloseTopMessage();
  return s;
}

string ProtoShortDebugString(
    const ::tensorflow::ResourceHandleProto& msg) {
  string s;
  ::tensorflow::strings::ProtoTextOutput o(&s, true);
  internal::AppendProtoDebugString(&o, msg);
  o.CloseTopMessage();
  return s;
}

namespace internal {

void AppendProtoDebugString(
    ::tensorflow::strings::ProtoTextOutput* o,
    const ::tensorflow::ResourceHandleProto& msg) {
  o->AppendStringIfNotEmpty("device", ProtobufStringToString(msg.device()));
  o->AppendStringIfNotEmpty("container", ProtobufStringToString(msg.container()));
  o->AppendStringIfNotEmpty("name", ProtobufStringToString(msg.name()));
  o->AppendNumericIfNotZero("hash_code", msg.hash_code());
  o->AppendStringIfNotEmpty("maybe_type_name", ProtobufStringToString(msg.maybe_type_name()));
  for (int i = 0; i < msg.dtypes_and_shapes_size(); ++i) {
    o->OpenNestedMessage("dtypes_and_shapes");
    ::tensorflow::internal::AppendProtoDebugString(o, msg.dtypes_and_shapes(i));
    o->CloseNestedMessage();
  }
  for (int i = 0; i < msg.allowed_devices_size(); ++i) {
    o->AppendString("allowed_devices", ProtobufStringToString(msg.allowed_devices(i)));
  }
}

}  // namespace internal

bool ProtoParseFromString(
    const string& s,
    ::tensorflow::ResourceHandleProto* msg) {
  msg->Clear();
  Scanner scanner(s);
  if (!internal::ProtoParseFromScanner(&scanner, false, false, msg)) return false;
  scanner.Eos();
  return scanner.GetResult();
}

namespace internal {

bool ProtoParseFromScanner(
    ::tensorflow::strings::Scanner* scanner, bool nested, bool close_curly,
    ::tensorflow::ResourceHandleProto* msg) {
  std::vector<bool> has_seen(7, false);
  while(true) {
    ProtoSpaceAndComments(scanner);
    if (nested && (scanner->Peek() == (close_curly ? '}' : '>'))) {
      scanner->One(Scanner::ALL);
      ProtoSpaceAndComments(scanner);
      return true;
    }
    if (!nested && scanner->empty()) { return true; }
    scanner->RestartCapture()
        .Many(Scanner::LETTER_DIGIT_UNDERSCORE)
        .StopCapture();
    StringPiece identifier;
    if (!scanner->GetResult(nullptr, &identifier)) return false;
    bool parsed_colon = false;
    (void)parsed_colon;
    ProtoSpaceAndComments(scanner);
    if (scanner->Peek() == ':') {
      parsed_colon = true;
      scanner->One(Scanner::ALL);
      ProtoSpaceAndComments(scanner);
    }
    if (identifier == "device") {
      if (has_seen[0]) return false;
      has_seen[0] = true;
      string str_value;
      if (!parsed_colon || !::tensorflow::strings::ProtoParseStringLiteralFromScanner(
          scanner, &str_value)) return false;
      SetProtobufStringSwapAllowed(&str_value, msg->mutable_device());
    }
    else if (identifier == "container") {
      if (has_seen[1]) return false;
      has_seen[1] = true;
      string str_value;
      if (!parsed_colon || !::tensorflow::strings::ProtoParseStringLiteralFromScanner(
          scanner, &str_value)) return false;
      SetProtobufStringSwapAllowed(&str_value, msg->mutable_container());
    }
    else if (identifier == "name") {
      if (has_seen[2]) return false;
      has_seen[2] = true;
      string str_value;
      if (!parsed_colon || !::tensorflow::strings::ProtoParseStringLiteralFromScanner(
          scanner, &str_value)) return false;
      SetProtobufStringSwapAllowed(&str_value, msg->mutable_name());
    }
    else if (identifier == "hash_code") {
      if (has_seen[3]) return false;
      has_seen[3] = true;
      uint64 value;
      if (!parsed_colon || !::tensorflow::strings::ProtoParseNumericFromScanner(scanner, &value)) return false;
      msg->set_hash_code(value);
    }
    else if (identifier == "maybe_type_name") {
      if (has_seen[4]) return false;
      has_seen[4] = true;
      string str_value;
      if (!parsed_colon || !::tensorflow::strings::ProtoParseStringLiteralFromScanner(
          scanner, &str_value)) return false;
      SetProtobufStringSwapAllowed(&str_value, msg->mutable_maybe_type_name());
    }
    else if (identifier == "dtypes_and_shapes") {
      const bool is_list = (scanner->Peek() == '[');
      do {
        if (is_list) {
          scanner->One(Scanner::ALL);
          ProtoSpaceAndComments(scanner);
        }
        const char open_char = scanner->Peek();
        if (open_char != '{' && open_char != '<') return false;
        scanner->One(Scanner::ALL);
        ProtoSpaceAndComments(scanner);
        if (!::tensorflow::internal::ProtoParseFromScanner(
            scanner, true, open_char == '{', msg->add_dtypes_and_shapes())) return false;
      } while (is_list && scanner->Peek() == ',');
      if (is_list && !scanner->OneLiteral("]").GetResult()) return false;
    }
    else if (identifier == "allowed_devices") {
      const bool is_list = (scanner->Peek() == '[');
      do {
        if (is_list) {
          scanner->One(Scanner::ALL);
          ProtoSpaceAndComments(scanner);
        }
        string str_value;
        if (!parsed_colon || !::tensorflow::strings::ProtoParseStringLiteralFromScanner(
            scanner, &str_value)) return false;
        SetProtobufStringSwapAllowed(&str_value, msg->add_allowed_devices());
      } while (is_list && scanner->Peek() == ',');
      if (is_list && !scanner->OneLiteral("]").GetResult()) return false;
    }
  }
}

}  // namespace internal

}  // namespace tensorflow
