// This file is generated by rust-protobuf 3.2.0. Do not edit
// .proto file is parsed by protoc --rust-out=...
// @generated

// https://github.com/rust-lang/rust-clippy/issues/702
#![allow(unknown_lints)]
#![allow(clippy::all)]

#![allow(unused_attributes)]
#![cfg_attr(rustfmt, rustfmt::skip)]

#![allow(box_pointers)]
#![allow(dead_code)]
#![allow(missing_docs)]
#![allow(non_camel_case_types)]
#![allow(non_snake_case)]
#![allow(non_upper_case_globals)]
#![allow(trivial_casts)]
#![allow(unused_results)]
#![allow(unused_mut)]

//! Generated file from `matrix.proto`

/// Generated files are compatible only with the same version
/// of protobuf runtime.
const _PROTOBUF_VERSION_CHECK: () = ::protobuf::VERSION_3_2_0;

#[derive(PartialEq,Clone,Default,Debug)]
// @@protoc_insertion_point(message:Mat)
pub struct Mat {
    // message fields
    // @@protoc_insertion_point(field:Mat.width)
    pub width: i32,
    // @@protoc_insertion_point(field:Mat.height)
    pub height: i32,
    // @@protoc_insertion_point(field:Mat.data)
    pub data: ::std::vec::Vec<f32>,
    // special fields
    // @@protoc_insertion_point(special_field:Mat.special_fields)
    pub special_fields: ::protobuf::SpecialFields,
}

impl<'a> ::std::default::Default for &'a Mat {
    fn default() -> &'a Mat {
        <Mat as ::protobuf::Message>::default_instance()
    }
}

impl Mat {
    pub fn new() -> Mat {
        ::std::default::Default::default()
    }

    fn generated_message_descriptor_data() -> ::protobuf::reflect::GeneratedMessageDescriptorData {
        let mut fields = ::std::vec::Vec::with_capacity(3);
        let mut oneofs = ::std::vec::Vec::with_capacity(0);
        fields.push(::protobuf::reflect::rt::v2::make_simpler_field_accessor::<_, _>(
            "width",
            |m: &Mat| { &m.width },
            |m: &mut Mat| { &mut m.width },
        ));
        fields.push(::protobuf::reflect::rt::v2::make_simpler_field_accessor::<_, _>(
            "height",
            |m: &Mat| { &m.height },
            |m: &mut Mat| { &mut m.height },
        ));
        fields.push(::protobuf::reflect::rt::v2::make_vec_simpler_accessor::<_, _>(
            "data",
            |m: &Mat| { &m.data },
            |m: &mut Mat| { &mut m.data },
        ));
        ::protobuf::reflect::GeneratedMessageDescriptorData::new_2::<Mat>(
            "Mat",
            fields,
            oneofs,
        )
    }
}

impl ::protobuf::Message for Mat {
    const NAME: &'static str = "Mat";

    fn is_initialized(&self) -> bool {
        true
    }

    fn merge_from(&mut self, is: &mut ::protobuf::CodedInputStream<'_>) -> ::protobuf::Result<()> {
        while let Some(tag) = is.read_raw_tag_or_eof()? {
            match tag {
                8 => {
                    self.width = is.read_int32()?;
                },
                16 => {
                    self.height = is.read_int32()?;
                },
                26 => {
                    is.read_repeated_packed_float_into(&mut self.data)?;
                },
                29 => {
                    self.data.push(is.read_float()?);
                },
                tag => {
                    ::protobuf::rt::read_unknown_or_skip_group(tag, is, self.special_fields.mut_unknown_fields())?;
                },
            };
        }
        ::std::result::Result::Ok(())
    }

    // Compute sizes of nested messages
    #[allow(unused_variables)]
    fn compute_size(&self) -> u64 {
        let mut my_size = 0;
        if self.width != 0 {
            my_size += ::protobuf::rt::int32_size(1, self.width);
        }
        if self.height != 0 {
            my_size += ::protobuf::rt::int32_size(2, self.height);
        }
        my_size += 5 * self.data.len() as u64;
        my_size += ::protobuf::rt::unknown_fields_size(self.special_fields.unknown_fields());
        self.special_fields.cached_size().set(my_size as u32);
        my_size
    }

    fn write_to_with_cached_sizes(&self, os: &mut ::protobuf::CodedOutputStream<'_>) -> ::protobuf::Result<()> {
        if self.width != 0 {
            os.write_int32(1, self.width)?;
        }
        if self.height != 0 {
            os.write_int32(2, self.height)?;
        }
        for v in &self.data {
            os.write_float(3, *v)?;
        };
        os.write_unknown_fields(self.special_fields.unknown_fields())?;
        ::std::result::Result::Ok(())
    }

    fn special_fields(&self) -> &::protobuf::SpecialFields {
        &self.special_fields
    }

    fn mut_special_fields(&mut self) -> &mut ::protobuf::SpecialFields {
        &mut self.special_fields
    }

    fn new() -> Mat {
        Mat::new()
    }

    fn clear(&mut self) {
        self.width = 0;
        self.height = 0;
        self.data.clear();
        self.special_fields.clear();
    }

    fn default_instance() -> &'static Mat {
        static instance: Mat = Mat {
            width: 0,
            height: 0,
            data: ::std::vec::Vec::new(),
            special_fields: ::protobuf::SpecialFields::new(),
        };
        &instance
    }
}

impl ::protobuf::MessageFull for Mat {
    fn descriptor() -> ::protobuf::reflect::MessageDescriptor {
        static descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::MessageDescriptor> = ::protobuf::rt::Lazy::new();
        descriptor.get(|| file_descriptor().message_by_package_relative_name("Mat").unwrap()).clone()
    }
}

impl ::std::fmt::Display for Mat {
    fn fmt(&self, f: &mut ::std::fmt::Formatter<'_>) -> ::std::fmt::Result {
        ::protobuf::text_format::fmt(self, f)
    }
}

impl ::protobuf::reflect::ProtobufValue for Mat {
    type RuntimeType = ::protobuf::reflect::rt::RuntimeTypeMessage<Self>;
}

static file_descriptor_proto_data: &'static [u8] = b"\
    \n\x0cmatrix.proto\"G\n\x03Mat\x12\x14\n\x05width\x18\x01\x20\x01(\x05R\
    \x05width\x12\x16\n\x06height\x18\x02\x20\x01(\x05R\x06height\x12\x12\n\
    \x04data\x18\x03\x20\x03(\x02R\x04dataJ\xdd\x01\n\x06\x12\x04\0\0\x06\
    \x01\n\x08\n\x01\x0c\x12\x03\0\0\x12\n\n\n\x02\x04\0\x12\x04\x02\0\x06\
    \x01\n\n\n\x03\x04\0\x01\x12\x03\x02\x08\x0b\n\x0b\n\x04\x04\0\x02\0\x12\
    \x03\x03\x02\x12\n\x0c\n\x05\x04\0\x02\0\x05\x12\x03\x03\x02\x07\n\x0c\n\
    \x05\x04\0\x02\0\x01\x12\x03\x03\x08\r\n\x0c\n\x05\x04\0\x02\0\x03\x12\
    \x03\x03\x10\x11\n\x0b\n\x04\x04\0\x02\x01\x12\x03\x04\x02\x13\n\x0c\n\
    \x05\x04\0\x02\x01\x05\x12\x03\x04\x02\x07\n\x0c\n\x05\x04\0\x02\x01\x01\
    \x12\x03\x04\x08\x0e\n\x0c\n\x05\x04\0\x02\x01\x03\x12\x03\x04\x11\x12\n\
    \x0b\n\x04\x04\0\x02\x02\x12\x03\x05\x02\x1a\n\x0c\n\x05\x04\0\x02\x02\
    \x04\x12\x03\x05\x02\n\n\x0c\n\x05\x04\0\x02\x02\x05\x12\x03\x05\x0b\x10\
    \n\x0c\n\x05\x04\0\x02\x02\x01\x12\x03\x05\x11\x15\n\x0c\n\x05\x04\0\x02\
    \x02\x03\x12\x03\x05\x18\x19b\x06proto3\
";

/// `FileDescriptorProto` object which was a source for this generated file
fn file_descriptor_proto() -> &'static ::protobuf::descriptor::FileDescriptorProto {
    static file_descriptor_proto_lazy: ::protobuf::rt::Lazy<::protobuf::descriptor::FileDescriptorProto> = ::protobuf::rt::Lazy::new();
    file_descriptor_proto_lazy.get(|| {
        ::protobuf::Message::parse_from_bytes(file_descriptor_proto_data).unwrap()
    })
}

/// `FileDescriptor` object which allows dynamic access to files
pub fn file_descriptor() -> &'static ::protobuf::reflect::FileDescriptor {
    static generated_file_descriptor_lazy: ::protobuf::rt::Lazy<::protobuf::reflect::GeneratedFileDescriptor> = ::protobuf::rt::Lazy::new();
    static file_descriptor: ::protobuf::rt::Lazy<::protobuf::reflect::FileDescriptor> = ::protobuf::rt::Lazy::new();
    file_descriptor.get(|| {
        let generated_file_descriptor = generated_file_descriptor_lazy.get(|| {
            let mut deps = ::std::vec::Vec::with_capacity(0);
            let mut messages = ::std::vec::Vec::with_capacity(1);
            messages.push(Mat::generated_message_descriptor_data());
            let mut enums = ::std::vec::Vec::with_capacity(0);
            ::protobuf::reflect::GeneratedFileDescriptor::new_generated(
                file_descriptor_proto(),
                deps,
                messages,
                enums,
            )
        });
        ::protobuf::reflect::FileDescriptor::new_generated_2(generated_file_descriptor)
    })
}
