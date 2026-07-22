#include <hdf5.h>

static inline hid_t zig_h5p_default(void) { return H5P_DEFAULT; }
static inline hid_t zig_h5s_all(void) { return H5S_ALL; }
static inline hid_t zig_h5e_default(void) { return H5E_DEFAULT; }
static inline hid_t zig_h5p_dataset_create(void) { return H5P_DATASET_CREATE; }
static inline hid_t zig_h5p_link_create(void) { return H5P_LINK_CREATE; }
static inline unsigned zig_h5f_acc_excl(void) { return H5F_ACC_EXCL; }
static inline unsigned zig_h5f_acc_trunc(void) { return H5F_ACC_TRUNC; }
static inline unsigned zig_h5f_acc_rdonly(void) { return H5F_ACC_RDONLY; }
static inline unsigned zig_h5f_acc_rdwr(void) { return H5F_ACC_RDWR; }
static inline H5F_scope_t zig_h5f_scope_global(void) { return H5F_SCOPE_GLOBAL; }

static inline hid_t zig_h5_native_u8(void) { return H5T_NATIVE_UINT8; }
static inline hid_t zig_h5_native_u16(void) { return H5T_NATIVE_UINT16; }
static inline hid_t zig_h5_native_u32(void) { return H5T_NATIVE_UINT32; }
static inline hid_t zig_h5_native_u64(void) { return H5T_NATIVE_UINT64; }
static inline hid_t zig_h5_native_i8(void) { return H5T_NATIVE_INT8; }
static inline hid_t zig_h5_native_i16(void) { return H5T_NATIVE_INT16; }
static inline hid_t zig_h5_native_i32(void) { return H5T_NATIVE_INT32; }
static inline hid_t zig_h5_native_i64(void) { return H5T_NATIVE_INT64; }
static inline hid_t zig_h5_native_f32(void) { return H5T_NATIVE_FLOAT; }
static inline hid_t zig_h5_native_f64(void) { return H5T_NATIVE_DOUBLE; }

static inline hid_t zig_h5_file_u8(void) { return H5T_STD_U8LE; }
static inline hid_t zig_h5_file_u16(void) { return H5T_STD_U16LE; }
static inline hid_t zig_h5_file_u32(void) { return H5T_STD_U32LE; }
static inline hid_t zig_h5_file_u64(void) { return H5T_STD_U64LE; }
static inline hid_t zig_h5_file_i8(void) { return H5T_STD_I8LE; }
static inline hid_t zig_h5_file_i16(void) { return H5T_STD_I16LE; }
static inline hid_t zig_h5_file_i32(void) { return H5T_STD_I32LE; }
static inline hid_t zig_h5_file_i64(void) { return H5T_STD_I64LE; }
static inline hid_t zig_h5_file_f32(void) { return H5T_IEEE_F32LE; }
static inline hid_t zig_h5_file_f64(void) { return H5T_IEEE_F64LE; }
static inline hid_t zig_h5_c_s1(void) { return H5T_C_S1; }

static inline herr_t zig_h5_disable_error_printing(void) {
    return H5Eset_auto2(H5E_DEFAULT, NULL, NULL);
}
