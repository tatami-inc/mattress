/* DO NOT MODIFY: this is automatically generated by the cpptypes */

#include <cstring>
#include <stdexcept>
#include <cstdint>

#ifdef _WIN32
#define PYAPI __declspec(dllexport)
#else
#define PYAPI
#endif

static char* copy_error_message(const char* original) {
    auto n = std::strlen(original);
    auto copy = new char[n + 1];
    std::strcpy(copy, original);
    return copy;
}

void extract_column(void*, int32_t, double*);

int extract_ncol(const void*);

int extract_nrow(const void*);

void extract_row(void*, int32_t, double*);

int extract_sparse(const void*);

void free_mat(void*);

void* initialize_compressed_sparse_matrix(int32_t, int32_t, uint64_t, const char*, void*, const char*, void*, void*, uint8_t);

void* initialize_delayed_binary_isometric_op(void*, void*, const char*);

void* initialize_delayed_combine(int32_t, uintptr_t*, int32_t);

void* initialize_delayed_subset(void*, int32_t, const int32_t*, int32_t);

void* initialize_delayed_transpose(void*);

void* initialize_delayed_unary_isometric_op_simple(void*, const char*);

void* initialize_delayed_unary_isometric_op_with_scalar(void*, const char*, bool, double);

void* initialize_delayed_unary_isometric_op_with_vector(void*, const char*, uint8_t, int32_t, const double*);

void* initialize_dense_matrix(int32_t, int32_t, const char*, void*, uint8_t);

extern "C" {

PYAPI void free_error_message(char** msg) {
    delete [] *msg;
}

PYAPI void py_extract_column(void* rawmat, int32_t c, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_column(rawmat, c, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int py_extract_ncol(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_ncol(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI int py_extract_nrow(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_nrow(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_extract_row(void* rawmat, int32_t r, double* output, int32_t* errcode, char** errmsg) {
    try {
        extract_row(rawmat, r, output);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI int py_extract_sparse(const void* mat, int32_t* errcode, char** errmsg) {
    int output = 0;
    try {
        output = extract_sparse(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void py_free_mat(void* mat, int32_t* errcode, char** errmsg) {
    try {
        free_mat(mat);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
}

PYAPI void* py_initialize_compressed_sparse_matrix(int32_t nr, int32_t nc, uint64_t nz, const char* dtype, void* dptr, const char* itype, void* iptr, void* indptr, uint8_t byrow, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_compressed_sparse_matrix(nr, nc, nz, dtype, dptr, itype, iptr, indptr, byrow);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_binary_isometric_op(void* left, void* right, const char* op, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_binary_isometric_op(left, right, op);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_combine(int32_t n, uintptr_t* ptrs, int32_t dim, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_combine(n, ptrs, dim);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_subset(void* ptr, int32_t dim, const int32_t* subset, int32_t len, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_subset(ptr, dim, subset, len);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_transpose(void* ptr, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_transpose(ptr);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_simple(void* ptr, const char* op, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_simple(ptr, op);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_with_scalar(void* ptr, const char* op, bool right, double arg, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_with_scalar(ptr, op, right, arg);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_delayed_unary_isometric_op_with_vector(void* ptr, const char* op, uint8_t right, int32_t along, const double* args, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_delayed_unary_isometric_op_with_vector(ptr, op, right, along, args);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

PYAPI void* py_initialize_dense_matrix(int32_t nr, int32_t nc, const char* type, void* ptr, uint8_t byrow, int32_t* errcode, char** errmsg) {
    void* output = NULL;
    try {
        output = initialize_dense_matrix(nr, nc, type, ptr, byrow);
    } catch(std::exception& e) {
        *errcode = 1;
        *errmsg = copy_error_message(e.what());
    } catch(...) {
        *errcode = 1;
        *errmsg = copy_error_message("unknown C++ exception");
    }
    return output;
}

}
