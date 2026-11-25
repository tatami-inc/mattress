#include "mattress.h"
#include "utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <cstddef>
#include <cstdint>

#include "tatami/tatami.hpp"

void free_mattress(std::uintptr_t ptr) {
    delete mattress::cast(ptr);
}

pybind11::tuple tatami_dim(std::uintptr_t ptr) {
    const auto& mat = mattress::cast(ptr)->ptr;
    pybind11::tuple output(2);
    output[0] = mat->nrow();
    output[1] = mat->ncol();
    return output;
}

bool tatami_is_sparse(std::uintptr_t ptr) {
    const auto& mat = mattress::cast(ptr)->ptr;
    return mat->is_sparse();
}

bool tatami_prefer_rows(std::uintptr_t ptr) {
    const auto& mat = mattress::cast(ptr)->ptr;
    return mat->prefer_rows();
}

pybind11::array_t<mattress::MatrixValue> tatami_get_single(std::uintptr_t ptr, bool row, mattress::MatrixIndex i) {
    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(row ? NC : NR);
    const auto optr = static_cast<mattress::MatrixValue*>(output.request().ptr);

    auto ext = tatami::consecutive_extractor<false, mattress::MatrixValue, mattress::MatrixIndex>(*mat, row, i, 1);
    auto out = ext->fetch(optr);
    tatami::copy_n(out, output.size(), optr);
    return output;
}

void init_tatami_utils(pybind11::module& m) {
    m.def("free_mattress", &free_mattress);
    m.def("tatami_dim", &tatami_dim);
    m.def("tatami_is_sparse", &tatami_is_sparse);
    m.def("tatami_prefer_rows", &tatami_prefer_rows);
    m.def("tatami_get_single", &tatami_get_single);
}
