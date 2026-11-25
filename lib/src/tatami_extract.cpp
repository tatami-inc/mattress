#include "mattress.h"
#include "utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <cstddef>
#include <cstdint>

#include "tatami/tatami.hpp"

pybind11::array_t<mattress::MatrixValue> tatami_extract_dense_subset(
    std::uintptr_t ptr,
    bool row_noop,
    const pybind11::array& row_sub,
    bool col_noop,
    const pybind11::array& col_sub
) {
    auto mat = mattress::cast(ptr)->ptr;

    if (!row_noop) {
        auto rptr = check_numpy_array<mattress::MatrixIndex>(row_sub);
        auto tmp = tatami::make_DelayedSubset<0>(std::move(mat), tatami::ArrayView<mattress::MatrixIndex>(rptr, row_sub.size()));
        mat.swap(tmp);
    }

    if (!col_noop) {
        auto cptr = check_numpy_array<mattress::MatrixIndex>(col_sub);
        auto tmp = tatami::make_DelayedSubset<1>(std::move(mat), tatami::ArrayView<mattress::MatrixIndex>(cptr, col_sub.size()));
        mat.swap(tmp);
    }

    const auto NR = mat->nrow(), NC = mat->ncol();
    pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ cast_Index_to_ArrayExtent(NR), cast_Index_to_ArrayExtent(NC) });
    auto optr = static_cast<mattress::MatrixValue*>(output.request().ptr);

    // Don't bother attempting parallelization here, as the delayedarray generic doesn't accept additional arguments.
    tatami::convert_to_dense(*mat, false, optr, {});
    return output;
}

pybind11::object tatami_extract_sparse_subset(
    std::uintptr_t ptr,
    bool row_noop,
    const pybind11::array& row_sub,
    bool col_noop,
    const pybind11::array& col_sub
) {
    auto mat = mattress::cast(ptr)->ptr;

    if (!row_noop) {
        auto rptr = check_numpy_array<mattress::MatrixIndex>(row_sub);
        auto tmp = tatami::make_DelayedSubset<0>(std::move(mat), tatami::ArrayView<mattress::MatrixIndex>(rptr, row_sub.size()));
        mat.swap(tmp);
    }

    if (!col_noop) {
        auto cptr = check_numpy_array<mattress::MatrixIndex>(col_sub);
        auto tmp = tatami::make_DelayedSubset<1>(std::move(mat), tatami::ArrayView<mattress::MatrixIndex>(cptr, col_sub.size()));
        mat.swap(tmp);
    }

    const auto NR = mat->nrow(), NC = mat->ncol();
    auto content = tatami::create_container_of_Index_size<pybind11::list>(NC);
    if (mat->prefer_rows()) {
        // First pass to extract the content (this could be parallelized).
        auto vcollection = tatami::create_container_of_Index_size<std::vector<std::vector<mattress::MatrixValue> > >(NR);
        auto icollection = tatami::create_container_of_Index_size<std::vector<std::vector<mattress::MatrixIndex> > >(NR);
        auto ranges = tatami::create_container_of_Index_size<std::vector<tatami::SparseRange<mattress::MatrixValue, mattress::MatrixIndex> > >(NR);

        auto ext = tatami::consecutive_extractor<true, mattress::MatrixValue, mattress::MatrixIndex>(*mat, true, 0, NR);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<mattress::MatrixValue> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<mattress::MatrixIndex> >(NC);

        for (I<decltype(NR)> r = 0; r < NR; ++r) {
            auto info = ext->fetch(vbuffer.data(), ibuffer.data());
            if (info.value == vbuffer.data()) {
                vcollection[r].insert(vcollection[r].end(), info.value, info.value + info.number);
                info.value = vcollection[r].data();
            }
            if (info.index == ibuffer.data()) {
                icollection[r].insert(icollection[r].end(), info.index, info.index + info.number);
                info.index = icollection[r].data();
            }
            ranges[r] = info;
        }

        // Second pass to pre-allocate the pybind arrays.
        auto counts = tatami::create_container_of_Index_size<std::vector<mattress::MatrixIndex> >(NC);
        for (I<decltype(NR)> r = 0; r < NR; ++r) {
            const auto& info = ranges[r];
            for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                counts[info.index[i]] += 1;
            }
        }
        auto ires = tatami::create_container_of_Index_size<std::vector<pybind11::array_t<mattress::MatrixIndex> > >(NC);
        auto vres = tatami::create_container_of_Index_size<std::vector<pybind11::array_t<mattress::MatrixValue> > >(NC);
        auto iptrs = tatami::create_container_of_Index_size<std::vector<mattress::MatrixIndex*> >(NC);
        auto vptrs = tatami::create_container_of_Index_size<std::vector<mattress::MatrixValue*> >(NC);
        tatami::cast_Index_to_container_size<pybind11::array_t<mattress::MatrixIndex> >(NR); // check that all allocations inside the loop are safe.
        tatami::cast_Index_to_container_size<pybind11::array_t<mattress::MatrixValue> >(NR);
        for (I<decltype(NC)> c = 0; c < NC; ++c) {
            if (counts[c]) {
                ires[c] = pybind11::array_t<mattress::MatrixIndex>(counts[c]);
                iptrs[c] = static_cast<mattress::MatrixIndex*>(ires[c].request().ptr);
                vres[c] = pybind11::array_t<mattress::MatrixValue>(counts[c]);
                vptrs[c] = static_cast<mattress::MatrixValue*>(vres[c].request().ptr);
            }
        }

        // Third pass to populate the pybind arrays.
        std::fill(counts.begin(), counts.end(), 0);
        for (I<decltype(NR)> r = 0; r < NR; ++r) {
            const auto& info = ranges[r];
            for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                auto& pos = counts[info.index[i]];
                iptrs[info.index[i]][pos] = r;
                vptrs[info.index[i]][pos] = info.value[i];
                ++pos;
            }
        }
        for (I<decltype(NC)> c = 0; c < NC; ++c) {
            if (counts[c]) {
                pybind11::list tmp(2);
                tmp[0] = std::move(ires[c]);
                tmp[1] = std::move(vres[c]);
                content[c] = std::move(tmp);
            } else {
                content[c] = pybind11::none();
            }
        }

    } else {
        auto ext = tatami::consecutive_extractor<true, mattress::MatrixValue, mattress::MatrixIndex>(*mat, false, 0, NC);
        auto vbuffer = tatami::create_container_of_Index_size<std::vector<mattress::MatrixValue> >(NC);
        auto ibuffer = tatami::create_container_of_Index_size<std::vector<mattress::MatrixIndex> >(NC);
        tatami::cast_Index_to_container_size<pybind11::array_t<mattress::MatrixIndex> >(NR); // check that all allocations inside the loop are safe.
        tatami::cast_Index_to_container_size<pybind11::array_t<mattress::MatrixValue> >(NR);

        for (I<decltype(NC)> c = 0; c < NC; ++c) {
            auto info = ext->fetch(vbuffer.data(), ibuffer.data());
            if (info.number) {
                pybind11::list tmp(2);
                tmp[0] = pybind11::array_t<mattress::MatrixIndex>(info.number, info.index);
                tmp[1] = pybind11::array_t<mattress::MatrixValue>(info.number, info.value);
                content[c] = std::move(tmp);
            } else {
                content[c] = pybind11::none();
            }
        }
    }

    pybind11::tuple shape(2);
    shape[0] = NR;
    shape[1] = NC;
    pybind11::module bu = pybind11::module::import("delayedarray");
    return bu.attr("SparseNdarray")(shape, content, pybind11::dtype("float64"), pybind11::dtype("uint32"), false, false);
}

void init_tatami_extract(pybind11::module& m) {
    m.def("tatami_extract_dense_subset", &tatami_extract_dense_subset);
    m.def("tatami_extract_sparse_subset", &tatami_extract_sparse_subset);
}
