#include "mattress.h"
#include "utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <cstddef>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "sanisizer/sanisizer.hpp"

pybind11::array_t<mattress::MatrixValue> tatami_sums(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::SumOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
    tatami_stats::sum(row, *mat, static_cast<mattress::MatrixValue*>(output.request().ptr), opt);
    return output;
}

pybind11::tuple tatami_variances(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::VarianceOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();
    const auto target_dim = (row ? NR : NC);

    tatami_stats::VarianceBuffers<mattress::MatrixValue> buf;
    auto voutput = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(target_dim);
    buf.variance = static_cast<mattress::MatrixValue*>(voutput.request().ptr);
    auto moutput = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(target_dim);
    buf.mean = static_cast<mattress::MatrixValue*>(moutput.request().ptr);

    tatami_stats::variance(row, *mat, buf, opt);

    pybind11::tuple output(2);
    output[0] = std::move(moutput);
    output[1] = std::move(voutput);
    return output;
}

pybind11::array_t<mattress::MatrixValue> tatami_medians(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::MedianOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
    tatami_stats::median(row, *mat, static_cast<mattress::MatrixValue*>(output.request().ptr), opt);
    return output;
}

pybind11::tuple tatami_ranges(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::RangeOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    const auto outlen = row ? NR : NC;
    auto minout = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(outlen);
    auto maxout = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(outlen);

    tatami_stats::RangeBuffers<mattress::MatrixValue> buf;
    buf.minimum = static_cast<mattress::MatrixValue*>(minout.request().ptr);
    buf.maximum = static_cast<mattress::MatrixValue*>(maxout.request().ptr);
    tatami_stats::range(row, *mat, buf, opt);

    pybind11::tuple output(2);
    output[0] = minout;
    output[1] = maxout;
    return output;
}

pybind11::array_t<mattress::MatrixIndex> tatami_nan_counts(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::CountOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
    tatami_stats::count(
        row,
        *mat,
        static_cast<mattress::MatrixValue*>(output.request().ptr),
        [](mattress::MatrixValue val) -> bool {
            return std::isnan(val);
        },
        opt
    );
    return output;
}

inline pybind11::array allocate_output_matrix(mattress::MatrixIndex num_groups, mattress::MatrixIndex otherdim, bool row) {
    if (row) {
        return pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style>({ cast_Index_to_ArrayExtent(otherdim), cast_Index_to_ArrayExtent(num_groups) });
    } else {
        return pybind11::array_t<mattress::MatrixValue, pybind11::array::c_style>({ cast_Index_to_ArrayExtent(num_groups), cast_Index_to_ArrayExtent(otherdim) });
    }
}

auto allocate_output_ptrs(pybind11::array& x, mattress::MatrixIndex num_groups, mattress::MatrixIndex otherdim) {
    sanisizer::product<std::size_t>(num_groups, otherdim); // check that this allocation doesn't overflow when accessed by raw pointers.
    const auto xptr = static_cast<mattress::MatrixValue*>(x.request().ptr);
    auto ptrs = sanisizer::create<std::vector<mattress::MatrixValue*> >(num_groups);
    for (I<decltype(num_groups)> g = 0; g < num_groups; ++g) {
        ptrs[g] = xptr + sanisizer::product_unsafe<std::size_t>(g, otherdim);
    }
    return ptrs;
}

pybind11::array_t<mattress::MatrixValue> tatami_group_sum(
    std::uintptr_t ptr,
    bool row,
    const pybind11::array& grouping,
    mattress::MatrixIndex num_groups,
    int num_threads
) {
    tatami_stats::GroupSumOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    const auto gptr = check_numpy_array<mattress::MatrixIndex>(grouping);
    if (!sanisizer::is_equal(grouping.size(), row ? NC : NR)) {
        throw std::runtime_error("length of 'group' is not consistent with the matrix dimensions");
    }

    const auto target_dim = row ? NR : NC;
    auto output = allocate_output_matrix(num_groups, target_dim, row);
    auto ptrs = allocate_output_ptrs(output, num_groups, target_dim);

    tatami_stats::group_sum(row, *mat, gptr, num_groups, ptrs, opt);
    return output;
}

pybind11::tuple tatami_group_variance(
    std::uintptr_t ptr,
    bool row,
    const pybind11::array& grouping,
    mattress::MatrixIndex num_groups,
    int num_threads
) {
    tatami_stats::GroupVarianceOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    const auto gptr = check_numpy_array<mattress::MatrixIndex>(grouping);
    if (!sanisizer::is_equal(grouping.size(), row ? NC : NR)) {
        throw std::runtime_error("length of 'group' is not consistent with the matrix dimensions");
    }

    const auto target_dim = row ? NR : NC;
    auto voutput = allocate_output_matrix(num_groups, target_dim, row);
    auto moutput = allocate_output_matrix(num_groups, target_dim, row);

    tatami_stats::GroupVarianceBuffers<mattress::MatrixValue> buf;
    buf.mean = allocate_output_ptrs(moutput, num_groups, target_dim);
    buf.variance = allocate_output_ptrs(voutput, num_groups, target_dim);

    tatami_stats::group_variance(row, *mat, gptr, num_groups, buf, opt);

    pybind11::tuple output(2);
    output[0] = std::move(moutput);
    output[1] = std::move(voutput);
    return output;
}

pybind11::array_t<mattress::MatrixValue> tatami_group_median(
    std::uintptr_t ptr,
    bool row,
    const pybind11::array& grouping,
    mattress::MatrixIndex num_groups,
    int num_threads
) {
    tatami_stats::GroupMedianOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto gptr = check_numpy_array<mattress::MatrixIndex>(grouping);
    if (!sanisizer::is_equal(grouping.size(), row ? NC : NR)) {
        throw std::runtime_error("length of 'grouping' is not consistent with the matrix dimensions");
    }

    const auto target_dim = row ? NR : NC;
    auto output = allocate_output_matrix(num_groups, target_dim, row);
    auto ptrs = allocate_output_ptrs(output, num_groups, target_dim);
    tatami_stats::group_median(row, *mat, gptr, num_groups, ptrs, opt);
    return output;
}

void init_tatami_stats(pybind11::module& m) {
    m.def("tatami_sums", &tatami_sums);
    m.def("tatami_variances", &tatami_variances);
    m.def("tatami_medians", &tatami_medians);
    m.def("tatami_ranges", &tatami_ranges);
    m.def("tatami_nan_counts", &tatami_nan_counts);

    m.def("tatami_group_sum", &tatami_group_sum);
    m.def("tatami_group_variance", &tatami_group_variance);
    m.def("tatami_group_median", &tatami_group_median);
}
