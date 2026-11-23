#include "mattress.h"
#include "utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <cstddef>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "tatami_stats/tatami_stats.hpp"
#include "tatami_mult/tatami_mult.hpp"
#include "sanisizer/sanisizer.hpp"

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

/*************
 *** Stats ***
 *************/

pybind11::array_t<mattress::MatrixValue> tatami_sums(std::uintptr_t ptr, bool row, int num_threads) {
    tatami_stats::SumOptions opt;
    opt.num_threads = num_threads;
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
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
    auto voutput = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(target_dim);
    buf.variance = static_cast<mattress::MatrixValue*>(voutput.request().ptr);
    auto moutput = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(target_dim);
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

    auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
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
    auto minout = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(outlen);
    auto maxout = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(outlen);

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

    auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(row ? NR : NC);
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

/*********************
 *** Grouped stats ***
 *********************/

typedef I<decltype(std::declval<pybind11::array>().request().shape[0])> Extent; 

inline pybind11::array allocate_output_matrix(mattress::MatrixIndex num_groups, mattress::MatrixIndex otherdim, bool row) {
    if (row) {
        return pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style>({ sanisizer::cast<Extent>(otherdim), sanisizer::cast<Extent>(num_groups) });
    } else {
        return pybind11::array_t<mattress::MatrixValue, pybind11::array::c_style>({ sanisizer::cast<Extent>(num_groups), sanisizer::cast<Extent>(otherdim) });
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

/******************
 *** Extraction ***
 ******************/

pybind11::array_t<mattress::MatrixValue> tatami_get_single(std::uintptr_t ptr, bool row, mattress::MatrixIndex i) {
    const auto& mat = mattress::cast(ptr)->ptr;
    const auto NR = mat->nrow();
    const auto NC = mat->ncol();

    auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(row ? NC : NR);
    const auto optr = static_cast<mattress::MatrixValue*>(output.request().ptr);

    auto ext = tatami::consecutive_extractor<false, mattress::MatrixValue, mattress::MatrixIndex>(*mat, row, i, 1);
    auto out = ext->fetch(optr);
    tatami::copy_n(out, output.size(), optr);
    return output;
}

pybind11::array_t<mattress::MatrixValue> tatami_get_dense_subset(
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
    pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ sanisizer::cast<Extent>(NR), sanisizer::cast<Extent>(NC) });
    auto optr = static_cast<mattress::MatrixValue*>(output.request().ptr);

    // Don't bother attempting parallelization here, as the delayedarray generic doesn't accept additional arguments.
    tatami::convert_to_dense(*mat, false, optr, {});
    return output;
}

pybind11::object tatami_get_sparse_subset(
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
    auto content = sanisizer::create<pybind11::list>(NC);
    if (mat->prefer_rows()) {
        // TODO: replace this with a templated convert_to_sparse_matrix.
        auto vcollection = sanisizer::create<std::vector<std::vector<mattress::MatrixValue> > >(NC);
        auto icollection = sanisizer::create<std::vector<std::vector<mattress::MatrixIndex> > >(NC);
        sanisizer::cast<I<decltype(vcollection.front().size())> >(NR);
        sanisizer::cast<I<decltype(icollection.front().size())> >(NR);

        auto ext = tatami::consecutive_extractor<true, mattress::MatrixValue, mattress::MatrixIndex>(*mat, true, 0, NR);
        auto vbuffer = sanisizer::create<std::vector<mattress::MatrixValue> >(NC);
        auto ibuffer = sanisizer::create<std::vector<mattress::MatrixIndex> >(NC);

        for (I<decltype(NR)> r = 0; r < NR; ++r) {
            const auto info = ext->fetch(vbuffer.data(), ibuffer.data());
            for (I<decltype(info.number)> i = 0; i < info.number; ++i) {
                const auto c = info.index[i];
                vcollection[c].push_back(info.value[i]);
                icollection[c].push_back(r);
            }
        }

        for (I<decltype(NC)> c = 0; c < NC; ++c) {
            if (vcollection[c].size()) {
                pybind11::list tmp(2);
                tmp[0] = pybind11::array_t<mattress::MatrixIndex>(icollection[c].size(), icollection[c].data());
                tmp[1] = pybind11::array_t<mattress::MatrixValue>(vcollection[c].size(), vcollection[c].data());
                content[c] = std::move(tmp);
            } else {
                content[c] = pybind11::none();
            }
        }

    } else {
        auto ext = tatami::consecutive_extractor<true, mattress::MatrixValue, mattress::MatrixIndex>(*mat, false, 0, NC);
        auto vbuffer = sanisizer::create<std::vector<mattress::MatrixValue> >(NC);
        auto ibuffer = sanisizer::create<std::vector<mattress::MatrixIndex> >(NC);

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

/**********************
 *** Multiplication ***
 **********************/

pybind11::array tatami_multiply_vector(std::uintptr_t ptr, const pybind11::array& other, bool right, int num_threads) {
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }
    tatami_mult::MultiplyWithSingleVectorOptions opt;
    tatami_mult::set_num_threads(opt, num_threads); 

    auto mat = mattress::cast(ptr)->ptr;
    auto rptr = check_numpy_array<mattress::MatrixValue>(other);

    if (right) {
        if (!sanisizer::is_equal(other.size(), mat->ncol())) {
            throw std::runtime_error("length of RHS vector does not match the number of matrix columns");
        }
        auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(mat->nrow());
        tatami_mult::multiply_with_single_vector(*mat, rptr, static_cast<mattress::MatrixValue*>(output.request().ptr), opt);
        return output;

    } else {
        if (!sanisizer::is_equal(other.size(), mat->nrow())) {
            throw std::runtime_error("length of LHS vector does not match the number of matrix rows");
        }
        auto output = sanisizer::create<pybind11::array_t<mattress::MatrixValue> >(mat->ncol());
        tatami_mult::multiply_with_single_vector(rptr, *mat, static_cast<mattress::MatrixValue*>(output.request().ptr), opt);
        return output;
    }
}

pybind11::array tatami_multiply_columns(std::uintptr_t ptr, const pybind11::array& other, bool right, int num_threads) {
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }

    const auto& othershape = other.request().shape;
    if (othershape.size() != 2) {
        throw std::runtime_error("'other' should be a 2-dimensional array");
    }
    const auto otherrow = othershape[0];
    const auto othercol = othershape[1];

    // Don't want to deal with C-style arrays for the time being.
    if (!(other.flags() & pybind11::array::f_style)) {
        throw std::runtime_error("'other' should be a column-major matrix");
    }

    if (!other.dtype().is(pybind11::dtype::of<mattress::MatrixValue>())) {
        throw std::runtime_error("unexpected dtype for NumPy array");
    }
    const auto otherptr = get_numpy_array_data<mattress::MatrixValue>(other);

    if (right) {
        const auto common_dim = otherrow;
        const auto num_other = othercol;

        const auto& mat = mattress::cast(ptr)->ptr;
        if (!sanisizer::is_equal(common_dim, mat->ncol())) {
            throw std::runtime_error("number of RHS rows does not match the number of matrix columns");
        }

        const auto out_nrow = mat->nrow();
        pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ sanisizer::cast<I<decltype(num_other)> >(out_nrow), num_other });
        const auto optr = static_cast<double*>(output.request().ptr);

        sanisizer::product<std::size_t>(num_other, common_dim); // check outside the functor so that we can do unsafe products inside the functor.
        auto fetch_other_col = [&](std::size_t rcol) -> const double* { return otherptr + sanisizer::product_unsafe<std::size_t>(rcol, common_dim); };

        const bool is_sparse = mat->is_sparse(); 
        if (mat->prefer_rows()) {
            if (is_sparse){
                tatami_mult::MultiplySparseRowWithDenseColumnMatrixToColumnOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_sparse_row_with_dense_column_matrix_to_column_output(*mat, num_other, fetch_other_col, optr, opt);
            } else {
                tatami_mult::MultiplyDenseRowWithDenseColumnMatrixToColumnOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_dense_row_with_dense_column_matrix_to_column_output(*mat, num_other, fetch_other_col, optr, opt);
            }

        } else {
            if (is_sparse){
                tatami_mult::MultiplySparseColumnWithDenseColumnMatrixToColumnOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_sparse_column_with_dense_column_matrix_to_column_output(*mat, num_other, fetch_other_col, optr, opt);
            } else {
                tatami_mult::MultiplyDenseColumnWithDenseColumnMatrixToColumnOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_dense_column_with_dense_column_matrix_to_column_output(*mat, num_other, fetch_other_col, optr, opt);
            }
        }

        return output;

    } else {
        // Effectively transposing 'other' here so that it can lie on the right as a row-major matrix.
        const auto common_dim = othercol;
        const auto num_other = otherrow;

        tatami::DelayedTranspose<mattress::MatrixValue, mattress::MatrixIndex> alt(mattress::cast(ptr)->ptr);
        if (!sanisizer::is_equal(common_dim, alt.ncol())) {
            throw std::runtime_error("number of columns of 'vals' does not match the number of rows of 'x'");
        }

        // We'll treat the output as row-major in our (conceptually) everything-is-transposed state.
        const auto out_nrow = alt.nrow();
        pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ num_other, sanisizer::cast<I<decltype(num_other)> >(out_nrow) });
        const auto optr = static_cast<mattress::MatrixValue*>(output.request().ptr);

        sanisizer::product<std::size_t>(num_other, common_dim); // check outside the functor so that we can do unsafe products inside the functor.
        auto fetch_other_row = [&](std::size_t rrow) -> const double* { return otherptr + sanisizer::product_unsafe<std::size_t>(rrow, num_other); };

        const bool is_sparse = alt.is_sparse(); 
        if (alt.prefer_rows()) {
            if (is_sparse){
                tatami_mult::MultiplySparseRowWithDenseRowMatrixToRowOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_sparse_row_with_dense_row_matrix_to_row_output(alt, num_other, fetch_other_row, optr, opt);
            } else {
                tatami_mult::MultiplyDenseRowWithDenseRowMatrixToRowOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_dense_row_with_dense_row_matrix_to_row_output(alt, num_other, fetch_other_row, optr, opt);
            }

        } else {
            if (is_sparse){
                tatami_mult::MultiplySparseColumnWithDenseRowMatrixToRowOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_sparse_column_with_dense_row_matrix_to_row_output(alt, num_other, fetch_other_row, optr, opt);
            } else {
                tatami_mult::MultiplyDenseColumnWithDenseRowMatrixToRowOutputOptions opt;
                opt.num_threads = num_threads;
                tatami_mult::multiply_dense_column_with_dense_row_matrix_to_row_output(alt, num_other, fetch_other_row, optr, opt);
            }
        }

        return output;
    }
}

pybind11::array tatami_multiply_matrix(std::uintptr_t ptr1, std::uintptr_t ptr2, bool right, int num_threads) {
    if (num_threads < 1) {
        throw std::runtime_error("'num_threads' should be a positive integer");
    }
    tatami_mult::MultiplyWithMatrixOptions opt;
    tatami_mult::set_num_threads(opt, num_threads); 

    const auto& mat1 = mattress::cast(ptr1)->ptr;
    const auto& mat2 = mattress::cast(ptr2)->ptr;
    const auto& lmat = (right ? mat1 : mat2);
    const auto& rmat = (right ? mat2 : mat1);
    if (lmat->ncol() != rmat->nrow()) {
        throw std::runtime_error("inconsistent common dimensions for matrix multiplication");
    }

    pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ sanisizer::cast<Extent>(lmat->nrow()), sanisizer::cast<Extent>(rmat->ncol()) });
    sanisizer::product<std::size_t>(lmat->nrow(), rmat->ncol()); // check that array is accessible via a size_t.
    tatami_mult::multiply_with_matrix(*lmat, *rmat, static_cast<double*>(output.request().ptr), false, opt);
    return output;
}

/**********************
 *** Initialization ***
 **********************/

void init_common(pybind11::module& m) {
    m.def("free_mattress", &free_mattress);

    m.def("tatami_dim", &tatami_dim);
    m.def("tatami_is_sparse", &tatami_is_sparse);
    m.def("tatami_prefer_rows", &tatami_prefer_rows);

    m.def("tatami_sums", &tatami_sums);
    m.def("tatami_variances", &tatami_variances);
    m.def("tatami_medians", &tatami_medians);
    m.def("tatami_ranges", &tatami_ranges);
    m.def("tatami_nan_counts", &tatami_nan_counts);

    m.def("tatami_group_sum", &tatami_group_sum);
    m.def("tatami_group_variance", &tatami_group_variance);
    m.def("tatami_group_median", &tatami_group_median);

    m.def("tatami_multiply_vector", &tatami_multiply_vector);
    m.def("tatami_multiply_columns", &tatami_multiply_columns);
    m.def("tatami_multiply_matrix", &tatami_multiply_matrix);

    m.def("tatami_get_single", &tatami_get_single);
    m.def("tatami_get_dense_subset", &tatami_get_dense_subset);
    m.def("tatami_get_sparse_subset", &tatami_get_sparse_subset);
}
