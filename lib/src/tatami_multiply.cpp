#include "mattress.h"
#include "utils.h"

#include "pybind11/pybind11.h"
#include "pybind11/numpy.h"

#include <cstddef>
#include <cstdint>

#include "tatami/tatami.hpp"
#include "tatami_mult/tatami_mult.hpp"
#include "sanisizer/sanisizer.hpp"

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
        auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(mat->nrow());
        tatami_mult::multiply_with_single_vector(*mat, rptr, static_cast<mattress::MatrixValue*>(output.request().ptr), opt);
        return output;

    } else {
        if (!sanisizer::is_equal(other.size(), mat->nrow())) {
            throw std::runtime_error("length of LHS vector does not match the number of matrix rows");
        }
        auto output = tatami::create_container_of_Index_size<pybind11::array_t<mattress::MatrixValue> >(mat->ncol());
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
        pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ cast_Index_to_ArrayExtent(out_nrow), num_other });
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
        pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ num_other, cast_Index_to_ArrayExtent(out_nrow) });
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

    pybind11::array_t<mattress::MatrixValue, pybind11::array::f_style> output({ cast_Index_to_ArrayExtent(lmat->nrow()), cast_Index_to_ArrayExtent(rmat->ncol()) });
    sanisizer::product<std::size_t>(lmat->nrow(), rmat->ncol()); // check that array is accessible via a size_t.
    tatami_mult::multiply_with_matrix(*lmat, *rmat, static_cast<double*>(output.request().ptr), false, opt);
    return output;
}

void init_tatami_multiply(pybind11::module& m) {
    m.def("tatami_multiply_vector", &tatami_multiply_vector);
    m.def("tatami_multiply_columns", &tatami_multiply_columns);
    m.def("tatami_multiply_matrix", &tatami_multiply_matrix);
}
