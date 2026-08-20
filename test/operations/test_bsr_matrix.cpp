#include <iostream>
#include <cmath>
#include <scfd/memory/host.h>
#include <nmfd/operations/bsr_matrix.h>
#include <nmfd/operations/bsr_operations.h>

using T = double;
using Memory = scfd::memory::host;
using bsr_t = nmfd::operations::bsr_matrix<T, Memory>;

void print_bsr(const bsr_t& A, const char* name)
{
    std::cout << "=== " << name << " ===" << std::endl;
    std::cout << "  nrows=" << A.nrows << " ncols=" << A.ncols
              << " nnz=" << A.nnz << " block_sz=" << A.block_sz << std::endl;

    for (std::ptrdiff_t i = 0; i < A.nrows; ++i)
    {
        std::cout << "  row " << i << ":";
        for (std::ptrdiff_t k = A.row_ptrs(i); k < A.row_ptrs(i + 1); ++k)
        {
            std::ptrdiff_t col = A.col_inds(k);
            std::cout << " [col=" << col << " block=(";
            for (std::ptrdiff_t r = 0; r < A.block_sz; ++r)
            {
                for (std::ptrdiff_t c = 0; c < A.block_sz; ++c)
                {
                    if (r > 0 || c > 0) std::cout << ",";
                    std::cout << A.block_val(k, r, c);
                }
            }
            std::cout << ")]";
        }
        std::cout << std::endl;
    }
}

int main()
{
    int errors = 0;

    // ====================================================================
    // Test 1: BSR matrix-vector product (1x1 blocks, diagonal)
    // ====================================================================
    std::cout << "Test 1: BSR mat-vec product (1x1 blocks)" << std::endl;

    // A = [ 2  0 ]
    //     [ 0  3 ]
    bsr_t A1;
    A1.init(2, 2, 2, 1);

    A1.row_ptrs(0) = 0;
    A1.row_ptrs(1) = 1;
    A1.row_ptrs(2) = 2;
    A1.col_inds(0) = 0;
    A1.col_inds(1) = 1;
    A1.block_val(0, 0, 0) = 2.0;
    A1.block_val(1, 0, 0) = 3.0;

    T x1[2] = {1.0, 2.0};
    T y1[2] = {0.0, 0.0};

    nmfd::operations::bsr_mat_vec_prod(A1, x1, y1);

    // Expected: y = [2, 6]
    if (std::abs(y1[0] - 2.0) > 1e-12 || std::abs(y1[1] - 6.0) > 1e-12)
    {
        std::cerr << "  FAIL: y = [" << y1[0] << ", " << y1[1]
                  << "], expected [2, 6]" << std::endl;
        ++errors;
    }
    else
    {
        std::cout << "  PASS" << std::endl;
    }

    // ====================================================================
    // Test 2: BSR matrix-vector product (2x2 blocks)
    // ====================================================================
    std::cout << "Test 2: BSR mat-vec product (2x2 blocks)" << std::endl;

    // Single 2x2 block: [1 2; 3 4]
    bsr_t A2;
    A2.init(1, 1, 1, 2);

    A2.row_ptrs(0) = 0;
    A2.row_ptrs(1) = 1;
    A2.col_inds(0) = 0;
    A2.block_val(0, 0, 0) = 1.0;
    A2.block_val(0, 0, 1) = 2.0;
    A2.block_val(0, 1, 0) = 3.0;
    A2.block_val(0, 1, 1) = 4.0;

    T x2[2] = {1.0, 2.0};
    T y2[2] = {0.0, 0.0};

    nmfd::operations::bsr_mat_vec_prod(A2, x2, y2);

    // Expected: y = [1*1+2*2, 3*1+4*2] = [5, 11]
    if (std::abs(y2[0] - 5.0) > 1e-12 || std::abs(y2[1] - 11.0) > 1e-12)
    {
        std::cerr << "  FAIL: y = [" << y2[0] << ", " << y2[1]
                  << "], expected [5, 11]" << std::endl;
        ++errors;
    }
    else
    {
        std::cout << "  PASS" << std::endl;
    }

    // ====================================================================
    // Test 3: BSR matrix-matrix product
    // ====================================================================
    std::cout << "Test 3: BSR mat-mat product" << std::endl;

    // A = [ I2  I2 ]  (2 block rows, 2 block cols, block_sz=2)
    //     [ I2   0  ]
    bsr_t Amat;
    Amat.init(2, 2, 3, 2);
    Amat.row_ptrs(0) = 0;
    Amat.row_ptrs(1) = 2;
    Amat.row_ptrs(2) = 3;
    Amat.col_inds(0) = 0;
    Amat.col_inds(1) = 1;
    Amat.col_inds(2) = 0;

    // A[0,0] = I2
    Amat.block_val(0, 0, 0) = 1.0; Amat.block_val(0, 0, 1) = 0.0;
    Amat.block_val(0, 1, 0) = 0.0; Amat.block_val(0, 1, 1) = 1.0;
    // A[0,1] = I2
    Amat.block_val(1, 0, 0) = 1.0; Amat.block_val(1, 0, 1) = 0.0;
    Amat.block_val(1, 1, 0) = 0.0; Amat.block_val(1, 1, 1) = 1.0;
    // A[1,0] = I2
    Amat.block_val(2, 0, 0) = 1.0; Amat.block_val(2, 0, 1) = 0.0;
    Amat.block_val(2, 1, 0) = 0.0; Amat.block_val(2, 1, 1) = 1.0;

    // B = [ I2   0  ]
    //     [ 0   I2 ]
    bsr_t Bmat;
    Bmat.init(2, 2, 2, 2);
    Bmat.row_ptrs(0) = 0;
    Bmat.row_ptrs(1) = 1;
    Bmat.row_ptrs(2) = 2;
    Bmat.col_inds(0) = 0;
    Bmat.col_inds(1) = 1;

    Bmat.block_val(0, 0, 0) = 1.0; Bmat.block_val(0, 0, 1) = 0.0;
    Bmat.block_val(0, 1, 0) = 0.0; Bmat.block_val(0, 1, 1) = 1.0;
    Bmat.block_val(1, 0, 0) = 1.0; Bmat.block_val(1, 0, 1) = 0.0;
    Bmat.block_val(1, 1, 0) = 0.0; Bmat.block_val(1, 1, 1) = 1.0;

    bsr_t Cmat;

    nmfd::operations::bsr_mat_mat_prod_skeleton(Amat, Bmat, Cmat);

    if (Cmat.nnz != 3)
    {
        std::cerr << "  FAIL: C.nnz = " << Cmat.nnz << ", expected 3" << std::endl;
        ++errors;
    }
    else
    {
        nmfd::operations::bsr_mat_mat_prod(Amat, Bmat, Cmat);

        bool val_ok = true;
        for (std::ptrdiff_t k = Cmat.row_ptrs(0); k < Cmat.row_ptrs(1); ++k)
        {
            for (int r = 0; r < 2; ++r)
                for (int c = 0; c < 2; ++c)
                {
                    T expected = (r == c) ? 1.0 : 0.0;
                    if (std::abs(Cmat.block_val(k, r, c) - expected) > 1e-12)
                    {
                        std::cerr << "  FAIL: C block k=" << k
                                  << " [" << r << "," << c << "] = "
                                  << Cmat.block_val(k, r, c)
                                  << ", expected " << expected << std::endl;
                        val_ok = false;
                        ++errors;
                    }
                }
        }

        if (val_ok)
            std::cout << "  PASS" << std::endl;
    }

    // ====================================================================
    // Test 4: BSR mat-vec (tridiagonal 4x4, 1x1 blocks)
    // ====================================================================
    std::cout << "Test 4: BSR mat-vec (4x4 tridiagonal)" << std::endl;

    bsr_t A4;
    A4.init(4, 4, 10, 1);
    A4.row_ptrs(0) = 0;
    A4.row_ptrs(1) = 2;
    A4.row_ptrs(2) = 5;
    A4.row_ptrs(3) = 8;
    A4.row_ptrs(4) = 10;
    A4.col_inds(0) = 0; A4.block_val(0, 0, 0) = 2.0;
    A4.col_inds(1) = 1; A4.block_val(1, 0, 0) = -1.0;
    A4.col_inds(2) = 0; A4.block_val(2, 0, 0) = -1.0;
    A4.col_inds(3) = 1; A4.block_val(3, 0, 0) = 2.0;
    A4.col_inds(4) = 2; A4.block_val(4, 0, 0) = -1.0;
    A4.col_inds(5) = 1; A4.block_val(5, 0, 0) = -1.0;
    A4.col_inds(6) = 2; A4.block_val(6, 0, 0) = 2.0;
    A4.col_inds(7) = 3; A4.block_val(7, 0, 0) = -1.0;
    A4.col_inds(8) = 2; A4.block_val(8, 0, 0) = -1.0;
    A4.col_inds(9) = 3; A4.block_val(9, 0, 0) = 2.0;

    T x4[4] = {1.0, 2.0, 3.0, 4.0};
    T y4[4] = {0.0, 0.0, 0.0, 0.0};

    nmfd::operations::bsr_mat_vec_prod(A4, x4, y4);

    T expected4[4] = {0.0, 0.0, 0.0, 5.0};
    bool ok4 = true;
    for (int i = 0; i < 4; ++i)
    {
        if (std::abs(y4[i] - expected4[i]) > 1e-12)
        {
            std::cerr << "  FAIL: y[" << i << "] = " << y4[i]
                      << ", expected " << expected4[i] << std::endl;
            ok4 = false;
            ++errors;
        }
    }
    if (ok4)
        std::cout << "  PASS" << std::endl;

    std::cout << std::endl;
    if (errors == 0)
        std::cout << "All tests passed!" << std::endl;
    else
        std::cerr << errors << " test(s) failed!" << std::endl;

    return errors;
}
