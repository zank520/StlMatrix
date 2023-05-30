#ifndef STL_MATRIX_HPP
#define STL_MATRIX_HPP

#include <vector>
#include <complex>
#include <algorithm>

#define MULTIPLY_LIMIT_LEN 128
#define EPS 1E-12

// Declare - public
template <class T>
std::vector<std::vector<T>> StrassenMul(int N, const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b);

template <class T>
std::vector<std::vector<T>> leftDivide(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b);

template <class T>
std::vector<std::vector<T>> rightDivide(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b);

template <class T>
std::vector<std::vector<T>> heapUp(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b, bool isHorizontal = true);

template <class T>
std::vector<std::vector<T>> eye(int n, T value);

template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &a);

template <class T>
std::vector<std::vector<std::complex<T>>> transpose(const std::vector<std::vector<std::complex<T>>> &a);

template <class T>
std::vector<std::vector<T>> inverse(const std::vector<std::vector<T>> &a, int type = 0);

template <class T>
std::vector<std::vector<T>> inverseGE(const std::vector<std::vector<T>> &a);

template <class T>
std::vector<std::vector<T>> inverseLU(const std::vector<std::vector<T>> &a);

template <class T>
double norm(const std::vector<std::vector<T>> &a, int type = 2);

template <class T>
double norm(const std::vector<std::vector<std::complex<T>>> &a, int type = 2);

template <class T>
int rank(const std::vector<std::vector<T>> &a);



template <class T>
bool solveLU(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &L, std::vector<std::vector<T>> &U);

template <class T>
bool solveQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R, int type = 1);

template <class T>
void schmidtQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R);

template <class T>
void householderQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R);

template <class T>
void givensQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R);

template <class T>
bool householderHessenberg(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &H);

template <class T>
bool solveEVD(const std::vector<std::vector<T>> &a, std::vector<T> &D, std::vector<std::vector<T>> &V);

template <class T>
bool solveSVD(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &U, std::vector<std::vector<T>> &S,
              std::vector<std::vector<T>> &Vt);




// todo, Iterative Method for Solving Large Sparse Matrices



// Declare - private
template <class T>
std::vector<std::vector<T>> inverseUpper(const std::vector<std::vector<T>> &a);

template <class T>
std::vector<std::vector<T>> inverseLower(const std::vector<std::vector<T>> &a);

template <class T>
T innerProduct(const std::vector<T> &a, const std::vector<T> &b);

template <class T>
T innerProduct(const std::vector<std::vector<T>> &a, int startRow, int col);

template <class T>
std::complex<T> innerProduct(const std::vector<std::complex<T>> &a, const std::vector<std::complex<T>> &b);

template <class T>
T innerProduct(const std::vector<std::vector<std::complex<T>>> &a, int startRow, int col);

template <class T>
T householderWHead(T a, double add);

template <class T>
std::complex<T> householderWHead(std::complex<T> a, double add);

template <class T>
std::vector<T> buildConj(const std::vector<T> &a);

template <class T>
std::vector<std::complex<T>> buildConj(const std::vector<std::complex<T>> &a);

template <class T>
std::vector<T> givensCS(T a, T b);

template <class T>
std::vector<std::complex<T>> givensCS(std::complex<T> a, std::complex<T> b);




template <class T>
void matrixAdd(int n, T** a, T** b, T** c);

template <class T>
void matrixSub(int n, T** a, T** b, T** c);

template <class T>
void matrixMul(T** a, T** b, T** c, int m, int k, int n);

template <class T>
void matrixMul(int n, T** a, T** b, T** c);



// Function

// Matrix operation for stl
template <class T>
std::vector<std::vector<T>> operator + (const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    std::vector<std::vector<T>> res;
    if(a.empty() || a.size() != b.size() || a[0].empty() || a[0].size() != b[0].size()){
        return res; // return empty vector
    }

    size_t rows = a.size();
    size_t cols = a[0].size();
    res.resize(rows, std::vector<T>(cols));

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            res[i][j] = a[i][j] + b[i][j];
        }
    }
    return res;
}

template <class T>
std::vector<std::vector<T>> operator - (const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    std::vector<std::vector<T>> res;
    if(a.empty() || a.size() != b.size() || a[0].empty() || a[0].size() != b[0].size()){
        return res; // return empty vector
    }

    size_t rows = a.size();
    size_t cols = a[0].size();
    res.resize(rows, std::vector<T>(cols));

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            res[i][j] = a[i][j] - b[i][j];
        }
    }
    return res;
}

template <class T>
std::vector<std::vector<T>> operator * (const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    std::vector<std::vector<T>> res;
    if(a.size() == 0 || b.size() == 0 || a[0].size() == 0 || b[0].size() == 0 || a[0].size() != b.size()){
        return res;
    }

    size_t rows = a.size();
    size_t cols = b[0].size();
    res.resize(rows, std::vector<T>(cols, T(0)));

    if(rows == b.size() && rows == cols && rows >= MULTIPLY_LIMIT_LEN && rows%2 == 0){
        res = StrassenMul(rows, a, b);
    }
    else{
        for(size_t i = 0; i < rows; ++i){
            for(size_t k = 0; k < a[0].size(); ++k){
                T temp(a[i][k]);
                for(size_t j = 0; j < cols; ++j){
                    res[i][j] += temp * b[k][j];
                }
            }
        }
    }

    return res;
}

// Only support N%2 == 0 and matrix size like: N*N
template <class T>
std::vector<std::vector<T>> StrassenMul(int N, const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    int n = N >> 1;

    T** S1 = new T* [n];
    T** S2 = new T* [n];
    T** S3 = new T* [n];
    T** S4 = new T* [n];
    T** S5 = new T* [n];
    T** S6 = new T* [n];
    T** S7 = new T* [n];
    T** S8 = new T* [n];
    T** S9 = new T* [n];
    T** S10 = new T* [n];

    T** A11 = new T* [n];
    T** A22 = new T* [n];
    T** B11 = new T* [n];
    T** B22 = new T* [n];

    for(int i = 0; i < n; ++i){
        S1[i] = new T[n];
        S2[i] = new T[n];
        S3[i] = new T[n];
        S4[i] = new T[n];
        S5[i] = new T[n];
        S6[i] = new T[n];
        S7[i] = new T[n];
        S8[i] = new T[n];
        S9[i] = new T[n];
        S10[i] = new T[n];
        A11[i] = new T[n];
        A22[i] = new T[n];
        B11[i] = new T[n];
        B22[i] = new T[n];
    }

    for(int i = 0; i < n; ++i){
        int ii = i + n;
        for(int j = 0; j < n; ++j){
            int jj = j + n;

            S1[i][j] = b[i][jj] - b[ii][jj];
            S2[i][j] = a[i][j] + a[i][jj];
            S3[i][j] = a[ii][j] + a[ii][jj];
            S4[i][j] = b[ii][j] - b[i][j];
            S5[i][j] = a[i][j] + a[ii][jj];
            S6[i][j] = b[i][jj] + b[ii][jj];
            S7[i][j] = a[i][jj] - a[ii][jj];
            S8[i][j] = b[ii][j] + b[ii][jj];
            S9[i][j] = a[i][j] - a[ii][j];
            S10[i][j] = b[i][j] + b[i][jj];

            A11[i][j] = a[i][j];
            A22[i][j] = a[ii][jj];
            B11[i][j] = b[i][j];
            B22[i][j] = b[ii][jj];
        }
    }

    T** P1 = new T* [n];
    T** P2 = new T* [n];
    T** P3 = new T* [n];
    T** P4 = new T* [n];
    T** P5 = new T* [n];
    T** P6 = new T* [n];
    T** P7 = new T* [n];

    for(int i = 0; i < n; ++i){
        P1[i] = new T[n];
        P2[i] = new T[n];
        P3[i] = new T[n];
        P4[i] = new T[n];
        P5[i] = new T[n];
        P6[i] = new T[n];
        P7[i] = new T[n];
    }

    matrixMul(n, A11, S1, P1);
    matrixMul(n, S2, B22, P2);
    matrixMul(n, S3, B11, P3);
    matrixMul(n, A22, S4, P4);
    matrixMul(n, S5, S6, P5);
    matrixMul(n, S7, S8, P6);
    matrixMul(n, S9, S10, P7);

    T** C11 = new T* [n];
    T** C12 = new T* [n];
    T** C21 = new T* [n];
    T** C22 = new T* [n];

    for(int i = 0; i < n; ++i){
        C11[i] = new T[n];
        C12[i] = new T[n];
        C21[i] = new T[n];
        C22[i] = new T[n];
    }

    matrixAdd(n, P5, P4, C11);
    matrixSub(n, C11, P2, C11);
    matrixAdd(n, C11, P6, C11);
    matrixAdd(n, P1, P2, C12);
    matrixAdd(n, P3, P4, C21);
    matrixAdd(n, P5, P1, C22);
    matrixSub(n, C22, P3, C22);
    matrixSub(n, C22, P7, C22);

    std::vector<std::vector<T>> res(N, std::vector<T>(N));
    for(int i = 0; i < n; ++i){
        int ii = i + n;
        for(int j = 0; j < n; ++j){
            int jj = j + n;

            res[i][j] = C11[i][j];
            res[i][jj] = C12[i][j];
            res[ii][j] = C21[i][j];
            res[ii][jj] = C22[i][j];
        }
    }
    return res;
}

// [x x] * [x x   -> [y y]
//          x x]
template <class T>
std::vector<T> operator * (const std::vector<T> &a, const std::vector<std::vector<T>> &b){
    std::vector<T> res;
    if(a.size() == 0 || b.size() == 0 || b[0].size() == 0 || a.size() != b.size()){
        return res;
    }

    size_t len = b[0].size();
    res.resize(len, T(0));

    for(size_t i = 0; i < len; ++i){
        for(size_t k = 0; k < a.size(); ++k){
            res[i] += a[k] * b[k][i];
        }
    }
    return res;
}

// [x x  * [x   -> [y
//  x x]    x]      y]
template <class T>
std::vector<T> operator * (const std::vector<std::vector<T>> &a, const std::vector<T> &b){
    std::vector<T> res;
    if(a.size() == 0 || b.size() == 0 || a[0].size() == 0 || a[0].size() != b.size()){
        return res;
    }

    size_t len = a.size();
    res.resize(len, T(0));

    for(size_t i = 0; i < len; ++i){
        for(size_t k = 0; k < b.size(); ++k){
            res[i] += a[i][k] * b[k];
        }
    }
    return res;
}

template <class T>
std::vector<std::vector<T>> operator / (const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    return rightDivide(a, b);
}

template <class T>
std::vector<std::vector<T>> leftDivide(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    std::vector<std::vector<T>> res;
    std::vector<std::vector<T>> inva = inverse(a);

    if(!inva.empty()){
        res = inva * b;
    }
    return res;
}

template <class T>
std::vector<std::vector<T>> rightDivide(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b){
    std::vector<std::vector<T>> res;
    std::vector<std::vector<T>> invb = inverse(b);

    if(!invb.empty()){
        res = a * invb;
    }
    return res;
}

// Combine two matrix to one
template <class T>
std::vector<std::vector<T>> heapUp(const std::vector<std::vector<T>> &a, const std::vector<std::vector<T>> &b, bool isHorizontal){
    std::vector<std::vector<T>> res;

    if(isHorizontal){
        if(a.empty() || a[0].empty() || a.size() != b.size()){
            return res;
        }

        size_t rows = a.size();
        size_t cols1 = a[0].size();
        size_t cols2 = b[0].size();
        res.resize(rows, std::vector<T>(cols1 + cols2));

        for(size_t i = 0; i < rows; ++i){
            size_t j = 0;
            for(; j < cols1; ++j){
                res[i][j] = a[i][j];
            }

            for(; j < cols1 + cols2; ++j){
                res[i][j] = b[i][j - cols1];
            }
        }
    }
    else{
        if(a.empty() || a[0].empty() || a[0].size() != b[0].size()){
            return res;
        }

        size_t rows1 = a.size();
        size_t rows2 = b.size();
        size_t cols = a[0].size();
        res.resize(rows1 + rows2, std::vector<T>(cols));

        for(size_t i = 0; i < rows1; ++i){
            res[i] = a[i];
        }
        for(size_t i = 0; i < rows2; ++i){
            res[i + rows1] = a[i];
        }
    }

    return res;
}

// Build diagonal matrix, mostly use eye so named 'eye'
template <class T>
std::vector<std::vector<T>> eye(int n, T value){
    std::vector<std::vector<T>> res(n, std::vector<T>(n, T(0)));
    for(size_t i = 0; i < n; ++i){
        res[i][i] = value;
    }
    return res;
}

// Matrix transposition
template <class T>
std::vector<std::vector<T>> transpose(const std::vector<std::vector<T>> &a){
    std::vector<std::vector<T>> res;

    if(a.empty() || a[0].empty()){
        return res;
    }

    int rows = a.size();
    int cols = a[0].size();
    res.resize(cols, std::vector<T>(rows));

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            res[j][i] = a[i][j];
        }
    }
    return res;
}

template <class T>
std::vector<std::vector<std::complex<T>>> transpose(const std::vector<std::vector<std::complex<T>>> &a){
    std::vector<std::vector<std::complex<T>>> res;

    if(a.empty() || a[0].empty()){
        return res;
    }

    int rows = a.size();
    int cols = a[0].size();
    res.resize(cols, std::vector<std::complex<T>>(rows));

    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            res[j][i] = conj(a[i][j]);
        }
    }
    return res;
}

template <class T>
std::vector<std::vector<T>> inverse(const std::vector<std::vector<T>> &a, int type){
    if(type == 1){
        return inverseLU(a);
    }

    return inverseGE(a);
}

// Using gaussian elimination method, build the augmented matrix
// [x x  [1 0  -> [1 x  [y y  -> [1 0  [y y
//  x x]  0 1]     0 1]  y y]     0 1]  y y]
template <class T>
std::vector<std::vector<T>> inverseGE(const std::vector<std::vector<T>> &a){
    std::vector<std::vector<T>> res;
    if(a.size() == 0 || a.size() != a[0].size()){
        return res;
    }

    size_t n = a.size();
    res = eye(n, T(1));
    std::vector<std::vector<T>> help = a;

    // Complete pivot Gaussian elimination
    for(size_t i = 0; i < n; ++i){
        size_t j = i;
        for(; j < n; ++j){
            if(abs(help[j][i]) > EPS){ // Find next avaliable row
                break;
            }
        }

        if(j == n){ // All value in current col is 0, can't inverse
            std::vector<std::vector<T>> emptyMat;
            return emptyMat;
        }
        else{
            if(j != i){ // Swap rows
                swap(help[i], help[j]);
                swap(res[i], res[j]);
            }

            // Make current row begin with 1
            // [0 1 x x...
            T temp = T(1)/help[i][i];
            for(size_t k = i; k < n; ++k){
                help[i][k] *= temp;
            }
            for(size_t k = 0; k <= i; ++k){
                res[i][k] *= temp;
            }

            // Make lower rows's front part 0
            // [0 0 x x...
            for(size_t k = i + 1; k < n; ++k){
                temp = T(-1)*help[k][i];
                for(size_t l = i; l < n; ++l){
                    help[k][l] += temp*help[i][l];
                }
                for(size_t l = 0; l <= i; ++l){
                    res[k][l] += temp*res[i][l];
                }
            }
        }
    }

    // Now res is [1 x  change it to [1 0
    //             0 1]               0 1]
    for(int i = n - 1; i >= 0; --i){
        for(size_t j = n - 1; j > i; --j){
            T temp = -help[i][j];
            for(size_t k = 0; k < n; ++k){
                res[i][k] += temp*res[j][k];
            }
        }
    }

    return res;
}

// Using LU decomposition
template <class T>
std::vector<std::vector<T>> inverseLU(const std::vector<std::vector<T>> &a){
    std::vector<std::vector<T>> res;
    if(a.size() == 0 || a.size() != a[0].size()){
        return res;
    }

    size_t n = a.size();
    std::vector<std::vector<T>> L;
    std::vector<std::vector<T>> U;
    if(!solveLU(a, L, U)){
        return res;
    }

    std::vector<std::vector<T>> invL = inverseLower(L);
    std::vector<std::vector<T>> invU = inverseUpper(U);
    res = invU * invL;

    return res;
}

// Calculate the norm
template <class T>
double norm(const std::vector<std::vector<T>> &a, int type){
    double res = 0;
    if(type == 0){
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res = std::max(res, (double)fabs(a[i][j]));
            }
        }
    }
    else if(type == 1){
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res += fabs(a[i][j]);
            }
        }
    }
    else{
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res += a[i][j]*a[i][j];
            }
        }
        res = sqrt(res);
    }

    return res;
}

template <class T>
double norm(const std::vector<std::vector<std::complex<T>>> &a, int type){
    double res = 0;
    if(type == 0){
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res = std::max(res, abs(a[i][j]));
            }
        }
    }
    else if(type == 1){
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res += abs(a[i][j]);
            }
        }
    }
    else{
        // 2-norm is the square root of inner product
        for(size_t i = 0; i < a.size(); ++i){
            for(size_t j = 0; j < a[0].size(); ++j){
                res += a[i][j].real()*a[i][j].real() + a[i][j].imag()*a[i][j].imag();
            }
        }
        res = sqrt(res);
    }

    return res;
}

// Get the rank of matrix
template <class T>
int rank(const std::vector<std::vector<T>> &a){
    int r = 0;
    std::vector<std::vector<T>> b = a;
    if(b.size() == 0 || b[0].size() == 0){
        return r;
    }

    size_t rows = b.size();
    size_t cols = b[0].size();

    for(size_t col = 0; col < cols; ++col){
        size_t i = r;
        for(; i < rows; ++i){
            if(abs(b[i][col]) > EPS){
                break;
            }
        }

        if(i == rows){
            continue;
        }
        else{
            if(i != r){
                swap(b[i], b[r]);
            }

            // Make lower rows's front part 0
            for(size_t k = r + 1; k < rows; ++k){
                T temp = T(-1)*b[k][col]/b[r][col];
                for(size_t j = col; j < cols; ++j){
                    b[k][j] += temp*b[r][j];
                }
            }

            r++;
        }
    }

    return r;
}



// LU decomposition: A = LU
// [x x x  -> [1 0 0  * [u u u
//  x x x      l 1 0]    0 u u
//  x x x]     l l 1]    0 0 u]
// Gaussian elimination method is also able to do LU decomposition, but we use another way
// Build the row of U and col of L step by step
template <class T>
bool solveLU(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &L, std::vector<std::vector<T>> &U){
    size_t rows = a.size();
    if(rows == 0) return false;

    size_t cols = a[0].size();
    if(cols == 0) return false;

    L.resize(rows, std::vector<T>(rows, T(0)));
    U.resize(rows, std::vector<T>(cols, T(0)));

    for(size_t i = 0; i < rows; ++i){
        L[i][i] = T(1);
    }

    // Build U's row and L's col one by one
    for(size_t i = 0; i < rows; ++i){

        // Build U's row
        // U: U(ij) = a(ij) - ∑L(ik)*U(kj) k: from 0 to i - 1
        for(size_t j = i; j < cols; ++j){
            T temp = T(0);
            for(size_t k = 0; k < i; ++k){
                temp += L[i][k]*U[k][j];
            }

            U[i][j] = a[i][j] - temp;
        }

        // Build L's col
        // L: L(ji) = a(ji) - ∑L(jk)*U(ki) k: from 0 to i - 1
        for(size_t j = i + 1; j < rows; ++j){
            T temp = T(0);
            for(size_t k = 0; k < i; ++k){
                temp += L[j][k]*U[k][i];
            }

            L[j][i] = (a[j][i] - temp)/U[i][i];
        }
    }

    return true;
}

// QR decomposition: A = QR
template <class T>
bool solveQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R, int type){
    if(a.size() == 0 || a[0].size() == 0){
        return false;
    }

    int m = a.size();
    int n = a[0].size();

    // Full rank check, only matrix with full rank has unique QR result
    if(m < n){
        return false;
    }
    int r = rank(a);
    if(r < n){
        return false;
    }

    if(type == 0){
        schmidtQR(a, Q, R);
    }
    else if(type == 1){
        householderQR(a, Q, R);
    }
    else if(type == 2){
        givensQR(a, Q, R);
    }
    else{
        householderQR(a, Q, R);
    }

    return true;
}

// QR decomposition, using Gram-Schmidt, it's the definition method
// It is not available for complex matrix
template <class T>
void schmidtQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R){
    size_t rows = a.size();
    size_t cols = a[0].size();

    // Orthogonalization
    std::vector<std::vector<T>> b = transpose(a);
    std::vector<T> innerQ(cols, 0);
    Q = b;
    R = eye(cols, T(1));
    for(size_t i = 1; i < cols; ++i){
        innerQ[i - 1] = innerProduct(Q[i - 1], Q[i - 1]);

        for(size_t j = 0; j < i; ++j){
            T temp = innerProduct(Q[j], b[i])/innerQ[j];
            R[j][i] = temp;

            for(size_t k = 0; k < rows; ++k){
                Q[i][k] -= temp*Q[j][k];
            }
        }
    }
    innerQ[cols - 1] = innerProduct(Q[cols - 1], Q[cols - 1]);

    // Unitization
    for(size_t i = 0; i < cols; ++i){
        T temp1 = sqrt(innerQ[i]);
        T temp2 = T(1)/temp1;

        // Build Q
        for(size_t j = 0; j < rows; ++j){
            Q[i][j] *= temp2;
        }

        // Build R
        for(size_t k = 0; k < cols; ++k){
            R[i][k] *= temp1;
        }
    }
    Q = transpose(Q);
}

// QR decomposition, using Householder, it's the default method
template <class T>
void householderQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R){
    size_t rows = a.size();
    size_t cols = a[0].size();

    size_t cnt = rows == cols ? cols - 1 : cols;        // Iterations of H
    std::vector<std::vector<T>> Qi = eye(rows, T(1));   // Q = H1 * H2 * ... * Hi  update in every iteration
    std::vector<std::vector<T>> Ri = a;                 // R = Hi * (H(i - 1) (... * (H1 * A)))

    for(size_t i = 0; i < cnt; ++i){
        double normRi = sqrt(innerProduct(Ri, i, i));
        size_t len = rows - i;

        // w = [(Rii - a) Rji Rji ...] j from i to rows, norm(a) = normRi
        // For complex, a*conj(Rii) should be real
        std::vector<T> w(len);
        std::vector<T> conjw; // It is only for complex
        w[0] = householderWHead(Ri[i][i], normRi);
        for(size_t j = 1; j < len; ++j){
            w[j] = Ri[j + i][i];
        }
        const T factor = -2.0/innerProduct(w, w);
        conjw = buildConj(w);

        // Update Qi, only need to update the right part
        // Qi = Q(i - 1) * Hi
        // Hi = I - 2 * w * w(trans)
        // Qi = Q(i - 1) - 2 * Q(i - 1) * (w * w(trans))
        // [A B  * ([a * [a b]) = [aaA+abB abA+bbB  = [(aA+bB)a (aA+bB)b
        //  C D]     b]            aaC+abD abC+bbD]    (aC+bD)a (aC+bD)b]
        // For complex : [A B  * ([a * [a' b']) = [a'aA+a'bB ab'A+bb'B  = [(aA+bB)a' (aA+bB)b'
        //                C D]     b]              a'aC+a'bD ab'C+bb'D]    (aC+bD)a' (aC+bD)b']
        for(size_t j = 0; j < rows; ++j){
            T temp(0);
            for(size_t k = 0; k < len; ++k){
                temp += w[k]*Qi[j][i + k];
            }
            for(size_t k = 0; k < len; ++k){
                Qi[j][i + k] += factor*(temp*conjw[k]);
            }
        }

        // Update Ri, only need to update the lower right part
        // Ri = Hi * A
        // Ri = A - 2 * w * w(trans) * A
        // [a * [a b]  * [A B  = [aaA+abC aaB+abB  = [(aA+bC)a (aB+bD)a
        //  b]            C D]    abA+bbC abB+bbD]    (aA+bC)b (aB+bD)b]
        for(size_t j = i; j < cols; ++j){
            T temp(0);
            for(size_t k = 0; k < len; ++k){
                temp += conjw[k]*Ri[i + k][j];
            }
            for(size_t k = 0; k < len; ++k){
                Ri[i + k][j] += factor*temp*w[k];
            }
        }
    }

    // Build Q R by Qi Ri
    Q.resize(rows, std::vector<T>(cols, T(0)));
    R.resize(cols, std::vector<T>(cols, T(0)));
    for(size_t i = 0; i < rows; ++i){
        for(size_t j = 0; j < cols; ++j){
            Q[i][j] = Qi[i][j];
        }
    }
    for(size_t i = 0; i < cols; ++i){
        for(size_t j = 0; j < cols; ++j){
            R[i][j] = Ri[i][j];
        }
    }
}

// QR decomposition, using Givens
// It is usually for sparse matrix
template <class T>
void givensQR(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &Q, std::vector<std::vector<T>> &R){
    size_t rows = a.size();
    size_t cols = a[0].size();

    size_t cnt = rows == cols ? cols - 1 : cols;
    std::vector<std::vector<T>> RA = a;             // RA = Rn * R(n - 1) * ... * R1 * A

    for(size_t i = 0; i < cnt; ++i){
        for(size_t j = i + 1; j < rows; ++j){
            if(abs(RA[j][i]) < EPS){
                continue;
            }

            // Rij = [1  0 0 0 0
            //        0  c 0 s 0
            //        0  0 1 0 0
            //        0 -s 0 c 0
            //        0  0 0 0 1]
            // Get s and c
            std::vector<T> cs = givensCS(RA[i][i], RA[j][i]);

            // Only need to update value in row of c and s
            RA[i][i] = RA[i][i] * cs[0] + RA[j][i] * cs[1];
            RA[j][i] = T(0);
            for(size_t k = i + 1; k < cols; ++k){
                T temp = RA[i][k];
                RA[i][k] = temp * cs[0] + RA[j][k] * cs[1];
                RA[j][k] = temp * cs[2] + RA[j][k] * cs[3];
            }
        }
    }

    // Build Q R by RA
    R.resize(cols, std::vector<T>(cols, T(0)));
    for(size_t i = 0; i < cols; ++i){
        for(size_t j = 0; j < cols; ++j){
            R[i][j] = RA[i][j];
        }
    }
    Q = a * inverseUpper(R);
}

// Get hessenberg matrix by Householder
// H : [x x x
//      x x x
//      0 x x]
template <class T>
bool householderHessenberg(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &H){
    if(a.size() == 0 || a.size() != a[0].size()){
        return false;
    }

    size_t n = a.size();
    size_t cnt = n - 2;
    H = a;

    for(size_t i = 0; i < cnt; ++i){
        double normH = sqrt(innerProduct(H, i + 1, i));
        size_t len = n - i - 1;

        std::vector<T> w(len);
        std::vector<T> conjw; // It is only for complex
        w[0] = householderWHead(H[i + 1][i], normH);
        for(size_t j = 1; j < len; ++j){
            w[j] = H[j + i + 1][i];
        }
        const T factor = -2.0/innerProduct(w, w);
        conjw = buildConj(w);

        // Hi = Qi(t) * H(i - 1) * Qi
        // Update the lower right part
        for(size_t j = i; j < n; ++j){
            T temp(0);
            for(size_t k = 0; k < len; ++k){
                temp += conjw[k]*H[i + k + 1][j];
            }
            for(size_t k = 0; k < len; ++k){
                H[i + k + 1][j] += factor*temp*w[k];
            }
        }

        // Update the right part
        for(size_t j = 0; j < n; ++j){
            T temp(0);
            for(size_t k = 0; k < len; ++k){
                temp += conjw[k]*H[j][i + k + 1];
            }
            for(size_t k = 0; k < len; ++k){
                H[j][i + k + 1] += factor*(temp*w[k]);
            }
        }
    }

    return true;
}


// EVD decomposition: Ax = bx
// x is the eigen vectors and b is the eigen values
template <class T>
bool solveEVD(const std::vector<std::vector<T>> &a, std::vector<T> &D, std::vector<std::vector<T>> &V){
    if(a.size() == 0 || a.size() != a[0].size()){
        return false;
    }

    // evd by qr
    // Change origin mastrix to hessenberg matrix
    std::vector<std::vector<T>> H;
    if(!householderHessenberg(a, H)){
        return false;
    }

    // Iterate by givensQR
    int len = a.size();
    std::vector<T> eigen_values(len);
    std::vector<T> eigen_values_new(len);
    for(int i = 0; i < len; ++i){
        eigen_values[i] = H[i][i];
    }

    std::vector<std::vector<T>> Q, R, eigen_vector;
    eigen_vector = eye(len, T(1));
    int max_iteration = 1000;
    double delta = 0.000001;
    for(int i = 0; i < max_iteration; ++i){
        givensQR(H, Q, R);
        H = R * Q;
        eigen_vector = eigen_vector * Q;

        for(int i = 0; i < len; ++i){
            eigen_values_new[i] = H[i][i];
        }

        double delta_sum = 0;
        for(int i = 0; i < len; ++i){
            delta_sum += abs(eigen_values_new[i] - eigen_values[i]) / abs(eigen_values_new[i] + eigen_values[i]) / 2;
        }

        delta_sum /= (double)eigen_values.size();
        if(delta_sum < delta){
            break;
        }
        eigen_values = eigen_values_new;
    }

    // Sort D and V by eigen value
    std::vector<int> eigen_value_index(len);
    for(int i = 0; i < len; ++i){
        eigen_value_index[i] = i;
    }
    std::sort(eigen_value_index.begin(), eigen_value_index.end(), [&](int x, int y){ return eigen_values_new[x] < eigen_values_new[y]; });

    D.resize(len);
    V.resize(len, std::vector<T>(len));
    for(int i = 0; i < len; ++i){
        D[i] = eigen_values_new[eigen_value_index[i]];
        for(int j = 0; j < len; ++j){
            V[j][i] = eigen_vector[j][eigen_value_index[i]];
        }
    }

    return true;
}


// SVD decomposition: A = U * S * V(T)
template <class T>
bool solveSVD(const std::vector<std::vector<T>> &a, std::vector<std::vector<T>> &U, std::vector<std::vector<T>> &S,
              std::vector<std::vector<T>> &Vt){

    // todo, using EVD or JacobiSVD



    return true;
}




// Calculate the inverse matrix of upper triangular matrix
template <class T>
std::vector<std::vector<T>> inverseUpper(const std::vector<std::vector<T>> &a){
    int n = a.size();
    std::vector<std::vector<T>> res = eye(n, T(1));

    for(int i = 0; i < n; ++i){
        res[i][i] /= a[i][i];
    }

    // Inverse upper matrix
    // invU(ij) = -1/U(ii) * ∑U(ik)*invU(k,j) k from i + 1 to j
    for(int j = 0; j < n; ++j){
        for(int i = j - 1; i >= 0; --i){
            T temp = T(0);
            for(int k = i + 1; k <= j; ++k){
                temp += a[i][k]*res[k][j];
            }

            res[i][j] -= temp/a[i][i];
        }
    }

    return res;
}

// Calculate the inverse matrix of lower triangular matrix
template <class T>
std::vector<std::vector<T>> inverseLower(const std::vector<std::vector<T>> &a){
    int n = a.size();
    std::vector<std::vector<T>> res = eye(n, T(1));

    for(int i = 0; i < n; ++i){
        res[i][i] /= a[i][i];
    }

    // Inverse lower matrix
    // invL(ij) = -L(jj) * ∑L(ik)*invL(k,j) k from j to i - 1
    for(int j = 0; j < n; ++j){
        for(int i = j + 1; i < n; ++i){
            T temp = T(0);
            for(int k = j; k < i; ++k){
                temp += a[i][k]*res[k][j];
            }

            res[i][j] -= temp*a[j][j];
        }
    }

    return res;
}

// Inner product
template <class T>
T innerProduct(const std::vector<T> &a, const std::vector<T> &b){
    T res = 0;
    if(a.size() != b.size()){
        return res;
    }

    for(size_t i = 0; i < a.size(); ++i){
        res += a[i]*b[i];
    }
    return res;
}

template <class T>
T innerProduct(const std::vector<std::vector<T>> &a, int startRow, int col){
    T res = 0;
    for(size_t i = startRow; i < a.size(); ++i){
        res += a[i][col]*a[i][col];
    }
    return res;
}

// Inner product for complex
template <class T>
std::complex<T> innerProduct(const std::vector<std::complex<T>> &a, const std::vector<std::complex<T>> &b){
    std::complex<T> res = 0;
    if(a.size() != b.size()){
        return res;
    }

    for(size_t i = 0; i < a.size(); ++i){
        res += conj(a[i])*b[i];
    }
    return res;
}

template <class T>
T innerProduct(const std::vector<std::vector<std::complex<T>>> &a, int startRow, int col){
    T res = 0;
    for(size_t i = startRow; i < a.size(); ++i){
        res += a[i][col].real()*a[i][col].real() + a[i][col].imag()*a[i][col].imag();
    }
    return res;
}


// Get the head of householderQR in every iteration
template <class T>
T householderWHead(T a, double add){
    return a > 0 ? a + add : a - add;
}

template <class T>
std::complex<T> householderWHead(std::complex<T> a, double add){
    std::complex<T> res;
    double mod = abs(a);
    res = a + add*a/mod; // res = x*a, so res * conj(a) will be real
    return res;
}

// Copy the vector and turn complex to conj
template <class T>
std::vector<T> buildConj(const std::vector<T> &a){
    std::vector<T> res = a;
    return res;
}

template <class T>
std::vector<std::complex<T>> buildConj(const std::vector<std::complex<T>> &a){
    std::vector<std::complex<T>> res(a.size());
    for(size_t i = 0; i < a.size(); ++i){
        res[i] = conj(a[i]);
    }
    return res;
}

// Calculate the c and s of givens
// [ c s
//  -s c]
template <class T>
std::vector<T> givensCS(T a, T b){
    double temp = 1.0/sqrt(a*a + b*b);
    std::vector<T> res(4);
    res[0] = a*temp;
    res[1] = b*temp;
    res[2] = -b*temp;
    res[3] = a*temp;

    return res;
}

// [ c*e(jθ1)  s*e(jθ2)
//  -s*e(-jθ2) c*e(-jθ1)]
template <class T>
std::vector<std::complex<T>> givensCS(std::complex<T> a, std::complex<T> b){
    double temp = 1.0/sqrt(a.real() * a.real() + a.imag() * a.imag() + b.real() * b.real() + b.imag() * b.imag());
    std::vector<std::complex<T>> res(4);
    res[0] = conj(a)*temp;
    res[1] = conj(b)*temp;
    res[2] = -b*temp;
    res[3] = a*temp;

    return res;
}



// "+" "-" "*" by C, only support matrix with size n*n
template <class T>
void matrixAdd(int n, T** a, T** b, T** c){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            c[i][j] = a[i][j] + b[i][j];
        }
    }
}

template <class T>
void matrixSub(int n, T** a, T** b, T** c){
    for(int i = 0; i < n; ++i){
        for(int j = 0; j < n; ++j){
            c[i][j] = a[i][j] - b[i][j];
        }
    }
}

template <class T>
void matrixMul(T** a, T** b, T** c, int m, int kk, int n){
    for(int k = 0; k < kk; ++k){
        for(int i = 0; i < m; ++i){
            T temp = a[i][k];
            for(int j = 0; j < n; ++j){
                c[i][j] += temp * b[k][j];
            }
        }
    }
}

template <class T>
void matrixMul(int n, T** a, T** b, T** c){
    if(n >= MULTIPLY_LIMIT_LEN){
        n = n >> 1;

        T** S1 = new T* [n];
        T** S2 = new T* [n];
        T** S3 = new T* [n];
        T** S4 = new T* [n];
        T** S5 = new T* [n];
        T** S6 = new T* [n];
        T** S7 = new T* [n];
        T** S8 = new T* [n];
        T** S9 = new T* [n];
        T** S10 = new T* [n];

        T** A11 = new T* [n];
        T** A22 = new T* [n];
        T** B11 = new T* [n];
        T** B22 = new T* [n];

        for(int i = 0; i < n; ++i){
            S1[i] = new T[n];
            S2[i] = new T[n];
            S3[i] = new T[n];
            S4[i] = new T[n];
            S5[i] = new T[n];
            S6[i] = new T[n];
            S7[i] = new T[n];
            S8[i] = new T[n];
            S9[i] = new T[n];
            S10[i] = new T[n];
            A11[i] = new T[n];
            A22[i] = new T[n];
            B11[i] = new T[n];
            B22[i] = new T[n];
        }

        for(int i = 0; i < n; ++i){
            int ii = i + n;
            for(int j = 0; j < n; ++j){
                int jj = j + n;

                S1[i][j] = b[i][jj] - b[ii][jj];
                S2[i][j] = a[i][j] + a[i][jj];
                S3[i][j] = a[ii][j] + a[ii][jj];
                S4[i][j] = b[ii][j] - b[i][j];
                S5[i][j] = a[i][j] + a[ii][jj];
                S6[i][j] = b[i][jj] + b[ii][jj];
                S7[i][j] = a[i][jj] - a[ii][jj];
                S8[i][j] = b[ii][j] + b[ii][jj];
                S9[i][j] = a[i][j] - a[ii][j];
                S10[i][j] = b[i][j] + b[i][jj];

                A11[i][j] = a[i][j];
                A22[i][j] = a[ii][jj];
                B11[i][j] = b[i][j];
                B22[i][j] = b[ii][jj];
            }
        }

        T** P1 = new T* [n];
        T** P2 = new T* [n];
        T** P3 = new T* [n];
        T** P4 = new T* [n];
        T** P5 = new T* [n];
        T** P6 = new T* [n];
        T** P7 = new T* [n];

        for(int i = 0; i < n; ++i){
            P1[i] = new T[n];
            P2[i] = new T[n];
            P3[i] = new T[n];
            P4[i] = new T[n];
            P5[i] = new T[n];
            P6[i] = new T[n];
            P7[i] = new T[n];
        }

        matrixMul(n, A11, S1, P1);
        matrixMul(n, S2, B22, P2);
        matrixMul(n, S3, B11, P3);
        matrixMul(n, A22, S4, P4);
        matrixMul(n, S5, S6, P5);
        matrixMul(n, S7, S8, P6);
        matrixMul(n, S9, S10, P7);

        T** C11 = new T* [n];
        T** C12 = new T* [n];
        T** C21 = new T* [n];
        T** C22 = new T* [n];

        for(int i = 0; i < n; ++i){
            C11[i] = new T[n];
            C12[i] = new T[n];
            C21[i] = new T[n];
            C22[i] = new T[n];
        }

        matrixAdd(n, P5, P4, C11);
        matrixSub(n, C11, P2, C11);
        matrixAdd(n, C11, P6, C11);
        matrixAdd(n, P1, P2, C12);
        matrixAdd(n, P3, P4, C21);
        matrixAdd(n, P5, P1, C22);
        matrixSub(n, C22, P3, C22);
        matrixSub(n, C22, P7, C22);

        for(int i = 0; i < n; ++i){
            int ii = i + n;
            for(int j = 0; j < n; ++j){
                int jj = j + n;

                c[i][j] = C11[i][j];
                c[i][jj] = C12[i][j];
                c[ii][j] = C21[i][j];
                c[ii][jj] = C22[i][j];
            }
        }
    }
    else{
        for(int i = 0; i < n; ++i){
            for(int k = 0; k < n; ++k){
                T temp(a[i][k]);
                for(int j = 0; j < n; ++j){
                    c[i][j] += temp * b[k][j];
                }
            }
        }
    }
}


#endif // STL_MATRIX_HPP
