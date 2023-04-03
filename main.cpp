#include <QCoreApplication>

#include "stlmatrix.hpp"
#include <windows.h>
#include <iostream>

int main(int argc, char *argv[])
{
    QCoreApplication a(argc, argv);

    LARGE_INTEGER freq;
    LARGE_INTEGER beginTime;
    LARGE_INTEGER endTime;

    QueryPerformanceFrequency(&freq);
    QueryPerformanceCounter(&beginTime);

//    std::vector<std::vector<double>> a1(3, std::vector<double>(4, 1.1));
//    std::vector<std::vector<double>> b1(3, std::vector<double>(4, 2.2));
//    std::vector<std::vector<std::complex<double>>> a2(4, std::vector<std::complex<double>>(4, std::complex<double>(1.1, 0.1)));
//    std::vector<std::vector<std::complex<double>>> b2(4, std::vector<std::complex<double>>(4, std::complex<double>(2.2, 0.2)));

//    std::vector<std::vector<double>> addc1 = a1 + b1;
//    std::vector<std::vector<std::complex<double>>> addc2 = a2 + b2;

//    std::vector<std::vector<double>> mulc1 = a2 * b2;
//    std::vector<std::vector<std::complex<double>>> mulc2 = a2 * b2;
//    std::vector<std::vector<std::complex<double>>> mulc3 = StrassenMul(4, a2, b2);

//    std::vector<std::vector<std::complex<double>>> trans1 = transpose(a2);

    std::vector<std::vector<double>> a3{{-2, 1, 1}, {0, 2, 0}, {-4, 1, 3}};
    std::vector<std::vector<std::complex<double>>> a4{{1.2, 3.5, -2.4}, {0.8, -7.9, 3.2}, {-4.1, -0.9, -0.9}};
    a4[0][0] = std::complex<double>(5.5, -3.7);
//    std::vector<std::vector<double>> inv1 = inverse(a3);
    std::vector<std::vector<std::complex<double>>> inv2 = inverse(a4);
    std::vector<std::vector<std::complex<double>>> inv21 = a4 * inv2;
//    std::vector<std::vector<std::complex<double>>> inv3 = inverse(a4, 1);

//    double norm1 = norm(a4, 1);
//    int r1 = rank(a2);
//    int r2 = rank(a4);

    std::vector<std::vector<double>> a5{{3, 0}, {0, 1}, {4, 1}};
    std::vector<std::vector<double>> a6{{1.7, 1, -1, 3, 6, 4.5, 3.9, 55}, {1, 0.4, 0, -3, 11, -1, 2.5, -6},
                                        {0, 1, 0, 7, -5, 2, 0, 5.9}, {9, -6, 4, 2, -5, -9.9, 24, 20},
                                        {6.6, 1.8, -0.4, 7, -9.9, 2.8, -44, 3.2}, {1.2, 3.5, -2.4, 7, 0.8, -7.9, 3.2, 8},
                                        {0, -10, 0.8, -7.9, 3.2, 2.8, -4, 20}, {-4.1, -0.9, 2.2, 9, 3.2, -8, -4, 0.1}};
    std::vector<std::vector<std::complex<double>>> a7{{1.2, 1}, {0, 1}, {2, 4}};
    a7[0][0] = std::complex<double>(0, -2);
    a7[1][0] = std::complex<double>(0, 1);
    std::vector<std::vector<std::complex<double>>> a8{{1.2, 1}, {0, 2}};
    a8[0][0] = std::complex<double>(1, 1);
    a8[0][1] = std::complex<double>(0, 1);
    a8[1][0] = std::complex<double>(0, -1);

//    std::vector<std::vector<double>> q1, r1;
//    bool qr1 = solveQR(a6, q1, r1, 0);
//    std::vector<std::vector<double>> qri1 = q1*transpose(q1);
//    std::vector<std::vector<double>> aa1 = q1*r1;

//    std::vector<std::vector<std::complex<double>>> q2, r2;
//    bool qr2 = solveQR(a8, q2, r2, 0);
//    std::vector<std::vector<std::complex<double>>> qri2 = q2*transpose(q2);
//    std::vector<std::vector<std::complex<double>>> aa2 = q2*r2;

//    std::vector<std::vector<std::complex<double>>> q3, r3;
//    bool qr4 = solveQR(a8, q3, r3, 1);
//    std::vector<std::vector<std::complex<double>>> qrii = q3*transpose(q3);
//    std::vector<std::vector<std::complex<double>>> aa = q3*r3;

    std::vector<std::vector<double>> H1;
    bool hh1 = householderHessenberg(a6, H1);

//    std::vector<double> eigenVal;
//    std::vector<std::vector<double>> eigenVec;
//    bool evd1 = solveEVD(a3, eigenVal, eigenVec);

    QueryPerformanceCounter(&endTime);
    int timeCost = (double)(endTime.QuadPart - beginTime.QuadPart) * 1e6 / (double)freq.QuadPart;
    std::cout << timeCost << std::endl;

    return a.exec();
}
