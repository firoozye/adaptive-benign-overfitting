#pragma once

#pragma once

#ifdef __cplusplus
extern "C" {
#endif

#include <cblas.h>
#include <lapacke.h>

#ifdef __cplusplus
}
#endif

#include <cstdlib>
#include <cstring>
// Function declarations for matrix operations in column-major format
double *addRowAndColumnColMajor(double *arr, int &rows, int &cols);
double *addRowColMajor(double *arr, int &rows, int cols);
double *addColColMajor(double *arr, int rows, int &cols);
double *deleteRowColMajor(double *arr, int rows, int cols);
double *deleteColColMajor(double *arr, int rows, int &cols);
