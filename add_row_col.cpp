double *addRowAndColumnColMajor(double *arr, int &rows, int &cols)
{
    int newRows = rows + 1;
    int newCols = cols + 1;
    int newSize = newRows * newCols;

    double *newArr = new double[newSize](); // Zero-initialize

    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            newArr[j * newRows + i] = arr[j * rows + i]; // Copy column-wise
        }
    }

    delete[] arr;
    return newArr;
}

double *addRowColMajor(double *arr, int &rows, int cols)
{
    int newRows = rows + 1;
    int newSize = newRows * cols;

    double *newArr = new double[newSize]();

    // Copy old data column by column
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            newArr[j * newRows + i] = arr[j * rows + i]; // Copy column-wise
        }
    }

    delete[] arr;
    return newArr;
}

double *addColColMajor(double *arr, int rows, int &cols)
{
    // int newRows = rows + 1;
    int newCols = cols + 1;
    int newSize = rows * newCols;

    double *newArr = new double[newSize]();

    // Copy old data column by column
    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            newArr[j * rows + i] = arr[j * rows + i]; // Copy column-wise
        }
    }

    delete[] arr;
    return newArr;
}

double *deleteRowColMajor(double *arr, int rows, int cols)
{
    int newRows = rows - 1;
    int newSize = newRows * cols;
    double *newArr = new double[newSize]();

    for (int j = 0; j < cols; ++j)
    {
        for (int i = 0; i < newRows; ++i)
        {
            newArr[j * newRows + i] = arr[j * rows + i + 1];
        }
    }

    delete[] arr;
    return newArr;
}

double *deleteColColMajor(double *arr, int rows, int &cols)
{
    int newCols = cols - 1;
    int newSize = rows * newCols;
    double *newArr = new double[newSize]();

    for (int j = 0; j < newCols; ++j)
    {
        for (int i = 0; i < rows; ++i)
        {
            newArr[j * rows + i] = arr[(j + 1) * rows + i];
        }
    }

    delete[] arr;
    return newArr;
}