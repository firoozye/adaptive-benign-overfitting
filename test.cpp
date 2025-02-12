#include <iostream>
#include "read_csv_func.cpp"

using namespace std;

int main()
{
    vector<vector<string>> data_set;

    data_set = read_csv_func("daily_vix.csv");

    return 0;
}