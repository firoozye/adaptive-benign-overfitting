#include "read_csv_func.h"
std::vector<std::vector<std::string>> read_csv_func(const std::string &dir_path) {
   // Open the CSV file
   std::ifstream file(dir_path);
   if (!file.is_open()) {
      std::cerr << "Error: Could not open file " << dir_path << std::endl;
      return {};  // Return an empty vector if file can't be opened
   }

   std::string line;
   std::vector<std::vector<std::string>> data;  // To store the parsed CSV data

   // Read the CSV file line by line
   while (std::getline(file, line)) {
      std::vector<std::string> row;
      std::stringstream ss(line);
      std::string cell;

      // Split each line by commas and store each cell in the row vector
      while (std::getline(ss, cell, ',')) {
         row.push_back(cell);
      }

      // Add the parsed row to the data vector
      data.push_back(row);
   }

   file.close();  // Close the file after reading
   return data;
}
