
// logger.h
#pragma once

#include <chrono>
#include <ctime>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

#ifndef LOG_LEVEL
#define LOG_LEVEL 1 // Default to ERROR if not specified
#endif
// In logger.h
// later can put log rotation, compression, cleanup rules etc.

namespace {
inline std::string getTimestamp() {
  auto now = std::chrono::system_clock::now();
  auto now_time = std::chrono::system_clock::to_time_t(now);
  std::ostringstream ts;
  struct tm *timeinfo = std::localtime(&now_time);
  ts << "[" << std::put_time(timeinfo, "%Y-%m-%d %H:%M:%S") << "] ";
  return ts.str();
}
} // namespace

class Logger {
public:
  static Logger &getInstance() {
    static Logger instance;
    return instance;
  }

  // Add this method to write logs
  static void log(const std::string &message) {
    if (getInstance().m_file.is_open()) {
      getInstance().m_file << message << std::endl;
    } else {
      std::cout << message << std::endl;
    }
  }

private:
  Logger() {
    auto now = std::chrono::system_clock::now();
    auto now_time = std::chrono::system_clock::to_time_t(now);
    std::ostringstream filename;

    std::filesystem::create_directories("logs");
    filename << "logs/logfile_"
             << std::put_time(std::localtime(&now_time), "%Y%m%d_%H%M%S")
             << ".txt";

    m_file.open(filename.str(), std::ios::app);
  }

  // Rest of your private members stay the same...
  std::ofstream m_file;
  // ... destructors and deleted functions
};

#if LOG_LEVEL >= 4
#define LOG_DEBUG(...)                                                         \
  do {                                                                         \
    std::ostringstream ss;                                                     \
    ss << getTimestamp() << "[DEBUG] [" << __FILE__ << ":" << __LINE__ << "] " \
       << __VA_ARGS__;                                                         \
    Logger::log(ss.str());                                                     \
  } while (0)
#else
#define LOG_DEBUG(...) ((void)0)
#endif

#if LOG_LEVEL >= 3
#define LOG_INFO(...)                                                          \
  do {                                                                         \
    std::ostringstream ss;                                                     \
    ss << getTimestamp() << "[INFO] [" << __FILE__ << ":" << __LINE__ << "] "  \
       << __VA_ARGS__;                                                         \
    Logger::log(ss.str());                                                     \
  } while (0)
#else
#define LOG_INFO(...) ((void)0)
#endif

#if LOG_LEVEL >= 2
#define LOG_WARNING(...)                                                       \
  do {                                                                         \
    std::ostringstream ss;                                                     \
    ss << getTimestamp() << "[WARNING] [" << __FILE__ << ":" << __LINE__       \
       << "] " << __VA_ARGS__;                                                 \
    Logger::log(ss.str());                                                     \
  } while (0)
#else
#define LOG_WARNING(...) ((void)0)
#endif

#if LOG_LEVEL >= 1
#define LOG_ERROR(...)                                                         \
  do {                                                                         \
    std::ostringstream ss;                                                     \
    ss << getTimestamp() << "[ERROR] [" << __FILE__ << ":" << __LINE__ << "] " \
       << __VA_ARGS__;                                                         \
    Logger::log(ss.str());                                                     \
  } while (0)
#else
#define LOG_ERROR(...) ((void)0)
#endif
// logger.h   with no need for logger.cc any longer
