#ifndef BLADE_LOGGER_HH
#define BLADE_LOGGER_HH

#include <iostream>

#include <fmt/format.h>
#include <fmt/color.h>

#define __FILENAME__ (strrchr(__FILE__, '/') ? strrchr(__FILE__, '/') + 1 : __FILE__)

#define BL_LOG_HEAD_DECR fmt::emphasis::bold
#define BL_LOG_HEAD_NAME fmt::format(BL_LOG_HEAD_DECR, "BLADE ")
#define BL_LOG_HEAD_FILE fmt::format(BL_LOG_HEAD_DECR, "[{}@{}] ", __FILENAME__, __LINE__)
#define BL_LOG_HEAD_TRACE fmt::format(BL_LOG_HEAD_DECR, "[TRACE] ")
#define BL_LOG_HEAD_DEBUG fmt::format(BL_LOG_HEAD_DECR, "[DEBUG] ")
#define BL_LOG_HEAD_WARN fmt::format(BL_LOG_HEAD_DECR, "[WARN]  ")
#define BL_LOG_HEAD_INFO fmt::format(BL_LOG_HEAD_DECR, "[INFO]  ")
#define BL_LOG_HEAD_ERROR fmt::format(BL_LOG_HEAD_DECR, "[ERROR] ")
#define BL_LOG_HEAD_FATAL fmt::format(BL_LOG_HEAD_DECR, "[FATAL] ")

#define BL_LOG_HEAD_SEPR fmt::format(BL_LOG_HEAD_DECR, "| ")

#if !defined(BL_TRACE) || !defined(NDEBUG)
#define BL_TRACE(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_TRACE << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::white), __VA_ARGS__) << std::endl;
#endif

#if !defined(BL_DEBUG) || defined(NDEBUG)
#define BL_DEBUG(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_DEBUG << BL_LOG_HEAD_SEPR << \
        fmt::format(fg(fmt::color::orange), __VA_ARGS__) << std::endl;
#endif

#if !defined(BL_WARN) || defined(NDEBUG)
#define BL_WARN(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_WARN << BL_LOG_HEAD_SEPR << \
        fmt::format(fg(fmt::color::yellow), __VA_ARGS__) << std::endl;
#endif

#if !defined(BL_INFO) || defined(NDEBUG)
#define BL_INFO(...) std::cout << BL_LOG_HEAD_NAME << BL_LOG_HEAD_INFO << BL_LOG_HEAD_SEPR << \
        fmt::format(fg(fmt::color::cyan), __VA_ARGS__) << std::endl;
#endif

#if !defined(BL_ERROR) || defined(NDEBUG)
#define BL_ERROR(...) std::cerr << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_ERROR << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::red), __VA_ARGS__) << std::endl;
#endif

#if !defined(BL_FATAL) || defined(NDEBUG)
#define BL_FATAL(...) std::cerr << BL_LOG_HEAD_NAME << BL_LOG_HEAD_FILE << BL_LOG_HEAD_FATAL << \
        BL_LOG_HEAD_SEPR << fmt::format(fg(fmt::color::magenta), __VA_ARGS__) << std::endl;
#endif

#endif
