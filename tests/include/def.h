#ifndef __TEST_DEF_H__
#define __TEST_DEF_H__

#include <gtest/gtest.h>

#define PRINTF(...)  do { testing::internal::ColoredPrintf(testing::internal::COLOR_GREEN, "[          ] "); testing::internal::ColoredPrintf(testing::internal::COLOR_YELLOW, __VA_ARGS__); } while(0)

#endif
