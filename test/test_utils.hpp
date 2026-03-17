/* SPDX-License-Identifier: MIT
 * Copyright (c) 2024 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <iostream>

/*
 * Colored output helpers for mj_kdl_wrapper test binaries.
 *
 * Each macro accepts a stream expression as its argument, e.g.:
 *   TEST_FAIL("expected 7 joints, got " << n);
 *   TEST_PASS("gravity compensation");
 */

// clang-format off
#define TEST_PASS(msg) (std::cout << "\033[32m[ PASS ]\033[0m " << msg << "\n")
#define TEST_FAIL(msg) (std::cerr << "\033[31m[ FAIL ]\033[0m " << msg << "\n")
#define TEST_WARN(msg) (std::cout << "\033[33m[ WARN ]\033[0m " << msg << "\n")
#define TEST_INFO(msg) (std::cout << "\033[36m[ INFO ]\033[0m " << msg << "\n")
#define TEST_SKIP(msg) (std::cout << "\033[90m[ SKIP ]\033[0m " << msg << "\n")
// clang-format on

/*
 * Structured test blocks.  BEGIN_TEST announces the case with [ RUN  ]; END_TEST
 * prints [ PASS ] on the successful path.  Use TEST_FAIL + return 1 inside to
 * fail early ([ PASS ] will not be printed in that case).
 *
 *   BEGIN_TEST("FK at home pose")
 *       if (fk.JntToCart(q, f) < 0) { TEST_FAIL("FK failed"); cleanup; return 1; }
 *       TEST_INFO("pos: [" << f.p.x() << "]");
 *   END_TEST
 */
// clang-format off
#define BEGIN_TEST(name) \
    { \
        const char *_test_name_ = (name); \
        std::cout << "[ RUN  ] " << _test_name_ << "\n";

#define END_TEST \
        std::cout << "\033[32m[ PASS ]\033[0m " << _test_name_ << "\n"; \
    }
// clang-format on
