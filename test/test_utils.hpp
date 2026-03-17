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
