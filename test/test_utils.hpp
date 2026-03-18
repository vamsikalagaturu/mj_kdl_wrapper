/* SPDX-License-Identifier: MIT
 * Copyright (c) 2024 Vamsi Kalagaturu
 * See LICENSE for details. */

#pragma once

#include <iostream>

/*
 * Colored output helpers for mj_kdl_wrapper test binaries.
 *
 * Each macro accepts a stream expression as its argument, e.g.:
 *   TEST_INFO("nq=" << n);
 *   TEST_WARN("IK did not converge");
 */

// clang-format off
#define TEST_WARN(msg) (std::cout << "\033[33m[ WARN ]\033[0m " << msg << "\n")
#define TEST_INFO(msg) (std::cout << "\033[36m[ INFO ]\033[0m " << msg << "\n")
// clang-format on
