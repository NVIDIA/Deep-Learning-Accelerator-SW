#
# SPDX-FileCopyrightText: Copyright (c) 2022-2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
"""Simple testbench for all op reconstructions."""
import logging
from collections import defaultdict
from registry import RECONSTRUCTOR_REGISTRY

MAXABSDIFF_THRESHOLDS = defaultdict(lambda: 0.0)
MAXABSDIFF_THRESHOLDS['GRU'] = 1e-6



def log_results(passed_tests, failed_tests):
    num_passed_tests = len(passed_tests)
    num_failed_tests = len(failed_tests)
    num_total_tests = num_passed_tests + num_failed_tests
    passed_percentage = num_passed_tests * 100.0 / num_total_tests
    logging.info(f'{len(passed_tests)}/{num_total_tests} ({passed_percentage}%) tests passed.')
    logging.info(f'{num_passed_tests} passed tests: {passed_tests}')
    logging.info(f'{num_failed_tests} failed tests: {failed_tests}')


def test():
    passed_tests = list()
    failed_tests = list()
    for reconstructor in RECONSTRUCTOR_REGISTRY:
        op = reconstructor()
        threshold = MAXABSDIFF_THRESHOLDS[op.op]
        maxabsdiff = op.test()
        if maxabsdiff <= threshold:
            passed_tests.append(op.op)
        else:
            failed_tests.append(op.op)
            logging.error(f'Test for {op.op} failed')
    log_results(passed_tests, failed_tests)


def main():
    test()


if __name__ == '__main__':
    main()
