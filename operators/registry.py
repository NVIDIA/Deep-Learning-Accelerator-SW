#
# SPDX-FileCopyrightText: Copyright (c) 2022 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: MIT
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Registry for op reconstructions."""

from op_reconstruction import *

RECONSTRUCTOR_REGISTRY = [
    Neg,
    DepthToSpace,
    SpaceToDepth,
    Xor,
    And,
    Or,
    Not,
    Gather,
    GRU,
    ScatterElements,
    Where
]

RECONSTRUCTOR_REGISTRY_DICT = dict()
for reconstructor in RECONSTRUCTOR_REGISTRY:
    RECONSTRUCTOR_REGISTRY_DICT[reconstructor.op_to_reconstruct()] = reconstructor
