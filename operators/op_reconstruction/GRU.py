#
# SPDX-FileCopyrightText: Copyright (c) 2022 - 2023 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: LicenseRef-NvidiaProprietary
#
# NVIDIA CORPORATION, its affiliates and licensors retain all intellectual
# property and proprietary rights in and to this material, related
# documentation and any modifications thereto. Any use, reproduction,
# disclosure or distribution of this material and related documentation
# without an express license agreement from NVIDIA CORPORATION or
# its affiliates is strictly prohibited.
#
"""Reconstruction of a GRU op."""

from logging import info
import onnx_graphsurgeon as gs
import onnx

from common.Operator import Operator
from common.onnxruntime_utils import ONNXRUNTIME_DT_TO_NUMPY_DT
import onnxruntime
import numpy as np

NUM_DIRECTIONS_DICT = dict(forward=1, reverse=1, bidirectional=2)


class GRU(Operator):
    def generate(self, input_shapes, attrs, use_bias=False, use_initial_h=True):
        if not isinstance(input_shapes, list):
            input_shapes = [input_shapes]
        assert len(input_shapes) in {1, 2}
        assert len(input_shapes[0]) == 3
        assert attrs.get('layout', 0) == 1
        batch_size, seq_length, input_size = input_shapes[0]

        dtype = np.float32
        inputs = list()
        hidden_size = attrs.get('hidden_size', 1)
        direction = attrs.get('direction', 'forward')
        assert direction in {'forward'}

        num_directions = NUM_DIRECTIONS_DICT[direction]
        X = gs.Variable(name=self.new_tensor_name(), dtype=dtype, shape=input_shapes[0])
        W_shape = [num_directions, 3 * hidden_size, input_size]
        W = gs.Constant(name=self.new_tensor_name(), values=np.random.rand(*W_shape).astype(dtype))
        R_shape = [num_directions, 3 * hidden_size, hidden_size]
        R = gs.Constant(name=self.new_tensor_name(), values=np.random.rand(*R_shape).astype(dtype))

        inputs = [X, W, R]
        if use_bias:
            B = gs.Constant(name=self.new_tensor_name(),
                            values=np.random.rand(num_directions, 6 * hidden_size).astype(dtype))
            inputs += [B]
        if use_initial_h:
            if not use_bias:
                B = gs.Constant(name=self.new_tensor_name(),
                                values=np.zeros([num_directions, 6 * hidden_size], dtype=dtype))
                inputs += [B]
            sequence_lens = gs.Constant(name=self.new_tensor_name(),
                                        values=np.array([seq_length] * batch_size, dtype=np.int32))
            initial_h_shape = [batch_size, num_directions, hidden_size]
            initial_h = gs.Variable(name=self.new_tensor_name(),
                                    dtype=dtype,
                                    shape=initial_h_shape)
            inputs += [sequence_lens, initial_h]

        outputs = [
            gs.Variable(name='', dtype=dtype),
            gs.Variable(name=self.new_tensor_name(), dtype=dtype)
        ]
        activations = attrs.get('activations', list())
        if len(activations) > 0:
            assert len(activations) == 2  # would be 4 for bidirectional
            attrs['activations'] = activations
        node = gs.Node(op=self.op,
                       inputs=inputs,
                       outputs=outputs,
                       attrs=attrs,
                       name=self.new_node_name())
        graph_inputs = [inp for inp in node.inputs if not isinstance(inp, gs.Constant)]
        graph = gs.Graph(nodes=[node],
                         inputs=graph_inputs,
                         outputs=[node.outputs[-1]],
                         name=node.name)
        graph.name = f'{self.op}_bias{int(use_bias)}_initialh{int(use_initial_h)}_orig'
        return graph

    @staticmethod
    def from2dto4d(data):
        return np.expand_dims(data, [-2, -1])

    @staticmethod
    def add_gru_conv(graph, var_input, prefix, conv_wt_vals, conv_bias_vals, index, dtype):
        assert isinstance(var_input, gs.Variable)
        conv_out = gs.Variable(name=prefix, dtype=dtype)
        conv_Wt = gs.Constant(name=f'{prefix}_wt', values=np.copy(conv_wt_vals[index]))
        inputs = [var_input, conv_Wt]
        if conv_bias_vals[index] is not None:
            inputs.append(gs.Constant(name=f'{prefix}_bias',
                                      values=np.copy(conv_bias_vals[index])))
        conv = gs.Node(op='Conv', inputs=inputs, outputs=[conv_out], name=conv_out.name)
        graph.nodes.append(conv)
        return conv_out

    @staticmethod
    def add_gru_bias(conv_node, conv_bias_vals, index):
        if len(conv_node.inputs) < 3:
            return
        bias = conv_node.inputs[2]
        if conv_bias_vals[index] is not None:
            bias.values += conv_bias_vals[index]

    @staticmethod
    def add_gru_elementwise(graph, ew_op, var_inputs, prefix, dtype, ew_out=None):
        assert isinstance(var_inputs, list)
        ew_out = ew_out or gs.Variable(name=prefix, dtype=dtype)
        ew = gs.Node(op=ew_op, inputs=var_inputs, outputs=[ew_out], name=ew_out.name)
        graph.nodes.append(ew)
        return ew_out

    @staticmethod
    def add_gru_act(graph, act_op, var_input, prefix, dtype):
        assert isinstance(var_input, gs.Variable)
        act_out = gs.Variable(name=prefix, dtype=dtype)
        act = gs.Node(op=act_op, inputs=[var_input], outputs=[act_out], name=act_out.name)
        graph.nodes.append(act)
        return act_out

    @staticmethod
    def add_gru_transpose(graph, var_input, perm, prefix, dtype):
        transpose_out = gs.Variable(name=prefix, dtype=dtype)
        transpose = gs.Node(op='Transpose',
                            inputs=[var_input],
                            outputs=[transpose_out],
                            attrs=dict(perm=perm),
                            name=transpose_out.name)
        graph.nodes.append(transpose)
        return transpose_out

    @staticmethod
    def add_gru_slice(graph, slice_channel_idx, slice_axis, var_input, prefix, dtype):
        assert isinstance(var_input, gs.Variable)
        slice_out = gs.Variable(name=prefix, dtype=dtype)
        starts = gs.Constant(name=f'{prefix}_starts',
                             values=np.array([slice_channel_idx], dtype=np.int64))
        ends = gs.Constant(name=f'{prefix}_ends',
                           values=np.array([slice_channel_idx + 1], dtype=np.int64))
        axes = gs.Constant(name=f'{prefix}_axes', values=np.array([slice_axis], dtype=np.int64))
        steps = gs.Constant(name=f'{prefix}_steps', values=np.array([1], dtype=np.int64))
        slice = gs.Node(op='Slice',
                        inputs=[var_input, starts, ends, axes, steps],
                        outputs=[slice_out],
                        name=slice_out.name)
        graph.nodes.append(slice)
        return slice_out

    @staticmethod
    def add_gru_const_op(graph, const_op, var_input, const_vals, prefix, dtype):
        assert isinstance(var_input, gs.Variable)
        assert isinstance(const_vals, np.ndarray)
        op_out = gs.Variable(name=prefix, dtype=dtype)
        const_vals = const_vals.reshape(1, -1, 1, 1)
        const_input = gs.Constant(name=f'{prefix}_const', values=const_vals.astype(dtype))
        op = gs.Node(op=const_op,
                     inputs=[var_input, const_input],
                     outputs=[op_out],
                     name=op_out.name)
        graph.nodes.append(op)
        return op_out

    @staticmethod
    def reconstruct_step(graph, node, step_idx, total_steps, X_slice, Ht_prev, W_vals, Wb_vals,
                         R_vals, Rb_vals, f, g, dtype, linear_before_reset):
        zt_index_W = 0
        zt_index_R = 0
        rt_index_W = 1
        rt_index_R = 1
        ht_index_W = 2
        ht_index_R = 2
        base_prefix = f'{node.name}_step{step_idx}'
        last_out = node.outputs[-1] if step_idx == total_steps - 1 else None
        if Ht_prev is not None:
            # zt = f(Xt*(Wz^T) + Ht-1*(Rz^T) + Wbz + Rbz)
            zt_X_out = GRU.add_gru_conv(graph, X_slice, f'{base_prefix}_zt_X', W_vals, Wb_vals,
                                        zt_index_W, dtype)
            zt_Ht_out = GRU.add_gru_conv(graph, Ht_prev, f'{base_prefix}_zt_Ht-1', R_vals, Rb_vals,
                                         zt_index_R, dtype)
            zt_add_out = GRU.add_gru_elementwise(graph, 'Add', [zt_X_out, zt_Ht_out],
                                                 f'{base_prefix}_zt_add', dtype)
            zt_act_out = GRU.add_gru_act(graph, f, zt_add_out, f'{base_prefix}_zt_act_f', dtype)
            # rt = f(Xt*(Wr^T) + Ht-1*(Rr^T) + Wbr + Rbr)
            rt_X_out = GRU.add_gru_conv(graph, X_slice, f'{base_prefix}_rt_X', W_vals, Wb_vals,
                                        rt_index_W, dtype)
            rt_Ht_out = GRU.add_gru_conv(graph, Ht_prev, f'{base_prefix}_rt_Ht-1', R_vals, Rb_vals,
                                         rt_index_R, dtype)
            rt_add_out = GRU.add_gru_elementwise(graph, 'Add', [rt_X_out, rt_Ht_out],
                                                 f'{base_prefix}_rt_add', dtype)
            rt_act_out = GRU.add_gru_act(graph, f, rt_add_out, f'{base_prefix}_rt_act_f', dtype)

            # Xt*(Wh^T)
            ht_X_out = GRU.add_gru_conv(graph, X_slice, f'{base_prefix}_X_ht', W_vals, Wb_vals,
                                        ht_index_W, dtype)
            # (rt (.) Ht-1)
            ht_mul_out = GRU.add_gru_elementwise(graph, 'Mul', [rt_act_out, Ht_prev],
                                                 f'{base_prefix}_Ht_mul', dtype)
            # (rt (.) Ht-1)*(Rh^T)
            ht_Ht_out = GRU.add_gru_conv(graph, ht_mul_out, f'{base_prefix}_Ht', R_vals, Rb_vals,
                                         ht_index_R, dtype)
            # Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh
            X_Ht_add_out = GRU.add_gru_elementwise(graph, 'Add', [ht_Ht_out, ht_X_out],
                                                   f'{base_prefix}_X_Ht_add', dtype)
            # linear_before_reset = 0: ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
            X_Ht_act_out = GRU.add_gru_act(graph, g, X_Ht_add_out, f'{base_prefix}_X_Ht_g', dtype)
            # linear_before_reset != 0: ht = g(Xt*(Wh^T) + (rt (.) (Ht-1*(Rh^T) + Rbh)) + Wbh)
            # h_default = self.g(np.dot(x, np.transpose(w_h)) + np.dot(r * H_t, np.transpose(r_h)) + w_bh + r_bh)
            # h_linear = self.g(np.dot(x, np.transpose(w_h)) + r * (np.dot(H_t, np.transpose(r_h)) + r_bh) + w_bh)
            # h = h_linear if self.LBR else h_default
            # (1 - zt)
            zt_sub_out = GRU.add_gru_const_op(graph, 'Add', zt_act_out, np.array([-1]),
                                              f'{base_prefix}_zt_const_add', dtype)
            zt_mul_out = GRU.add_gru_const_op(graph, 'Mul', zt_sub_out, np.array([-1]),
                                              f'{base_prefix}_zt_const_mul', dtype)
            # (1 - zt) (.) ht
            zt_mul0_out = GRU.add_gru_elementwise(graph, 'Mul', [zt_mul_out, X_Ht_act_out],
                                                  f'{base_prefix}_zt_mul0', dtype)
            # zt (.) Ht-1
            zt_mul1_out = GRU.add_gru_elementwise(graph, 'Mul', [zt_act_out, Ht_prev],
                                                  f'{base_prefix}_zt_mul1', dtype)
            # Ht = (1 - zt) (.) ht + zt (.) Ht-1
            out = GRU.add_gru_elementwise(graph,
                                          'Add', [zt_mul0_out, zt_mul1_out],
                                          f'{base_prefix}_out',
                                          dtype,
                                          ew_out=last_out)
        else:
            # zt = f(Xt*(Wz^T) + Wbz + Rbz)
            zt_X_out = GRU.add_gru_conv(graph, X_slice, f'{base_prefix}_zt_X', W_vals, Wb_vals,
                                        zt_index_W, dtype)
            GRU.add_gru_bias(graph.nodes[-1], Rb_vals, zt_index_R)

            zt_act_out = GRU.add_gru_act(graph, f, zt_X_out, f'{base_prefix}_zt_act_f', dtype)

            # Xt*(Wh^T)
            ht_X_out = GRU.add_gru_conv(graph, X_slice, f'{base_prefix}_X_ht', W_vals, Wb_vals,
                                        ht_index_W, dtype)
            GRU.add_gru_bias(graph.nodes[-1], Rb_vals, ht_index_R)

            # ht = g(Xt*(Wh^T) + (rt (.) Ht-1)*(Rh^T) + Rbh + Wbh)
            Ht_act_out = GRU.add_gru_act(graph, g, ht_X_out, f'{base_prefix}_X_Ht_g', dtype)
            # (1 - zt)
            zt_sub_out = GRU.add_gru_const_op(graph, 'Add', zt_act_out, np.array([-1]),
                                              f'{base_prefix}_zt_const_add', dtype)
            zt_mul_out = GRU.add_gru_const_op(graph, 'Mul', zt_sub_out, np.array([-1]),
                                              f'{base_prefix}_zt_const_mul', dtype)
            # (1 - zt) (.) ht
            out = GRU.add_gru_elementwise(graph,
                                          'Mul', [zt_mul_out, Ht_act_out],
                                          f'{base_prefix}_zt_mul0',
                                          dtype,
                                          ew_out=last_out)
        return out

    @staticmethod
    def qualifies_for_reconstruction(node):
        result = node.op == 'GRU'
        SUPPORTED_GRU_ACTIVATIONS = {'Sigmoid', 'Tanh', 'Relu'}
        SUPPORTED_GRU_DIRECTIONS = {'forward'}
        if result:
            attrs = node.attrs
            result &= attrs.get('layout', 0) == 1
            result &= attrs.get('linear_before_reset', 0) == 0
            result &= attrs.get('direction', 'forward') in SUPPORTED_GRU_DIRECTIONS
            result &= attrs.get('activation_alpha', None) is None
            result &= attrs.get('activation_beta', None) is None
            result &= attrs.get('clip', None) is None
            activations = attrs.get('activations', ['Sigmoid', 'Tanh'])
            result &= len(activations) == 2
            if result:
                f, g = activations
                result &= f in SUPPORTED_GRU_ACTIVATIONS
                result &= g in SUPPORTED_GRU_ACTIVATIONS
        return result

    @staticmethod
    def reconstruct(node, graph):
        if GRU.qualifies_for_reconstruction(node):
            info(f'Reconstructing {node.op} node "{node.name}"...')
            X, W, R = node.inputs[:3]

            batch_size, seq_length, input_size = X.shape
            dtype = X.dtype
            attrs = node.attrs
            hidden_size = attrs['hidden_size']
            direction = attrs.get('direction', 'forward')
            activations = attrs.get('activations', ['Sigmoid', 'Tanh'])
            linear_before_reset = attrs.get('linear_before_reset', 0) == 1
            f, g = activations
            num_directions = NUM_DIRECTIONS_DICT[direction]
            Ht_prev_shape = [batch_size, hidden_size, num_directions, 1]
            Ht_prev = None
            if len(node.inputs) > 5:
                Ht_prev = node.inputs[5]
                Ht_prev.shape = Ht_prev_shape
            assert isinstance(W, gs.Constant)
            W_vals = [
                GRU.from2dto4d(W.values[0, x * hidden_size:(x + 1) * hidden_size])
                for x in range(3)
            ]
            assert isinstance(R, gs.Constant)
            R_vals = [
                GRU.from2dto4d(R.values[0, x * hidden_size:(x + 1) * hidden_size])
                for x in range(3)
            ]
            Wb_vals = [None, None, None]
            Rb_vals = [None, None, None]
            if len(node.inputs) > 3:
                assert isinstance(node.inputs[3], gs.Constant)
                B_vals = node.inputs[3].values
                B_dims = B_vals.shape
                assert B_dims[0] == 1
                cutoff = B_dims[1] // 6
                vals = [B_vals[0, x * cutoff:(x + 1) * cutoff] for x in range(6)]
                Wb_vals = vals[:3]
                Rb_vals = vals[3:]

            if seq_length == 1:
                X.shape = [batch_size, input_size, 1, 1]
                out = GRU.reconstruct_step(graph, node, 0, seq_length, X, Ht_prev, W_vals, Wb_vals,
                                           R_vals, Rb_vals, f, g, dtype, linear_before_reset)
            else:
                X.shape = [batch_size, seq_length, input_size, 1]
                X_transposed = GRU.add_gru_transpose(graph, X, (0, 2, 1, 3),
                                                     f'{node.name}_transpose_in', dtype)
                for step_idx in range(seq_length):
                    X_slice = GRU.add_gru_slice(graph, step_idx, 2, X_transposed,
                                                f'{node.name}_slice{step_idx}', dtype)
                    Ht_prev = GRU.reconstruct_step(graph, node, step_idx, seq_length, X_slice,
                                                   Ht_prev, W_vals, Wb_vals, R_vals, Rb_vals, f, g,
                                                   dtype, linear_before_reset)
                out = Ht_prev
            node.inputs.clear()
            node.outputs.clear()
            graph.cleanup().toposort()

    @staticmethod
    def onnx_inference(session, input_data, prev_output, reshape_input_dims=None):
        if not isinstance(input_data, list):
            input_data = [input_data]
        session_inputs = session.get_inputs()
        if reshape_input_dims is not None:
            input_data[0] = np.reshape(input_data, reshape_input_dims)
        if len(session_inputs) == 2:
            input_data = [input_data[0], prev_output]
        assert len(session_inputs) == len(
            input_data), f'{len(session_inputs)} != {len(input_data)}'
        input_dict = dict()
        for tensor, data in zip(session_inputs, input_data):
            assert tensor.type in ONNXRUNTIME_DT_TO_NUMPY_DT, 'No mapping from ONNX RT to NumPy data type detected, you may need to extend it'
            dtype = ONNXRUNTIME_DT_TO_NUMPY_DT[tensor.type]
            if data is None:
                data = np.zeros(tensor.shape)
            input_dict[tensor.name] = data.astype(dtype)
        output_names = [out.name for out in session.get_outputs()]
        outputs = session.run(output_names, input_dict)
        return outputs

    @staticmethod
    def insert_transpose(graph, node, input_direction=True):
        perm = (1, 0, 2)
        tensors = node.inputs if input_direction else node.outputs
        for idx, tensor in enumerate(tensors):
            if not isinstance(tensor, gs.Constant):
                transpose_name = f'{tensor.name}_transpose'
                transpose_tmp = gs.Variable(name=transpose_name, dtype=tensor.dtype)
                if input_direction:
                    node_inputs = [tensor]
                    node_outputs = [transpose_tmp]
                    node.inputs[idx] = transpose_tmp
                else:
                    node_inputs = [transpose_tmp]
                    node_outputs = [tensor]
                    node.outputs[idx] = transpose_tmp
                transpose_node = gs.Node(op='Transpose',
                                         attrs=dict(perm=perm),
                                         name=transpose_name,
                                         inputs=node_inputs,
                                         outputs=node_outputs)
                graph.nodes.append(transpose_node)

    @staticmethod
    def patch_gru_layout(onnx_path_orig):
        # layout 1 is not supported in ONNX Runtime on CPU yet:
        # initialization: /Users/runner/work/1/s/onnxruntime/core/providers/cpu/rnn/deep_cpu_gru.h:55 onnxruntime::DeepCpuGruOp::DeepCpuGruOp(const onnxruntime::OpKernelInfo &) layout_ == 0 was false. Batchwise recurrent operations (layout == 1) are not supported. If you need support create a github issue with justification
        graph = gs.import_onnx(onnx.load(onnx_path_orig))
        patched = False
        patched_path = onnx_path_orig
        for node in graph.nodes:
            if node.op == 'GRU' and node.attrs.get('layout', 0) == 1:
                node.attrs['layout'] = 0
                GRU.insert_transpose(graph, node, input_direction=True)
                GRU.insert_transpose(graph, node, input_direction=False)
                patched = True
        if patched:
            graph.cleanup().toposort()
            patched_path = 'tmp.onnx'
            onnx.save(gs.export_onnx(graph), patched_path)
        return patched_path

    def run_comparison(self,
                       onnx_paths,
                       input_shapes=None,
                       incremental_vals=False,
                       input_data=None,
                       num_iterations=3,
                       index_reconstructed=-1):
        assert len(onnx_paths) == 2
        self.verify_reconstruction(onnx_paths[index_reconstructed][0])
        if input_data is None:
            assert input_shapes is not None
            if incremental_vals:
                input_data = [
                    np.arange(np.product(shape), dtype=np.float32).reshape(shape)
                    for shape in input_shapes
                ]
            else:
                input_data = [np.random.rand(*shape).astype(np.float32) for shape in input_shapes]
        else:
            assert not incremental_vals
            assert input_shapes is None
        index_orig = index_reconstructed - 1
        onnx_paths[index_orig] = list(onnx_paths[index_orig])
        onnx_paths[index_orig][0] = GRU.patch_gru_layout(onnx_paths[index_orig][0])
        session_outputs = list()
        for onnx_file, reshape_input_dims in onnx_paths:
            session = onnxruntime.InferenceSession(onnx_file)
            prev_output = None
            iteration_outputs = list()
            for _ in range(num_iterations):
                session_output = GRU.onnx_inference(session, input_data, prev_output,
                                                    reshape_input_dims)
                prev_output = session_output[0]
                iteration_outputs.append(prev_output)
            session_outputs.append(iteration_outputs)
        for iteration in range(num_iterations):
            maxabsdiff = np.abs(session_outputs[0][iteration].flatten().astype(np.float32) -
                                session_outputs[1][iteration].flatten().astype(np.float32)).max()
        info(
            f'Max absdiff between {onnx_paths[0]} and {onnx_paths[1]} after {num_iterations} iterations: {maxabsdiff}'
        )
        return maxabsdiff

    def test_config(self,
                    input_shapes=None,
                    hidden_size=None,
                    num_iterations=None,
                    use_bias=False,
                    use_initial_h=True,
                    linear_before_reset=0,
                    layout=1):
        attrs = dict(hidden_size=hidden_size,
                     linear_before_reset=linear_before_reset,
                     layout=layout)
        orig_graph = self.generate(input_shapes,
                                   attrs,
                                   use_bias=use_bias,
                                   use_initial_h=use_initial_h)
        orig_path = Operator.save_gs_graph(orig_graph, run_shape_inference=True, opset=14)
        reconstructed_graph = gs.import_onnx(onnx.load(orig_path))
        for node in reconstructed_graph.nodes:
            self.reconstruct(node, reconstructed_graph)
        reconstructed_graph.name = orig_graph.name.replace('_orig', '_reconstructed')
        reconstructed_path = GRU.save_gs_graph(reconstructed_graph, run_shape_inference=True)
        reconstructed_input_shape = reconstructed_graph.inputs[0].shape
        maxabsdiff = self.run_comparison([(orig_path, None),
                                          (reconstructed_path, reconstructed_input_shape)],
                                         input_shapes=input_shapes,
                                         num_iterations=num_iterations)
        return maxabsdiff

    def test(self, input_shapes=None, hidden_size=None, num_iterations=None):
        input_shapes = input_shapes or [(2, 4, 3)]
        hidden_size = hidden_size or 5
        num_iterations = num_iterations or 2
        maxabsdiff = 0.0
        for use_bias in [False, True]:
            for use_initial_h in [False, True]:
                maxabsdiff = max(
                    maxabsdiff,
                    self.test_config(input_shapes=input_shapes,
                                     hidden_size=hidden_size,
                                     num_iterations=num_iterations,
                                     use_bias=use_bias,
                                     use_initial_h=use_initial_h))
        return maxabsdiff


def main():
    op = GRU()
    num_iterations = 3
    op.test(num_iterations=num_iterations)


if __name__ == '__main__':
    main()
