// SPDX-FileCopyrightText: © 2025 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "move_program_factory.hpp"

#include <tt-metalium/experimental/program_descriptor_patching.hpp>

#include "ttnn/operations/data_movement/copy/device/copy_device_operation.hpp"
#include "ttnn/operations/data_movement/copy/device/copy_device_operation_types.hpp"

namespace ttnn::prim {

tt::tt_metal::ProgramDescriptor MoveProgramFactory::create_descriptor(
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value) {
    const Tensor& input = tensor_args.input_tensor;
    Tensor& output = tensor_return_value;
    using copy_attrs_t = CopyDeviceOperation::operation_attributes_t;
    using copy_args_t = CopyDeviceOperation::tensor_args_t;

    const copy_attrs_t copy_attrs{
        operation_attributes.output_mem_config, output.dtype(), operation_attributes.backwards};
    const copy_args_t copy_args{input, std::make_optional(output)};

    return CopyDeviceOperation::SameMemoryConfig::create_descriptor(copy_attrs, copy_args, output);
}

void MoveProgramFactory::override_runtime_arguments(
    tt::tt_metal::Program& program,
    const MoveOperationAttributes& operation_attributes,
    const MoveTensorArgs& tensor_args,
    Tensor& tensor_return_value,
    const std::optional<ttnn::MeshCoordinate>& /*mesh_dispatch_coordinate*/) {
    auto desc = MoveProgramFactory::create_descriptor(operation_attributes, tensor_args, tensor_return_value);
    tt::tt_metal::apply_descriptor_runtime_args(program, desc);
}

}  // namespace ttnn::prim
