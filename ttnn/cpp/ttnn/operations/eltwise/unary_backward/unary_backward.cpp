// SPDX-FileCopyrightText: © 2024 Tenstorrent USA, Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <numbers>
#include <utility>
#include "ttnn/operations/eltwise/unary_backward/unary_backward.hpp"
#include "ttnn/operations/data_movement/bcast/bcast.hpp"
#include <tt-metalium/constants.hpp>
#include "ttnn/operations/data_movement/common/common.hpp"
#include "ttnn/operations/eltwise/unary/unary.hpp"
#include "ttnn/operations/eltwise/binary/binary.hpp"
#include "ttnn/operations/moreh/moreh_sum/moreh_sum.hpp"
#include "ttnn/operations/data_movement/permute/permute.hpp"
#include "ttnn/operations/data_movement/pad/pad.hpp"
#include "ttnn/operations/data_movement/slice/slice.hpp"
#include "ttnn/operations/data_movement/tilize_with_val_padding/tilize_with_val_padding.hpp"
#include "ttnn/operations/data_movement/untilize/untilize.hpp"
#include "ttnn/operations/reduction/prod/prod.hpp"
#include "ttnn/operations/eltwise/ternary/ternary.hpp"
#include "ttnn/operations/eltwise/unary/unary_composite.hpp"
#include "ttnn/operations/creation/creation.hpp"
#include "ttnn/operations/eltwise/complex/complex.hpp"
#include "gelu_bw/device/gelu_bw_device_operation.hpp"
#include "ttnn/operations/eltwise/complex_unary/complex_unary.hpp"
#include "ttnn/operations/eltwise/complex_binary/device/complex_binary_op.hpp"
#include "ttnn/operations/reduction/generic/generic_reductions.hpp"
#include "ttnn/operations/eltwise/binary/binary_composite.hpp"
#include "tools/profiler/op_profiler.hpp"
#include "tanh_bw/device/tanh_bw_device_operation.hpp"
#include "ttnn/tensor/tensor_utils.hpp"
#include <tt-metalium/hal.hpp>

namespace ttnn {

std::vector<Tensor> clamp_bw(
    const Tensor& grad,
    const Tensor& input,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_mem_config);
        Tensor result = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    if (!min.has_value()) {
        Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_mem_config);
        Tensor result = ttnn::multiply(grad, maxT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_memory_config);
    Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_memory_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_memory_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> clamp_bw(
    const Tensor& grad,
    const Tensor& input,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    TT_FATAL((max.has_value() || min.has_value()), "Only one of 'min' or 'max' can be None. Please provide one value");
    if (!max.has_value()) {
        Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_mem_config);
        Tensor in_grad = ttnn::multiply(grad, minT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(in_grad);
        return grad_tensor;
    }
    if (!min.has_value()) {
        Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_mem_config);
        Tensor in_grad = ttnn::multiply(grad, maxT, std::nullopt, output_mem_config);
        grad_tensor.emplace_back(in_grad);
        return grad_tensor;
    }
    Tensor minT = ttnn::ge(input, min.value(), std::nullopt, output_memory_config);
    Tensor maxT = ttnn::le(input, max.value(), std::nullopt, output_memory_config);
    Tensor result = ttnn::logical_and(minT, maxT, std::nullopt, output_memory_config);
    result = ttnn::multiply(grad, result, std::nullopt, output_memory_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> clip_bw(
    const Tensor& grad,
    const Tensor& input,
    std::optional<float> min,
    std::optional<float> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return clamp_bw(grad, input, min, max, output_mem_config);
}

std::vector<Tensor> clip_bw(
    const Tensor& grad,
    const Tensor& input,
    std::optional<Tensor> min,
    std::optional<Tensor> max,
    const std::optional<MemoryConfig>& output_mem_config) {
    return clamp_bw(grad, input, std::move(min), std::move(max), output_mem_config);
}

// Hardtanh
// result: torch.where((input <= min) | (input >= max), 0.0, grad)
std::vector<Tensor> hardtanh_bw(
    const Tensor& grad,
    const Tensor& input,
    float min,
    float max,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::where(
        ttnn::le(input, min, std::nullopt, output_mem_config),
        0.f,
        ttnn::where(ttnn::ge(input, max, std::nullopt, output_mem_config), 0.f, grad),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// threshold
// if input <= threshold = 0 else grad
std::vector<Tensor> threshold_bw(
    const Tensor& grad,
    const Tensor& input,
    float threshold,
    float /*value*/,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::where(
        ttnn::gtz(ttnn::add(input, -threshold, std::nullopt, output_mem_config), output_mem_config),
        grad,
        0.f,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Softplus
std::vector<Tensor> softplus_bw(
    const Tensor& grad,
    const Tensor& input,
    float beta,
    float threshold,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor mul_input_beta = ttnn::multiply(input, beta, std::nullopt, output_mem_config);
    Tensor exp_beta_self = ttnn::exp(mul_input_beta, false, output_mem_config);
    Tensor sub_result = ttnn::add(mul_input_beta, -threshold, std::nullopt, output_mem_config);
    Tensor temp = ttnn::multiply(
        ttnn::multiply(grad, exp_beta_self, std::nullopt, output_mem_config),
        ttnn::reciprocal(ttnn::add(exp_beta_self, 1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    Tensor grad_result = ttnn::where(ttnn::gtz(sub_result, output_mem_config), grad, temp, output_mem_config);
    mul_input_beta.deallocate();
    exp_beta_self.deallocate();
    sub_result.deallocate();
    temp.deallocate();
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> rdiv_bw(
    const Tensor& grad,
    const Tensor& input,
    float scalar,
    const std::optional<std::string>& rounding_mode,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    TT_FATAL(
        (rounding_mode == std::nullopt || rounding_mode == "trunc" || rounding_mode == "floor"),
        "Incorrect rounding mode (expected None, 'trunc', or 'floor')");
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    if (rounding_mode == std::nullopt) {
        Tensor result = ttnn::where(
            ttnn::nez(input),
            ttnn::multiply(
                ttnn::neg(grad, output_mem_config),
                (ttnn::multiply(
                    ttnn::reciprocal(ttnn::square(input, output_mem_config)), scalar, std::nullopt, output_mem_config)),
                std::nullopt,
                output_mem_config),
            t_nan,
            output_mem_config);
        if (scalar > 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::ltz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::gtz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                -t_inf,
                result,
                output_mem_config);
        } else if (scalar < 0) {
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::ltz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                -t_inf,
                result,
                output_mem_config);
            result = ttnn::where(
                ttnn::logical_and(
                    ttnn::eqz(input, output_mem_config),
                    ttnn::gtz(grad, output_mem_config),
                    std::nullopt,
                    output_mem_config),
                t_inf,
                result,
                output_mem_config);
        }
        grad_tensor.emplace_back(result);
    } else {
        Tensor result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
        grad_tensor.emplace_back(result);
    }
    return grad_tensor;
}

// unary_pow:
// grad_input = grad * exponent * torch.pow(input, exponent - 1)
std::vector<std::optional<Tensor>> pow_bw(
    const Tensor& grad,
    const Tensor& input,
    float exponent,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    const float ZERO_THRESHOLD = std::numeric_limits<float>::epsilon() * 10.0f;
    TT_FATAL(exponent >= 0.0, "negative exponents are not supported; use recip(pow(input,abs(exponent)))");
    if (std::abs(exponent) < ZERO_THRESHOLD) {
        input_grad = ttnn::zeros_like(input);
        grad_tensor.emplace_back(input_grad);
        return grad_tensor;
    }

    Tensor power_input = ttnn::pow(input, std::fabs(exponent - 1.0f), output_mem_config);
    if (exponent < 1.0f) {
        power_input = ttnn::reciprocal(power_input, output_mem_config);
    }

    Tensor result = ttnn::multiply(power_input, exponent, std::nullopt, output_mem_config);
    power_input.deallocate();
    Tensor final_result = ttnn::multiply(result, grad, std::nullopt, output_mem_config);
    result.deallocate();
    // Handle negative inputs by returning infinity
    where(ttnn::lez(input), std::numeric_limits<float>::infinity(), final_result, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> exp_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor exp_result = ttnn::exp(input, false, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config, input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> tanh_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    DataType output_dtype = input.dtype();
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    auto result_tensor = ttnn::operations::unary_backward::tanh_bw::launch_tanh_bw(
        grad, input, output_dtype, output_memory_config, input_grad);
    grad_tensor.emplace_back(result_tensor);
    return grad_tensor;
}

std::vector<std::optional<Tensor>> sqrt_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> grad_tensor;

    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    ttnn::sqrt(input, false, output_mem_config, input_grad);
    ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(input_grad.value(), 2.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config,
        input_grad);
    where(ttnn::lez(input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        input_grad.value(),
        output_mem_config,
        input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        input_grad.value(),
        output_mem_config,
        input_grad);
    grad_tensor.emplace_back(input_grad);
    return grad_tensor;
}

std::vector<Tensor> multigammaln_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor digamma_result =
        ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    Tensor digamma_result_2 = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -0.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);

    Tensor grad_result = ttnn::add(digamma_result, digamma_result_2, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -1.0f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    digamma_result = ttnn::multiply(
        grad,
        ttnn::digamma(ttnn::add(input, -1.5f, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = ttnn::add(grad_result, digamma_result, std::nullopt, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> lgamma_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(grad, ttnn::digamma(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> frac_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.emplace_back(grad);
    return grad_tensor;
}

std::vector<Tensor> trunc_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// return: grad_output * (max_deriv - sign * (z / (1 + z)))
// z = exp(-abs(input))
std::vector<Tensor> log_sigmoid_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor max_deriv = ttnn::where(ttnn::ltz(input, output_mem_config), 1.f, 0.f, output_mem_config);
    Tensor in_sign = ttnn::where(ttnn::ltz(input, output_mem_config), 1.f, -1.f, output_mem_config);
    Tensor in_abs = ttnn::abs(input, output_mem_config);
    Tensor z = ttnn::exp(ttnn::neg(in_abs, output_mem_config), false, output_mem_config);

    Tensor mul_z = ttnn::multiply(
        z,
        ttnn::reciprocal((ttnn::add(z, 1.0f, std::nullopt, output_mem_config)), output_mem_config),
        std::nullopt,
        output_mem_config);

    Tensor mul_sign = ttnn::multiply(in_sign, mul_z, std::nullopt, output_mem_config);
    Tensor sub_max = ttnn::subtract(max_deriv, mul_sign, std::nullopt, output_mem_config);

    Tensor grad_result = ttnn::multiply(grad, sub_max, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> fill_zero_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

//   name: i0(Tensor self) -> Tensor
//   self: grad * at::special_i1(self)
//   result: auto_element_wise
std::vector<Tensor> i0_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor i1_input = ttnn::i1(input, output_mem_config);
    Tensor result = ttnn::multiply(grad, i1_input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> tan_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tan_result = ttnn::tan(input, output_mem_config);
    Tensor result = ttnn::multiply(
        grad,
        ttnn::add(ttnn::square(tan_result, output_mem_config), 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// grad(sigmoid) = grad*(1 - sigmoid(x))*sigmoid(x)
std::vector<Tensor> sigmoid_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    grad_tensor.reserve(1);
    Tensor sig_result = ttnn::sigmoid(
        input,
        (int)ttnn::operations::unary::VecMode::RC,
        ttnn::operations::unary::SigmoidMode::ACCURATE,
        output_mem_config);
    Tensor rsub_term = ttnn::rsub(sig_result, 1.0f, std::nullopt, output_mem_config);
    Tensor prod_term_1 = ttnn::multiply(sig_result, rsub_term, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(prod_term_1, grad, std::nullopt, output_mem_config));
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> rsqrt_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad);
    }
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");

    ttnn::rsqrt(input, false, output_mem_config, input_grad);
    ttnn::power(input_grad.value(), 3, output_mem_config, input_grad);
    ttnn::multiply(
        ttnn::multiply(grad, input_grad.value(), std::nullopt, output_mem_config),
        -0.5f,
        std::nullopt,
        output_mem_config,
        input_grad);
    where(ttnn::eqz(input, output_mem_config), t_inf, input_grad.value(), output_mem_config, input_grad);
    where(ttnn::ltz(input, output_mem_config), t_nan, input_grad.value(), output_mem_config, input_grad);
    where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        input_grad.value(),
        output_mem_config,
        input_grad);

    result.emplace_back(input_grad);
    return result;
}

std::vector<std::optional<Tensor>> neg_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    input_grad = input_grad.value_or(ttnn::empty_like(input));
    result[0] = ttnn::neg(grad, output_mem_config, input_grad);
    return result;
}

std::vector<Tensor> relu_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(ttnn::gtz(input, output_mem_config), grad, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// fill_bw:
// name: fill.Scalar(Tensor self, Scalar value) -> Tensor
// self: zeros_like(grad)
// result: at::fill(self_t, 0)
std::vector<std::optional<Tensor>> fill_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    const std::optional<Tensor>& input_grad) {
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    std::vector<std::optional<Tensor>> result = {std::nullopt};
    result[0] = input_grad.has_value()
                    ? ttnn::zeros_like(grad, std::nullopt, std::nullopt, std::nullopt, std::nullopt, input_grad)
                    : ttnn::zeros_like(grad);
    return result;
}

std::vector<Tensor> hardsigmoid_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::where(
        ttnn::logical_or(
            ttnn::le(input, -3.0f, std::nullopt, output_mem_config),
            ttnn::ge(input, 3.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        0.f,
        ttnn::multiply(grad, 1.0f / 6.0f),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// name: cos(Tensor self) -> Tensor
// self: grad * -self.sin()
std::vector<Tensor> cos_bw(
    const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        grad,
        (ttnn::neg(ttnn::sin(input_tensor, output_mem_config), output_mem_config)),
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> acosh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor in_sq = ttnn::square(input, output_mem_config);
    Tensor in_rsqrt =
        ttnn::rsqrt(ttnn::subtract(in_sq, 1.0f, std::nullopt, output_mem_config), false, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    float t_nan = tt::tt_metal::hal::get_nan();
    float t_inf = tt::tt_metal::hal::get_inf();

    Tensor check_condition =
        ttnn::multiply(ttnn::signbit(grad, output_mem_config), -1.0f, std::nullopt, output_mem_config);

    grad_a = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(in_sq, 1.0f, std::nullopt, output_mem_config),
            ttnn::logical_and(
                ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
                ttnn::eqz(grad, output_mem_config),
                std::nullopt,
                output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        ttnn::where(
            ttnn::logical_and(
                ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
                ttnn::ge(input, -1.0f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(
                ttnn::add(
                    check_condition, ttnn::eqz(check_condition, output_mem_config), std::nullopt, output_mem_config),
                t_inf,
                std::nullopt,
                output_mem_config),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// # - name: acos(Tensor self) -> Tensor
// #   self: grad * -((-self * self + 1).rsqrt())
std::vector<Tensor> acos_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor neg_in = ttnn::neg(input, output_mem_config);
    Tensor in_rsqrt = ttnn::rsqrt(
        ttnn::add(
            ttnn::multiply(neg_in, input, std::nullopt, output_mem_config), 1.0f, std::nullopt, output_mem_config),
        false,
        output_mem_config);
    in_rsqrt = ttnn::neg(in_rsqrt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, in_rsqrt, std::nullopt, output_mem_config);
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_or(
            ttnn::lt(input, -1.0f, std::nullopt, output_mem_config),
            ttnn::gt(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::eq(input, -1.0f, std::nullopt, output_mem_config),
        t_inf,
        where(ttnn::eq(input, 1.0f, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> atan_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> rad2deg_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_180_PI = 180 / M_PI;
    Tensor grad_result = ttnn::multiply(grad, M_180_PI, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> logit_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(
            input, ttnn::rsub(input, 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config)),
        std::nullopt,
        output_mem_config);
    Tensor status = ttnn::logical_and(
        ttnn::ge(input, 0.0f, std::nullopt, output_mem_config),
        ttnn::le(input, 1.0f, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(ttnn::eq(status, 1.0f, std::nullopt, output_mem_config), grad_result, std::nanf(""));
    grad_result = where(
        ttnn::logical_or(
            ttnn::eq(input, 0.0f, std::nullopt, output_mem_config),
            ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        ttnn::multiply(
            ttnn::sign(grad, output_mem_config),
            std::numeric_limits<float>::infinity(),
            std::nullopt,
            output_mem_config),
        grad_result,
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}
// square
// result:  2 * input * grad_data
std::vector<Tensor> square_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = ttnn::multiply(
        ttnn::multiply(grad, 2.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> hardshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor hardshrink_result = ttnn::hardshrink(input_tensor, lambd, output_mem_config);
    Tensor result = where(ttnn::eqz(hardshrink_result, output_mem_config), 0.0f, grad, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// softshrink
//  result: torch.where(self < -lambd, grad, torch.where(self > lambd, grad, torch.tensor(0.0)))
std::vector<Tensor> softshrink_bw(
    const Tensor& grad, const Tensor& input_tensor, float lambd, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(input_tensor, -lambd, std::nullopt, output_mem_config),
            ttnn::gt(input_tensor, lambd, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        0.f,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Leaky_Relu
// result: torch.where(self > 0, grad_output, grad_output * negative_slope)
std::vector<Tensor> leaky_relu_bw(
    const Tensor& grad,
    const Tensor& input,
    float negative_slope,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad,
        ttnn::multiply(grad, negative_slope, std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// ELU
// result : grad * (torch.where(input > 0, 1, alpha * torch.exp(input)))
std::vector<Tensor> elu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad,
        ttnn::multiply(
            grad,
            ttnn::multiply(ttnn::exp(input, false, output_mem_config), alpha, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Celu
// result: torch.where((input > 0), grad, grad * torch.exp(input / alpha))
std::vector<Tensor> celu_bw(
    const Tensor& grad, const Tensor& input, float alpha, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float div_alpha = (1.0 / alpha);
    Tensor div_result = ttnn::multiply(input, div_alpha, std::nullopt, output_mem_config);
    Tensor exp_result = ttnn::exp(div_result, false, output_mem_config);
    Tensor grad_result = where(
        ttnn::gt(input, 0.0f, std::nullopt, output_mem_config),
        grad,
        ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> rpow_bw(
    const Tensor& grad, const Tensor& input, float exponent, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    Tensor grad_result = ttnn::zeros_like(input, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    if (exponent != 0.0) {
        grad_result = ttnn::multiply(
            grad,
            ttnn::multiply(pow(input, exponent - 1, output_mem_config), exponent, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config);
        grad_result = ttnn::where(ltz(input, output_mem_config), t_nan, grad_result, output_mem_config);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<Tensor> floor_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> round_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor t_zero = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(t_zero);
    return grad_tensor;
}

std::vector<Tensor> log_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            std::nanf(""),
            ttnn::multiply(
                ttnn::sign(grad, output_mem_config),
                std::numeric_limits<float>::infinity(),
                std::nullopt,
                output_mem_config),
            output_mem_config),
        grad_a,
        output_mem_config));
    return grad_tensor;
}

std::vector<Tensor> relu6_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(ttnn::le(input, 0.0f, std::nullopt, output_mem_config), 0.0f, 6.0f, output_mem_config);
    grad_result = where(
        ttnn::logical_and(
            ttnn::gtz(input, output_mem_config),
            ttnn::lt(input, 6.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        grad,
        grad_result,
        output_mem_config);
    grad_result = where(ttnn::ge(input, 6.0f, std::nullopt, output_mem_config), 0.0f, grad_result, output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Silu
// result:  grad * sigmoid_result * (1 + input * (1 - sigmoid_result))
std::vector<std::optional<Tensor>> silu_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result = {std::nullopt};

    input_grad = input_grad.value_or(ttnn::empty_like(input));
    Tensor sigmoid_res = ttnn::sigmoid(
        input,
        (int)ttnn::operations::unary::VecMode::RC,
        ttnn::operations::unary::SigmoidMode::ACCURATE,
        output_mem_config);
    Tensor grad_sigmoid = ttnn::multiply(grad, sigmoid_res, std::nullopt, output_mem_config);
    Tensor add_sub = ttnn::add(
        ttnn::multiply(
            ttnn::rsub(sigmoid_res, 1.0f, std::nullopt, output_mem_config), input, std::nullopt, output_mem_config),
        1.0f,
        std::nullopt,
        output_mem_config);
    ttnn::multiply(grad_sigmoid, add_sub, std::nullopt, output_mem_config, input_grad);

    result[0] = input_grad;
    return result;
}

// Selu
// result:  torch.where(input > 0, grad * lambd, grad * lambd * alpha * torch.exp(input))
std::vector<Tensor> selu_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_lambd = ttnn::multiply(grad, 1.0507f, std::nullopt, output_mem_config);
    Tensor grad_result = where(
        ttnn::gtz(input, output_mem_config),
        grad_lambd,
        ttnn::multiply(
            ttnn::multiply(grad_lambd, 1.673260f, std::nullopt, output_mem_config),
            ttnn::exp(input, false, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// Hardswish
// result: torch.where(input < -3,0.0,torch.where(input <= 3, grad * ((input / 3) + 0.5), grad),)
std::vector<Tensor> hardswish_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_result = where(
        ttnn::lt(input, -3.0f, std::nullopt, output_mem_config),
        0.f,
        where(
            ttnn::le(input, 3.0f, std::nullopt, output_mem_config),
            ttnn::multiply(
                grad,
                ttnn::add(
                    ttnn::multiply(input, 0.3333f, std::nullopt, output_mem_config),
                    0.5f,
                    std::nullopt,
                    output_mem_config),
                std::nullopt,
                output_mem_config),
            grad),
        output_mem_config);

    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// tanhshrink
// result:  torch.square(torch.tanh(input)) * grad_data
std::vector<Tensor> tanhshrink_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor tanh_res = ttnn::square(ttnn::tanh(input, output_mem_config), output_mem_config);
    grad_tensor.emplace_back(ttnn::multiply(grad, tanh_res, std::nullopt, output_mem_config));
    return grad_tensor;
}

# atanh_bw: NaN for a zero incoming gradient — change set

Base: `tenstorrent/tt-metal` @ `587a4f30937e8bd5eea684434ef985d32486fb55` (main, 2026-08-06 14:24 UTC).
Verified identical at `EazyHood/tt-metal` @ `62a086c5` for both files touched.

Branch name to use: `fix/atanh-bw-zero-grad`
PR title: `[Bug fix] fix(eltwise): atanh_bw returns NaN for a zero incoming gradient`

---

## (a) Files and exact lines to touch

| # | File | Lines | Required? |
|---|------|-------|-----------|
| 1 | `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp` | 951-990 (function `atanh_bw`); the defect is on **966** | yes |
| 2 | `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_atanh.py` | import on line 8, plus a new test function | yes |

Nothing else. No header change is needed: `<tt-metalium/hal.hpp>` is already included at
`unary_backward.cpp:32`, and `acosh_bw` at line 580-581 already calls `tt::tt_metal::hal::get_nan()`
and `get_inf()` from this same translation unit.

Do **not** touch `ttnn/ttnn/operations/unary_backward.py`. The registered golden
(lines 85-90, `torch.atanh` through autograd) is correct and is the thing this PR makes the
kernel agree with.

---

## (b) Old code and new code, literal

### 1. `ttnn/cpp/ttnn/operations/eltwise/unary_backward/unary_backward.cpp`

**OLD — lines 951-990 exactly as they are on main today:**

```cpp
std::vector<Tensor> atanh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = std::nanf("");
    float t_inf = std::numeric_limits<float>::infinity();
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::NEG},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};

    Tensor grad_a =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)),
        0.f,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::logical_or(
                ttnn::eq(input, 1, std::nullopt, output_mem_config),
                ttnn::eq(input, -1, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::nez(grad, output_mem_config)),
        t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eq(grad_a, t_inf, std::nullopt, output_mem_config), ttnn::ltz(grad, output_mem_config)),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
```

**NEW — replace the whole function with this:**

```cpp
std::vector<Tensor> atanh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_nan = tt::tt_metal::hal::get_nan();
    float t_inf = tt::tt_metal::hal::get_inf();
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::SUB_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::NEG},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};

    // |input| == 1 is the only singular point of d/dx atanh(x) = 1 / (1 - x^2).
    Tensor at_singularity = ttnn::logical_or(
        ttnn::eq(input, 1, std::nullopt, output_mem_config),
        ttnn::eq(input, -1, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);

    Tensor grad_a =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    // A zero incoming gradient is 0 * finite = 0 everywhere the local derivative is finite;
    // only at the singularity is it 0 * inf, which is NaN.
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(grad, output_mem_config), at_singularity, std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(at_singularity, ttnn::nez(grad, output_mem_config)), t_inf, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eq(grad_a, t_inf, std::nullopt, output_mem_config), ttnn::ltz(grad, output_mem_config)),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}
```

What changed, in words:

1. The unconditional `where(eqz(grad), t_nan, ...)` on line 966 now also requires `|input| == 1`.
2. The rescue block on lines 967-971 (`eqz(grad) && eqz(input) -> 0.f`) is **deleted**: it becomes
   dead code, because with the corrected condition `input == 0, grad == 0` already falls through to
   `grad_a = 0 * 1 = 0`. Its behaviour is preserved exactly (see the case table in (c)).
3. The `logical_or(eq(input, 1), eq(input, -1))` predicate that the function already built inline is
   hoisted into `at_singularity` and reused, so it is computed once instead of being recomputed.
4. `std::nanf("")` / `std::numeric_limits<float>::infinity()` become `hal::get_nan()` / `hal::get_inf()`,
   which is what `acosh_bw` in the same file already uses (lines 580-581).

Net device-op count goes **down**: two `eqz` and one `where` removed, one `logical_and` added.

If a reviewer wants the diff even smaller, changes 3 and 4 can be dropped independently and the fix
still stands on change 1 + 2 alone.

### 2. `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_atanh.py`

**OLD — line 8:**

```python
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import data_gen_with_range, compare_pcc
```

**NEW — line 8 (same shape as `test_backward_acosh.py:8-12`):**

```python
from tests.ttnn.nightly.unit_tests.operations.eltwise.backward.utility_funcs import (
    data_gen_with_range,
    compare_pcc,
    compare_results,
)
```

**NEW — insert this function after the import block, before `test_bw_atanh`:**

```python
# A zero incoming gradient must produce a zero outgoing gradient wherever the local
# derivative d/dx atanh(x) = 1 / (1 - x^2) is finite, i.e. everywhere except |x| == 1.
@pytest.mark.parametrize(
    "in_val, grad_val",
    [
        (0.5, 0.0),
        (-0.5, 0.0),
        (0.99, 0.0),
        (2.0, 0.0),
        (0.0, 0.0),
        (0.5, 1.0),
    ],
)
def test_bw_atanh_zero_grad(in_val, grad_val, device):
    in_data = (torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=True) * in_val).bfloat16()
    input_tensor = ttnn.Tensor(in_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)
    grad_data = (torch.ones(torch.Size([1, 1, 32, 32]), requires_grad=False) * grad_val).bfloat16()
    grad_tensor = ttnn.Tensor(grad_data, ttnn.bfloat16).to(ttnn.TILE_LAYOUT).to(device)

    tt_output_tensor_on_device = ttnn.atanh_bw(grad_tensor, input_tensor)

    golden_function = ttnn.get_golden_function(ttnn.atanh_bw)
    golden_tensor = golden_function(grad_data, in_data)

    comp_pass = compare_results(tt_output_tensor_on_device, golden_tensor)
    assert comp_pass
```

The existing `test_bw_atanh` is left untouched.

---

## (c) The test: why the existing ones cannot see this, and what the new one does

### Three tests exercise `atanh_bw` today. None of them can reach the bug.

**1. `tests/ttnn/nightly/unit_tests/operations/eltwise/backward/test_backward_atanh.py:19-29`**

```python
in_data, input_tensor = data_gen_with_range(input_shapes, -100, 100, device, required_grad=True)
grad_data, grad_tensor = data_gen_with_range(input_shapes, -100, 100, device)
```

`data_gen_with_range` calls `torch.manual_seed(seed)` on **every** invocation with
`seed=DEFAULT_SEED=213919` (`utility_funcs.py:53-55, 13`), so the two calls return the **same
tensor**: `torch.equal(in_data, grad_data)` is `True`. Every element whose gradient is exactly zero
therefore also has input exactly zero — the one case the old rescue on lines 967-971 covers. Of the
1,081 exact zeros the test generates, 0 have a non-zero input. The test cannot trip the bug.

Even with the seeds decoupled it would still pass: `get_pcc`
(`ttnn/tt_lib/_internal/comparison_funcs.py:41-55`) zeroes NaN and Inf in **both** tensors before
correlating, so 1,098 NaN-vs-finite mismatches still score PCC = 0.9999975 against a 0.99 threshold.
That is why this PR does **not** try to fix the test by changing the seed — it would not turn red,
and it would newly exercise `|x|` up to 100 for reasons unrelated to this change.

**2. `tests/ttnn/nightly/unit_tests/operations/eltwise/test_backward.py:47-52` (`test_atanh`)**
parametrizes `in_val` over `[-1, 0, 1]` with `grad_val=1`. The gradient is never zero.

**3. `tests/ttnn/nightly/unit_tests/operations/eltwise/test_backward.py:55-61` (`test_atanh_nan`)**
does parametrize `grad_val=0`, but only over `in_val` in `[-1, 0, 1]` — the three inputs that happen
to be handled correctly. It is also `@pytest.mark.skipif(is_wormhole_b0() or is_blackhole())`, so it
does not run on current hardware at all.

**4.** There is no `atanh_bw` entry in `tests/sweep_framework/` (`atan_bw`, `acosh_bw`, `acos_bw`,
`i0_bw` and ~60 others are there). Confirmed against the full repository tree at `587a4f30`.

### What the new test covers

| in_val | grad_val | golden | before the fix | after the fix | role |
|---|---|---|---|---|---|
| 0.5  | 0.0 | 0.0  | **NaN** | 0.0  | red -> green |
| -0.5 | 0.0 | 0.0  | **NaN** | 0.0  | red -> green |
| 0.99 | 0.0 | 0.0  | **NaN** | 0.0  | red -> green, near the singularity |
| 2.0  | 0.0 | -0.0 | **NaN** | -0.0 | red -> green, outside the atanh domain |
| 0.0  | 0.0 | 0.0  | 0.0     | 0.0  | guards the rescue block being deleted |
| 0.5  | 1.0 | 1.336| 1.336   | 1.336| guards the ordinary path |

`compare_results` is the right comparator here and it is the one `test_bw_acosh_edge_cases`
(`test_backward_acosh.py:16-37`) already uses for exactly this kind of constant-tensor edge case.
For a constant tensor `get_pcc` short-circuits: golden all-finite vs calculated all-NaN hits
*"One tensor is all nan, the other is not"* and returns 0.0, and `comp_allclose` also returns False,
so `comp_pass | comp_all` is False and the assert fires. `compare_pcc` on non-constant tensors would
not fire, which is the whole reason this bug survived.

### The fix does not change any case the existing tests do cover

All nine `(in_val, grad_val)` combinations of `test_atanh_nan`, emulated line by line:

```
in=-1.0 grad=-1.0  before=-inf  after=-inf   identical
in=-1.0 grad= 0.0  before= nan  after= nan   identical
in=-1.0 grad= 1.0  before= inf  after= inf   identical
in= 0.0 grad=-1.0  before=-1.0  after=-1.0   identical
in= 0.0 grad= 0.0  before= 0.0  after= 0.0   identical
in= 0.0 grad= 1.0  before= 1.0  after= 1.0   identical
in= 1.0 grad=-1.0  before=-inf  after=-inf   identical
in= 1.0 grad= 0.0  before= nan  after= nan   identical
in= 1.0 grad= 1.0  before= inf  after= inf   identical
```

---

## (d) PR body, ready to paste

### What is wrong

`atanh_bw` writes NaN into every element whose incoming gradient is exactly `0`, regardless of the
input. `unary_backward.cpp:966`:

```cpp
Tensor grad_a =
    ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
grad_a = where(ttnn::eqz(grad, output_mem_config), t_nan, grad_a, output_mem_config);   // <-- unconditional
grad_a = where(
    ttnn::logical_and(ttnn::eqz(grad, output_mem_config), ttnn::eqz(input, output_mem_config)),
    0.f,
    grad_a,
    output_mem_config);
```

The rescue on the next lines only takes back the case `input == 0`. Every other input keeps the NaN:

```
ttnn.atanh_bw(grad=0.0, input=0.5)   -> nan      golden  0.0
ttnn.atanh_bw(grad=0.0, input=-0.5)  -> nan      golden  0.0
ttnn.atanh_bw(grad=0.0, input=0.99)  -> nan      golden  0.0
ttnn.atanh_bw(grad=0.0, input=2.0)   -> nan      golden -0.0
ttnn.atanh_bw(grad=0.0, input=0.0)   -> 0.0      golden  0.0     (rescued)
ttnn.atanh_bw(grad=0.0, input=1.0)   -> nan      golden  nan     (correct: 0/0)
```

The invariant broken is `0 * finite = 0`. The local derivative `d/dx atanh(x) = 1 / (1 - x^2)` is
finite everywhere except `|x| = 1`, so a zero incoming gradient must come out zero everywhere else.
The registered golden (`ttnn/ttnn/operations/unary_backward.py:85-90`, `torch.atanh` through
autograd) agrees: it returns `0.0` for `|x| < 1`, `-0.0` for `|x| > 1`, and NaN only at `|x| = 1`.

### How wide it is

Sweeping the input over **every finite bfloat16 pattern** with `grad = 0.0`:

```
finite bf16 input patterns          65,280
NaN returned where golden is finite 65,276   (99.99%)
inputs handled correctly            -1.0, 0.0, 1.0     (three of them)
restricted to the atanh domain |x|<1   32,512 patterns, 32,510 wrong
```

Controls, so the measurement is not one-sided: the identical sweep with `grad = 1.0` gives **0**
mismatches out of 65,280 — the comparator is not simply flagging everything. Sweeping the *gradient*
at a fixed `x = 0.5` over all 65,280 finite bf16 patterns gives exactly **2** NaN, and they are
`+0.0` and `-0.0`; the trigger is precisely `grad == 0`, not a precision problem.

On a realistic tensor — shape `(1, 3, 320, 384)`, inputs uniform in `(-0.8, 0.8)`, gradient with 10%
exact zeros — 36,883 of 368,640 elements (10.01%) come back NaN, and the sum of the returned
gradient is `nan` where the golden sum is `-110.79`. Second control: replacing those zeros with
`1e-4` gives 0 NaN out of 368,640.

Zero gradients are not exotic. They are what a padding mask, a dropout mask, a dead ReLU branch or a
`.grad` accumulated over an unused slice produce. A single NaN propagates through the optimizer into
every parameter it touches, so the failure is one wrecked training step, not a rounding difference.

### The correct form is already in this file, twice

`acosh_bw` (line 573) has the same singularity and conditions the NaN on it
(`unary_backward.cpp:586-596`):

```cpp
grad_a = ttnn::where(
    ttnn::logical_or(
        ttnn::lt(in_sq, 1.0f, std::nullopt, output_mem_config),
        ttnn::logical_and(
            ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
            ttnn::eqz(grad, output_mem_config),
            ...
```

`log_bw` (line 829) does the same, nesting the zero-gradient test *inside* the singular case
(`unary_backward.cpp:833-843`):

```cpp
where(ttnn::eqz(input, output_mem_config),
      where(ttnn::eqz(grad, output_mem_config), std::nanf(""), ...),
      grad_a, ...)
```

And `atanh_bw` itself already knows where its singularity is — it builds
`logical_or(eq(input, 1), eq(input, -1))` twenty lines further down, for the `±inf` case. It simply
was not used for the zero-gradient branch.

### What this changes

`where(eqz(grad), nan, ...)` becomes `where(eqz(grad) && |input| == 1, nan, ...)`, reusing the
singularity predicate the function already builds; the `eqz(grad) && eqz(input) -> 0.f` rescue is
then dead and is removed. `std::nanf("")` and `std::numeric_limits<float>::infinity()` become
`hal::get_nan()` / `hal::get_inf()`, matching `acosh_bw` at lines 580-581 in this same file
(`<tt-metalium/hal.hpp>` is already included at line 32).

Net device-op count goes down by two `eqz` and one `where`.

After the change, the full 65,280-pattern sweep at `grad = 0.0` gives **0** mismatches against the
golden, and all nine `(in_val, grad_val)` combinations of the existing `test_atanh_nan` are
bit-identical to before.

### Why no test caught it

- `test_backward_atanh.py:20-21` calls `data_gen_with_range` twice, and that helper does
  `torch.manual_seed(DEFAULT_SEED)` on every call (`utility_funcs.py:53-55`), so `in_data` and
  `grad_data` come out **bit-identical** — `torch.equal` is `True`. Every one of the 1,081 exact
  zeros in the gradient therefore has input `0` too, i.e. every one lands in the single case the old
  rescue covers.
- Decoupling the seeds would not turn it red either: with an independent gradient there are 1,098
  NaN-vs-finite mismatches, but `get_pcc` (`ttnn/tt_lib/_internal/comparison_funcs.py:41-55`) sets
  NaN and Inf to zero in **both** tensors before correlating, giving PCC = 0.9999975 against the
  0.99 threshold. This blind spot applies to every backward op compared with `compare_pcc`, not just
  this one.
- `test_backward.py:51` (`test_atanh`) uses `grad_val=1` only. `test_backward.py:60`
  (`test_atanh_nan`) does use `grad_val=0`, but only with `in_val` in `[-1, 0, 1]` — exactly the
  three inputs that are already correct — and it is skipped on Wormhole and Blackhole.
- There is no `atanh_bw` sweep in `tests/sweep_framework/`, although `atan_bw` and `acosh_bw` have one.

The new test is modelled on `test_bw_acosh_edge_cases` (`test_backward_acosh.py:16-37`), which was
added for the same class of problem in #6583: constant tensors, explicit `(in_val, grad_val)` pairs,
and `compare_results` instead of `compare_pcc`, because on a constant tensor `get_pcc` short-circuits
on all-NaN and actually reports the failure.

### How this was measured

No Tenstorrent hardware was used. The analysis is from the source, and the numbers come from
reimplementing `unary_backward.cpp:951-990` line by line in float32 with bfloat16 inputs and outputs
and comparing against the registered golden (`torch.atanh` through autograd, torch 2.13.0+cpu).
Both the current code and the proposed code were emulated, which is where the "0 mismatches after"
figure comes from. The `where`/`eqz`/`logical_or` semantics were read from
`ttnn/cpp/ttnn/operations/eltwise/ternary/ternary.hpp` — `ttnn::where(predicate, scalar, tensor)` is
a real select, so the NaN is written, not folded away. The on-device numbers should match, since the
op is a host-side composite with no dtype or architecture branch, but CI is the first real
compilation and the first real run.

### Notes for reviewers

- The behaviour at `|input| = 1` is deliberately unchanged: `grad == 0` there is `0/0` and torch
  returns NaN too, so the NaN is kept.
- The removal of the `eqz(grad) && eqz(input) -> 0.f` block is not a behaviour change; `input == 0`
  now reaches `0 * 1 = 0` through the normal path. The new test carries `(0.0, 0.0)` as the guard.
- The `hal::get_nan()` / `hal::get_inf()` swap is a separate one-line cleanup that matches
  `acosh_bw` above; happy to drop it if you would rather keep the diff to the condition alone.
- `test_backward_atanh.py`'s seed collision is described above but **not** changed here, because
  fixing it does not make the test fail and it would newly exercise `|x|` up to 100 for reasons
  unrelated to this PR. Worth its own issue if you want it.


// Asin
// result: grad * (-self * self + 1).rsqrt()
std::vector<Tensor> asin_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::NEG},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RSQRT}};

    Tensor grad_result =
        ttnn::multiply(grad, unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor sub_one = ttnn::add(input, -1, std::nullopt, output_mem_config);
    Tensor sub_minus_one = ttnn::add(input, 1, std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::ltz(sub_minus_one, output_mem_config),
        t_nan,
        where(
            ttnn::gtz(sub_one, output_mem_config),
            t_nan,
            where(
                ttnn::eqz(sub_minus_one, output_mem_config),
                ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                where(
                    ttnn::eqz(sub_one, output_mem_config),
                    ttnn::multiply(ttnn::sign(grad, output_mem_config), t_inf, std::nullopt, output_mem_config),
                    grad_result,
                    output_mem_config),
                output_mem_config),
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// Asinh
// result: grad * (self * self + 1).rsqrt()
std::vector<Tensor> asinh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::RSQRT}};
    Tensor grad_result =
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// name: sin(Tensor self) -> Tensor
// self: grad * self.cos()
std::vector<Tensor> sin_bw(
    const Tensor& grad, const Tensor& input_tensor, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor grad_input =
        ttnn::multiply(grad, ttnn::cos(input_tensor, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_input);
    return grad_tensor;
}

// name: sinh(Tensor self) -> Tensor
// self: grad * self.cosh()
std::vector<Tensor> sinh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, 88.5f, std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, -88.5f, std::nullopt, output_mem_config),
            t_inf,
            ttnn::multiply(grad, ttnn::cosh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38f, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38f, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log10(in)) = grad/(in * 2.30258509299404568402)
std::vector<Tensor> log10_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad,
        ttnn::reciprocal(
            ttnn::multiply(input, std::numbers::ln10_v<float>, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// bw(log1p(in)) = grad/(in + 1)
// for -1 = inf
std::vector<Tensor> log1p_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor t_inp1 = ttnn::add(input, 1.0f, std::nullopt, output_mem_config);
    Tensor grad_a = ttnn::multiply(grad, ttnn::reciprocal(t_inp1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(ttnn::eq(input, -1.0f, std::nullopt, output_mem_config), t_inf, grad_a, output_mem_config);
    grad_a = where(
        ttnn::logical_and(ttnn::eqz(t_inp1, output_mem_config), eqz(grad, output_mem_config)),
        std::nanf(" "),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> erfc_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        static_cast<float>(-M_2_SQRTPI),
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> ceil_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& /*output_mem_config*/) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

// softsign
// result = grad_data / torch.square(1 + torch.abs(input))
std::vector<Tensor> softsign_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    using ttnn::operations::unary::EltwiseUnaryWithParam;
    using ttnn::operations::unary::UnaryOpType;
    std::vector<EltwiseUnaryWithParam> ops_chain = {
        EltwiseUnaryWithParam{UnaryOpType::ABS},
        EltwiseUnaryWithParam{UnaryOpType::ADD_UNARY_SFPU, 1.0f},
        EltwiseUnaryWithParam{UnaryOpType::SQUARE},
        EltwiseUnaryWithParam{UnaryOpType::RECIP}};
    grad_tensor.emplace_back(
        ttnn::multiply(grad, ttnn::unary_chain(input, ops_chain, output_mem_config), std::nullopt, output_mem_config));
    return grad_tensor;
}

// name: cosh(Tensor self) -> Tensor
// self: grad * self.sinh()
std::vector<Tensor> cosh_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor t_neg_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), -std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    Tensor grad_a = where(
        ttnn::gt(input, 88.50f, std::nullopt, output_mem_config),
        t_inf,
        where(
            ttnn::lt(input, -88.50f, std::nullopt, output_mem_config),
            t_neg_inf,
            ttnn::multiply(grad, ttnn::sinh(input, output_mem_config), std::nullopt, output_mem_config),
            output_mem_config),
        output_mem_config);
    t_neg_inf.deallocate();
    t_inf.deallocate();
    grad_a = where(
        ttnn::ge(grad_a, 3.4e+38f, std::nullopt, output_mem_config),
        std::numeric_limits<float>::infinity(),
        where(
            ttnn::le(grad_a, -3.4e+38f, std::nullopt, output_mem_config),
            -std::numeric_limits<float>::infinity(),
            grad_a,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// Torch reference
// # if eps is not None:
// #         lo = eps
// #         hi = 1.0 - lo
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= lo, self <= hi),
// #             grad_output / (self * (1.0 - self)),
// #             0.0,
// #         )
// #     else:
// #         return torch.where(
// #             torch.ttnn::logical_and(self >= 0.0, self <= 1.0),
// #             grad_output / (self * (1.0 - self)),
// #             self.new_full((), float("nan")),
// #         )
std::vector<Tensor> logiteps_bw(
    const Tensor& grad, const Tensor& input, float eps, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float low, high;
    low = eps;
    high = 1.0 - low;
    Tensor grad_result = ttnn::multiply(
        grad,
        ttnn::reciprocal(ttnn::multiply(
            input, ttnn::rsub(input, 1.0f, std::nullopt, output_mem_config), std::nullopt, output_mem_config)),
        std::nullopt,
        output_mem_config);
    Tensor t_eps = ttnn::full_like(input, eps, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    Tensor ltl_gth = ttnn::logical_or(
        ttnn::lt(input, low, std::nullopt, output_mem_config),
        ttnn::gt(input, high, std::nullopt, output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_result = where(
        ttnn::eq(ltl_gth, 1.0f, std::nullopt, output_mem_config),
        where(ttnn::ltz(t_eps, output_mem_config), std::nanf(" "), 0.f, output_mem_config),
        where(
            ttnn::logical_or(
                ttnn::eq(input, 0.0f, std::nullopt, output_mem_config),
                ttnn::eq(input, 1.0f, std::nullopt, output_mem_config),
                std::nullopt,
                output_mem_config),
            ttnn::multiply(
                ttnn::sign(grad, output_mem_config),
                std::numeric_limits<float>::infinity(),
                std::nullopt,
                output_mem_config),
            grad_result,
            output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

// bw(log2(in)) = grad/(in * 0.69314718055994530942)
std::vector<Tensor> log2_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor t_inf = where(
        ttnn::ltz(grad, output_mem_config),
        -std::numeric_limits<float>::infinity(),
        std::numeric_limits<float>::infinity(),
        output_mem_config);
    Tensor grad_a = ttnn::multiply(
        grad,
        ttnn::reciprocal(
            ttnn::multiply(input, std::numbers::ln2_v<float>, std::nullopt, output_mem_config), output_mem_config),
        std::nullopt,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        std::nanf(" "),
        where(ttnn::eqz(input, output_mem_config), t_inf, grad_a, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> sign_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor zero_grad = ttnn::zeros_like(grad, grad.dtype(), grad.layout(), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(zero_grad);
    return grad_tensor;
}

std::vector<Tensor> div_no_nan_bw(
    const Tensor& grad, const Tensor& input, float scalar, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor val = ttnn::full_like(input, scalar, input.dtype(), input.layout(), std::nullopt, output_mem_config);
    Tensor result = where(
        ttnn::eq(val, 0, std::nullopt, output_mem_config),
        0.f,
        ttnn::multiply(grad, 1 / scalar, std::nullopt, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// #  bw (exp2) = grad * exp2(input) * M_LN2
// # M_LN2 = 0.693147180559945309417
std::vector<Tensor> exp2_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor exp_result = ttnn::exp2(input, output_mem_config);
    exp_result = ttnn::multiply(exp_result, std::numbers::ln2_v<float>, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, exp_result, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

// bw(expm1) = grad * expm1(input) + 1
std::vector<Tensor> expm1_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor eresult = ttnn::expm1(input, output_mem_config);
    Tensor rp1 = ttnn::add(eresult, 1.0f, std::nullopt, output_mem_config);
    Tensor result = ttnn::multiply(grad, rp1, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> reciprocal_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    grad_tensor.emplace_back(where(
        ttnn::eqz(input, output_mem_config),
        where(
            ttnn::eqz(grad, output_mem_config),
            t_nan,
            ttnn::multiply(
                ttnn::neg(ttnn::sign(grad, output_mem_config), output_mem_config),
                t_inf,
                std::nullopt,
                output_mem_config),
            output_mem_config),
        ttnn::multiply(
            ttnn::neg(grad, output_mem_config),
            ttnn::reciprocal(ttnn::square(input, output_mem_config), output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config));
    return grad_tensor;
}

std::vector<ComplexTensor> reciprocal_bw(
    const ComplexTensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor condition_nan = ttnn::logical_and(
        ttnn::eqz(input.real(), output_mem_config),
        ttnn::eqz(input.imag(), output_mem_config),
        std::nullopt,
        output_mem_config);
    ComplexTensor neg_grad =
        ComplexTensor({ttnn::neg(grad.real(), output_mem_config), ttnn::neg(grad.imag(), output_mem_config)});
    ComplexTensor inp_recip = ttnn::reciprocal(input, output_mem_config);
    ComplexTensor grad_inp = ttnn::operations::complex_binary::multiply(
        neg_grad,
        ttnn::conj(
            ttnn::operations::complex_binary::multiply(inp_recip, inp_recip, output_mem_config), output_mem_config),
        output_mem_config);
    neg_grad.deallocate();
    inp_recip.deallocate();
    Tensor grad_inp_r = where(condition_nan, std::nanf(""), grad_inp.real(), output_mem_config);
    Tensor grad_inp_i = where(condition_nan, std::nanf(""), grad_inp.imag(), output_mem_config);
    condition_nan.deallocate();
    grad_inp = ComplexTensor({grad_inp_r, grad_inp_i});
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

std::vector<Tensor> abs_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(grad, ttnn::sign(input, output_mem_config), std::nullopt, output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<ComplexTensor> abs_bw(
    const Tensor& grad, const ComplexTensor& input, const MemoryConfig& output_mem_config) {
    std::vector<ComplexTensor> grad_tensor;
    Tensor result = ttnn::abs(input, output_mem_config);
    Tensor grad_inp_r = where(
        ttnn::eqz(result, output_mem_config),
        0.f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.real(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    Tensor grad_inp_i = where(
        ttnn::eqz(result, output_mem_config),
        0.f,
        ttnn::multiply(
            grad,
            ttnn::multiply(input.imag(), ttnn::reciprocal(result, output_mem_config), std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        output_mem_config);
    ComplexTensor grad_inp = ComplexTensor({grad_inp_r, grad_inp_i});
    result.deallocate();
    grad_inp_r.deallocate();
    grad_inp_i.deallocate();
    grad_tensor.emplace_back(grad_inp);
    return grad_tensor;
}

std::vector<Tensor> digamma_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_inf = std::numeric_limits<float>::infinity();
    float t_nan = std::nanf("");
    Tensor grad_a = ttnn::multiply(grad, ttnn::polygamma(input, 1, output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::eqz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        -t_inf,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        t_inf,
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

std::vector<Tensor> polygamma_bw(
    const Tensor& grad, const Tensor& input, int n, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(input.memory_config());
    float t_nan = std::nanf("");
    float pos_neg = 1.0f;
    if (n == 2 || n == 4 || n == 6 || n == 8 || n == 10) {
        pos_neg = -1.0f;
    }
    Tensor grad_a =
        ttnn::multiply(grad, ttnn::polygamma(input, (n + 1), output_mem_config), std::nullopt, output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::le(input, 0.0f, std::nullopt, output_mem_config),
            ttnn::eqz(grad, output_mem_config),
            std::nullopt,
            output_mem_config),
        t_nan,
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::gtz(grad, output_mem_config), std::nullopt, output_mem_config),
        (-std::numeric_limits<float>::infinity() * pos_neg),
        grad_a,
        output_mem_config);
    grad_a = where(
        ttnn::logical_and(
            ttnn::eqz(input, output_mem_config), ttnn::ltz(grad, output_mem_config), std::nullopt, output_mem_config),
        (std::numeric_limits<float>::infinity() * pos_neg),
        grad_a,
        output_mem_config);
    grad_tensor.emplace_back(grad_a);
    return grad_tensor;
}

// erfinv
// self: 0.5 * sqrt(M_PI) * exp(self.erfinv().pow(2)) * grad
// for input -1 and 1: grad.sign() * inf, for input > 1 or < -1 : nan
std::vector<Tensor> erfinv_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float m_sqrtpi = 1.77245385090551602792981f;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::multiply(
                ttnn::exp(
                    ttnn::square(ttnn::erfinv(input, output_mem_config), output_mem_config), false, output_mem_config),
                grad,
                std::nullopt,
                output_mem_config),
            m_sqrtpi,
            std::nullopt,
            output_mem_config),
        0.5f,
        std::nullopt,
        output_mem_config);
    Tensor t_inf = ttnn::multiply(
        ttnn::sign(grad, output_mem_config), std::numeric_limits<float>::infinity(), std::nullopt, output_mem_config);
    result = ttnn::where(
        ttnn::logical_or(
            ttnn::lt(input, -1.0f, std::nullopt, output_mem_config),
            ttnn::gt(input, 1.0f, std::nullopt, output_mem_config),
            std::nullopt,
            output_mem_config),
        std::nanf(" "),
        result,
        output_mem_config);
    result = ttnn::where(
        ttnn::eq(input, -1.0f, std::nullopt, output_mem_config),
        t_inf,
        ttnn::where(ttnn::eq(input, 1.0f, std::nullopt, output_mem_config), t_inf, result, output_mem_config),
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> erf_bw(
    const Tensor& grad, const Tensor& input, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    Tensor result = ttnn::multiply(
        ttnn::multiply(
            ttnn::exp(ttnn::neg(ttnn::square(input, output_mem_config), output_mem_config), false, output_mem_config),
            grad,
            std::nullopt,
            output_mem_config),
        2.0f * std::numbers::inv_sqrtpi_v<float>,
        std::nullopt,
        output_mem_config);
    grad_tensor.emplace_back(result);
    return grad_tensor;
}

std::vector<Tensor> deg2rad_bw(
    const Tensor& grad, const Tensor& /*input*/, const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    float M_PI_180 = M_PI / 180;
    Tensor grad_result = ttnn::multiply(grad, M_PI_180, std::nullopt, output_mem_config);
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

std::vector<std::optional<ttnn::Tensor>> gelu_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::string& approximate,
    const std::optional<MemoryConfig>& output_mem_config,
    std::optional<Tensor> input_grad) {
    std::vector<std::optional<Tensor>> result;
    if (!input_grad.has_value()) {
        input_grad = ttnn::empty_like(grad);
    }

    auto output_memory_config =
        input_grad.has_value() ? input_grad->memory_config() : output_mem_config.value_or(input.memory_config());
    TT_FATAL((approximate == "none" || approximate == "tanh"), "Incorrect approximate mode (expected 'none', 'tanh')");

    DataType output_dtype = input.dtype();
    auto result_tensor = ttnn::operations::unary_backward::gelu_bw::launch_gelu_bw(
        grad, input, approximate == "tanh", output_dtype, output_memory_config, input_grad);
    result.push_back(result_tensor);

    return result;
}

std::vector<Tensor> repeat_bw(
    const Tensor& grad,
    const Tensor& input,
    const ttnn::Shape& shape,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed

    auto shape_wh = input.padded_shape();
    TT_FATAL(shape_wh[0] == 1, "Input shape[0] must be 1 but got {}", shape_wh[0]);
    auto* ttnn_device = input.device();
    // input.padded_shape()[0]
    // If repeat shape has 0's, it returns zeros of given input
    if (shape[0] == 0 || shape[1] == 0 || shape[2] == 0 || shape[3] == 0) {
        Tensor zero_tensor = ttnn::zeros_like(input, input.dtype(), input.layout(), std::nullopt, output_memory_config);
        grad_tensor.emplace_back(zero_tensor);
        return grad_tensor;
    }
    if (shape[0] > 1) {
        ttsl::SmallVector<int64_t> dim = {0};
        TT_FATAL(shape[1] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[1], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {1, shape_wh[1], shape_wh[2], shape_wh[3]};
        const auto required = ttnn::Shape(intended_shape_array);
        Tensor result = ttnn::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.dtype(), input.layout(), *ttnn_device, output_memory_config),
            output_memory_config,
            std::nullopt);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    if (shape[1] > 1) {
        ttsl::SmallVector<int64_t> dim = {1};
        TT_FATAL(shape[0] == 1 && shape[2] == 1 && shape[3] == 1, "repeat[0], [2], [3] should be 1");
        std::array<std::uint32_t, 4> intended_shape_array = {shape_wh[0], 1, shape_wh[2], shape_wh[3]};
        const auto required = ttnn::Shape(intended_shape_array);
        Tensor result = ttnn::moreh_sum(
            grad,
            dim,
            true,
            ttnn::zeros(required, input.dtype(), input.layout(), *ttnn_device, output_memory_config),
            output_memory_config,
            std::nullopt);
        grad_tensor.emplace_back(result);
        return grad_tensor;
    }
    return grad_tensor;
}

}  // namespace ttnn

namespace ttnn::operations::unary_backward {
// Autoformat support
Tensor change_layout_to_tile(const Tensor& input_tensor, const MemoryConfig& /*output_mem_config*/) {
    auto formatted_input_tensor = input_tensor;
    if (input_tensor.layout() == Layout::ROW_MAJOR) {
        auto a_pad_shape = ttnn::operations::data_movement::pad_to_tile_shape(input_tensor.padded_shape());
        formatted_input_tensor =
            ttnn::tilize_with_val_padding(input_tensor, a_pad_shape, PadValue(1.0f), input_tensor.memory_config());
    }
    return formatted_input_tensor;
}

}  // namespace ttnn::operations::unary_backward

namespace ttnn {

// Prod
// along a single dimension --> result: grad_data * (y / input )
std::vector<Tensor> prod_bw(
    const Tensor& grad,
    const Tensor& input,
    const std::optional<int64_t> dim,
    const std::optional<MemoryConfig>& output_mem_config) {
    std::vector<Tensor> grad_tensor;
    auto output_memory_config = output_mem_config.value_or(
        input.memory_config());  // TODO: Remove after ternary forward ops migration is completed

    const bool all_dimensions = !dim.has_value();
    const bool keepdim = !all_dimensions;
    Tensor prod_result = ttnn::prod(input, dim, keepdim, output_memory_config);

    if (prod_result.layout() == Layout::ROW_MAJOR && prod_result.storage_type() == StorageType::DEVICE) {
        prod_result = ttnn::operations::unary_backward::change_layout_to_tile(prod_result, output_memory_config);
    }

    if (all_dimensions) {
        Tensor temp = ttnn::multiply(
            prod_result, grad, std::nullopt, output_memory_config);  // result is stored in the first position
        Tensor fill_tensor = ttnn::fill_first_val_into_tensor<::bfloat16>(
            temp, temp.dtype(), temp.layout(), temp.device(), output_memory_config);
        Tensor all_dimension_result = ttnn::multiply(
            ttnn::reciprocal(input, output_memory_config), fill_tensor, std::nullopt, output_memory_config);
        grad_tensor.emplace_back(all_dimension_result);
        return grad_tensor;
    }

    // all_dimensions = False
    Tensor updated_grad = prod_result;
    auto step = ttsl::SmallVector<uint32_t>({1, 1, 1, 1});
    if (prod_result.logical_shape() != grad.padded_shape()) {
        if (*dim == 3 || *dim == -1) {
            ttsl::SmallVector<int64_t> after_permute_dims = {0, 3, 1, 2};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttsl::SmallVector<uint32_t> end_index = {
                grad.padded_shape()[0], 1, grad.padded_shape()[1], grad.padded_shape()[2]};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
            after_permute_dims = {0, 2, 3, 1};
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if (updated_grad.storage_type() != StorageType::DEVICE) {
                Tensor pad_updated_grad = updated_grad.pad_to_tile(1.0f);
                pad_updated_grad = pad_updated_grad.to_layout(Layout::TILE);
                updated_grad = pad_updated_grad.to_device(input.device());
            }
        } else if (*dim == 2 || *dim == -2) {
            ttsl::SmallVector<int64_t> after_permute_dims = {0, 2, 1, 3};
            Tensor required = ttnn::permute(grad, after_permute_dims, output_memory_config);
            ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttsl::SmallVector<uint32_t> end_index = {
                grad.padded_shape()[0], 1, grad.padded_shape()[1], grad.padded_shape()[3]};
            Tensor new_slice_tensor = ttnn::slice(required, start_index, end_index, step, std::nullopt);
            updated_grad = ttnn::permute(new_slice_tensor, after_permute_dims, output_memory_config);
            if (updated_grad.layout() == Layout::ROW_MAJOR) {
                updated_grad =
                    ttnn::operations::unary_backward::change_layout_to_tile(updated_grad, output_memory_config);
            }
        }
    }
    Tensor reciprocal_input = ttnn::reciprocal(input, output_memory_config);
    Tensor temp = ttnn::multiply(
        prod_result,
        (*dim == 1 || *dim == 0 || *dim == -4 || *dim == -3) ? grad : updated_grad,
        std::nullopt,
        output_memory_config);
    if (temp.layout() == Layout::ROW_MAJOR) {
        temp = ttnn::operations::unary_backward::change_layout_to_tile(temp, output_memory_config);
    }
    if (*dim == 3 || *dim == -1) {
        Tensor grad_result =
            ttnn::bcast(reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    if (*dim == 2 || *dim == -2) {
        Tensor grad_result =
            ttnn::bcast(reciprocal_input, temp, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::H, output_memory_config);
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    if (*dim == 1 || *dim == -3) {
        Tensor tensor_1_temp = reciprocal_input;
        if (reciprocal_input.padded_shape()[1] % 32 != 0) {
            ttsl::SmallVector<std::array<uint32_t, 2>> padding = {
                {0, 0}, {0, 32 - (reciprocal_input.padded_shape()[1] % 32)}, {0, 0}, {0, 0}};
            tensor_1_temp = ttnn::pad(reciprocal_input, padding, 0, true, std::nullopt);
        }
        ttsl::SmallVector<int64_t> after_permute_dims = {0, 2, 3, 1};
        Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
        Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

        // put the tensor back on device because permute throws it off device
        // See: Remove auto format within permute_op.cpp #9404
        auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(tensor_1.padded_shape());
        // tensor_2 is always TILE layout (from permute of TILE temp)
        // Only need to convert if tensor_1 is ROW_MAJOR
        tensor_2 = tensor_2.to_device(tensor_1.device());
        if (tensor_1.layout() == Layout::ROW_MAJOR) {
            // Need to untilize tensor_2 to match tensor_1's ROW_MAJOR layout.
            // untilize may drop tile padding (returning padded_shape == logical_shape), so decide whether
            // padding is required from the tensor *after* untilize rather than before.
            tensor_2 = ttnn::untilize(tensor_2, tensor_1.memory_config());
            if (tensor_2.padded_shape() != padded_shape) {
                tensor_2 = ttnn::pad(
                    tensor_2,
                    padded_shape.to_array_4D(),
                    ttnn::Array4D({0, 0, 0, 0}),
                    0.0f,
                    false,
                    tensor_1.memory_config());
            }
        }
        // If tensor_1 is TILE, tensor_2 is already correct (both TILE, shapes match by assumption)

        after_permute_dims = {0, 3, 1, 2};
        Tensor result = permute(
            ttnn::bcast(tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
            after_permute_dims,
            output_memory_config);
        Tensor grad_result = result;
        if (reciprocal_input.padded_shape()[1] % 32 != 0) {
            ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
            ttsl::SmallVector<uint32_t> end_index = {
                input.padded_shape()[0], input.padded_shape()[1], input.padded_shape()[2], input.padded_shape()[3]};
            auto step = ttsl::SmallVector<uint32_t>({1, 1, 1, 1});
            grad_result = ttnn::slice(result, start_index, end_index, step, std::nullopt);
        }
        grad_tensor.emplace_back(grad_result);
        return grad_tensor;
    }
    // dim 0
    Tensor tensor_1_temp = reciprocal_input;
    if (reciprocal_input.padded_shape()[0] % 32 != 0) {
        ttsl::SmallVector<std::array<uint32_t, 2>> padding = {
            {0, (32 - (reciprocal_input.padded_shape()[0] % 32))}, {0, 0}, {0, 0}, {0, 0}};
        tensor_1_temp = ttnn::pad(reciprocal_input, padding, 0, false, std::nullopt);
    }
    ttsl::SmallVector<int64_t> after_permute_dims = {3, 1, 2, 0};
    Tensor tensor_1 = ttnn::permute(tensor_1_temp, after_permute_dims, output_memory_config);
    Tensor tensor_2 = ttnn::permute(temp, after_permute_dims, output_memory_config);

    // put the tensor back on device because permute throws it off device
    // See: Remove auto format within permute_op.cpp #9404
    auto padded_shape = ttnn::operations::data_movement::pad_to_tile_shape(tensor_2.padded_shape());
    // tensor_2 is always TILE layout (from permute of TILE temp)
    // Only need to convert if tensor_1 is ROW_MAJOR
    tensor_2 = tensor_2.to_device(tensor_1.device());
    if (tensor_1.layout() == Layout::ROW_MAJOR) {
        // Need to untilize tensor_2 to match tensor_1's ROW_MAJOR layout.
        // untilize may drop tile padding (returning padded_shape == logical_shape), so decide whether
        // padding is required from the tensor *after* untilize rather than before.
        tensor_2 = ttnn::untilize(tensor_2, tensor_1.memory_config());
        if (tensor_2.padded_shape() != padded_shape) {
            tensor_2 = ttnn::pad(
                tensor_2,
                padded_shape.to_array_4D(),
                ttnn::Array4D({0, 0, 0, 0}),
                0.0f,
                false,
                tensor_1.memory_config());
        }
    }
    // If tensor_1 is TILE, tensor_2 is already correct (both TILE, shapes match by assumption)

    Tensor result = ttnn::permute(
        ttnn::bcast(tensor_1, tensor_2, ttnn::BcastOpMath::MUL, ttnn::BcastOpDim::W, output_memory_config),
        after_permute_dims,
        output_memory_config);
    Tensor grad_result = result;
    if (reciprocal_input.padded_shape()[0] % 32 != 0) {
        ttsl::SmallVector<uint32_t> start_index = {0, 0, 0, 0};
        ttsl::SmallVector<uint32_t> end_index = {
            input.padded_shape()[0], input.padded_shape()[1], input.padded_shape()[2], input.padded_shape()[3]};
        grad_result = ttnn::slice(result, start_index, end_index, step, std::nullopt);
    }
    grad_tensor.emplace_back(grad_result);
    return grad_tensor;
}

}  // namespace ttnn
