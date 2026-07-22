//! Bounded multistart soft-L1 fitting of the damped-cosine model.

const std = @import("std");
const dynamics_analysis = @import("dynamics_analysis");
const differentiate = @import("backend/differentiate.zig");
const laplace = @import("laplace.zig");
const linalg = dynamics_analysis.backend.linalg;
const options = @import("options.zig");
const result = @import("result.zig");
const schema = @import("schema.zig");

pub const parameter_count = differentiate.parameter_count;
pub const Parameters = differentiate.Parameters;
const phase_margin = 1.0e-6;

const FitState = struct {
    parameters: Parameters,
    objective: f64,
    converged: bool,
};

const StartValues = struct {
    values: [3]f64 = undefined,
    len: usize = 0,
};

pub fn dampedCosine(
    allocator: std.mem.Allocator,
    time: []const f64,
    parameters: Parameters,
) ![]f64 {
    const values = try allocator.alloc(f64, time.len);
    errdefer allocator.free(values);
    try fillDampedCosine(values, time, parameters);
    return values;
}

pub fn dampedCosineJacobian(
    allocator: std.mem.Allocator,
    time: []const f64,
    parameters: Parameters,
) ![]f64 {
    return differentiate.dampedCosineJacobian(allocator, time, parameters);
}

pub fn fitDampedCosine(
    allocator: std.mem.Allocator,
    context: *dynamics_analysis.backend.Context,
    correlation: dynamics_analysis.CorrelationSeries,
    omega_grid: []const f64,
    fit_options: options.FitOptions,
) !result.DampedCosineFit {
    const spacing = try schema.validateFitInput(correlation, omega_grid, fit_options);
    const time = correlation.lag_times;
    const observed = correlation.pearson;
    const duration = time[time.len - 1];
    const nyquist = std.math.pi / spacing;
    const preferred_omega = (try laplace.preferredCoordinate(
        allocator,
        context,
        correlation,
        .omega,
        omega_grid,
    )).coordinate;

    const tail_count = @max(@as(usize, 3), observed.len / 10);
    const tail = try allocator.dupe(f64, observed[observed.len - tail_count ..]);
    defer allocator.free(tail);
    std.mem.sort(f64, tail, {}, comptime std.sort.asc(f64));
    const initial_offset = clamp(median(tail), -0.5, 0.5);
    const initial_amplitude = clamp(1.0 - initial_offset, 0.05, 2.0);
    const lower = Parameters{ 0.0, 0.0, 0.0, -0.5 * std.math.pi + phase_margin };
    const upper = Parameters{ 2.0, 1.0 / spacing, nyquist, 0.5 * std.math.pi - phase_margin };

    const weights = try allocator.alloc(f64, observed.len);
    defer allocator.free(weights);
    const origin_zero = @as(f64, @floatFromInt(correlation.origin_counts[0]));
    for (correlation.origin_counts, weights) |count, *weight| {
        weight.* = @sqrt(@as(f64, @floatFromInt(count)) / origin_zero);
    }

    const rate_starts = uniqueClamped(
        .{ 1.0 / duration, 3.0 / duration, 10.0 / duration },
        std.math.floatEps(f64),
        upper[1] * (1.0 - 1.0e-8),
    );
    const omega_starts = uniqueClamped(
        .{ 0.5 * preferred_omega, preferred_omega, 1.5 * preferred_omega },
        std.math.floatEps(f64),
        upper[2] * (1.0 - 1.0e-8),
    );

    const prediction_scratch = try allocator.alloc(f64, time.len);
    defer allocator.free(prediction_scratch);
    const jacobian_scratch = try allocator.alloc(f64, try std.math.mul(usize, time.len, parameter_count));
    defer allocator.free(jacobian_scratch);

    var evaluations: usize = 0;
    var best: ?FitState = null;
    var best_finite: ?FitState = null;
    for (rate_starts.values[0..rate_starts.len]) |initial_rate| {
        for (omega_starts.values[0..omega_starts.len]) |initial_omega| {
            const state = try optimizeStart(
                context,
                time,
                observed,
                weights,
                .{ initial_amplitude, initial_rate, initial_omega, 0.0 },
                lower,
                upper,
                fit_options,
                prediction_scratch,
                jacobian_scratch,
                &evaluations,
            );
            if (finiteState(state) and
                (best_finite == null or state.objective < best_finite.?.objective))
            {
                best_finite = state;
            }
            if (state.converged and finiteState(state) and
                (best == null or state.objective < best.?.objective))
            {
                best = state;
            }
        }
    }
    const selected = best orelse best_finite orelse return error.NoFiniteFit;
    const prediction = try dampedCosine(allocator, time, selected.parameters);
    errdefer allocator.free(prediction);

    var mean: f64 = 0.0;
    for (observed) |value| mean += value;
    mean /= @as(f64, @floatFromInt(observed.len));
    var residual_sum: f64 = 0.0;
    var total_sum: f64 = 0.0;
    for (observed, prediction) |actual, fitted| {
        const residual = actual - fitted;
        const centered = actual - mean;
        residual_sum += residual * residual;
        total_sum += centered * centered;
    }
    if (!std.math.isFinite(total_sum) or total_sum <= 0.0) {
        return error.ConstantCorrelation;
    }

    const amplitude = selected.parameters[0];
    const rate = selected.parameters[1];
    const omega = selected.parameters[2];
    const phase = selected.parameters[3];
    return .{
        .amplitude = amplitude,
        .rate = rate,
        .omega = omega,
        .phase = phase,
        .offset = 1.0 - amplitude * @cos(phase),
        .r_squared = 1.0 - residual_sum / total_sum,
        .evaluations = evaluations,
        .converged = selected.converged,
        .rate_at_lower_boundary = rate <= (1.0 / duration) * 1.0e-6,
        .rate_at_upper_boundary = rate >= upper[1] * (1.0 - 1.0e-4),
        .amplitude_at_upper_boundary = amplitude >= upper[0] * (1.0 - 1.0e-4),
        .prediction = prediction,
    };
}

fn optimizeStart(
    context: *dynamics_analysis.backend.Context,
    time: []const f64,
    observed: []const f64,
    weights: []const f64,
    initial: Parameters,
    lower: Parameters,
    upper: Parameters,
    fit_options: options.FitOptions,
    prediction: []f64,
    jacobian: []f64,
    evaluations: *usize,
) !FitState {
    var parameters = initial;
    var objective_value = objective(time, observed, weights, parameters, fit_options.soft_l1_scale);
    evaluations.* += 1;
    var start_evaluations: usize = 1;
    var damping: f64 = 1.0e-3;
    var converged = false;

    while (start_evaluations < fit_options.maximum_evaluations) {
        try fillDampedCosine(prediction, time, parameters);
        try differentiate.fillDampedCosineJacobian(jacobian, time, parameters);
        var gradient_system = try differentiate.assembleSoftL1GradientSystem(
            prediction,
            observed,
            weights,
            jacobian,
            fit_options.soft_l1_scale,
        );
        const gradient_maximum = differentiate.projectedScaledGradientMaximum(
            parameters,
            gradient_system.gradient,
            gradient_system.column_norm_squared,
            lower,
            upper,
        );
        if (gradient_maximum <= fit_options.tolerance) {
            converged = true;
            break;
        }
        for (0..parameter_count) |column| {
            gradient_system.normal_matrix[column * parameter_count + column] +=
                damping * @max(gradient_system.column_norm_squared[column], 1.0e-24);
        }
        var rhs: Parameters = undefined;
        for (gradient_system.gradient, &rhs) |value, *target| target.* = -value;
        const step = try solveStep(
            context,
            gradient_system.normal_matrix,
            rhs,
            fit_options.rank_tolerance,
        );

        var candidate = parameters;
        var scaled_step: f64 = 0.0;
        for (0..parameter_count) |index| {
            candidate[index] = clamp(parameters[index] + step[index], lower[index], upper[index]);
            scaled_step = @max(
                scaled_step,
                @abs(candidate[index] - parameters[index]) / (1.0 + @abs(parameters[index])),
            );
        }
        const candidate_objective = objective(
            time,
            observed,
            weights,
            candidate,
            fit_options.soft_l1_scale,
        );
        evaluations.* += 1;
        start_evaluations += 1;
        const secondary_tolerance = @sqrt(fit_options.tolerance);

        if (candidate_objective < objective_value) {
            const relative_change = (objective_value - candidate_objective) /
                (1.0 + @abs(objective_value));
            parameters = candidate;
            objective_value = candidate_objective;
            damping = @max(damping * 0.3, 1.0e-12);
            if ((relative_change <= fit_options.tolerance or scaled_step <= fit_options.tolerance) and
                gradient_maximum <= secondary_tolerance)
            {
                converged = true;
                break;
            }
        } else {
            if (scaled_step <= fit_options.tolerance and gradient_maximum <= secondary_tolerance) {
                converged = true;
                break;
            }
            damping = @min(damping * 10.0, 1.0e12);
        }
    }
    return .{ .parameters = parameters, .objective = objective_value, .converged = converged };
}

fn solveStep(
    context: *dynamics_analysis.backend.Context,
    normal_matrix: [parameter_count * parameter_count]f64,
    rhs: Parameters,
    rank_tolerance: f64,
) !Parameters {
    var coefficients = try linalg.Matrix(f64).fromHost(
        context,
        parameter_count,
        parameter_count,
        &normal_matrix,
    );
    defer coefficients.deinit();
    var right_hand_side = try linalg.Matrix(f64).fromHost(context, parameter_count, 1, &rhs);
    defer right_hand_side.deinit();
    var solved = try linalg.leastSquares(
        f64,
        coefficients.constView(),
        right_hand_side.constView(),
        .{ .rcond = rank_tolerance },
    );
    defer solved.deinit();
    var step: Parameters = undefined;
    try solved.solution.copyToHost(&step);
    return step;
}

fn fillDampedCosine(output: []f64, time: []const f64, parameters: Parameters) !void {
    if (output.len != time.len) return error.DimensionMismatch;
    try schema.validateModelInput(time, &parameters);
    const amplitude = parameters[0];
    const rate = parameters[1];
    const omega = parameters[2];
    const phase = parameters[3];
    const offset = 1.0 - amplitude * @cos(phase);
    for (time, output) |sample_time, *value| {
        value.* = amplitude * @exp(-rate * sample_time) *
            @cos(omega * sample_time + phase) + offset;
        if (!std.math.isFinite(value.*)) return error.NonFiniteModel;
    }
}

fn objective(
    time: []const f64,
    observed: []const f64,
    weights: []const f64,
    parameters: Parameters,
    scale: f64,
) f64 {
    const amplitude = parameters[0];
    const rate = parameters[1];
    const omega = parameters[2];
    const phase = parameters[3];
    const offset = 1.0 - amplitude * @cos(phase);
    var value: f64 = 0.0;
    for (time, observed, weights) |sample_time, actual, weight| {
        const predicted = amplitude * @exp(-rate * sample_time) *
            @cos(omega * sample_time + phase) + offset;
        const scaled = weight * (predicted - actual) / scale;
        value += 2.0 * scale * scale * (std.math.hypot(1.0, scaled) - 1.0);
    }
    return value;
}

fn finiteState(state: FitState) bool {
    if (!std.math.isFinite(state.objective)) return false;
    for (state.parameters) |value| if (!std.math.isFinite(value)) return false;
    return true;
}

fn uniqueClamped(input: [3]f64, lower: f64, upper: f64) StartValues {
    var output = StartValues{};
    for (input) |raw| {
        const value = clamp(raw, lower, upper);
        var duplicate = false;
        for (output.values[0..output.len]) |existing| {
            if (@as(u64, @bitCast(existing)) == @as(u64, @bitCast(value))) duplicate = true;
        }
        if (!duplicate) {
            output.values[output.len] = value;
            output.len += 1;
        }
    }
    std.mem.sort(f64, output.values[0..output.len], {}, comptime std.sort.asc(f64));
    return output;
}

fn median(sorted: []const f64) f64 {
    const middle = sorted.len / 2;
    return if (sorted.len % 2 == 0)
        0.5 * (sorted[middle - 1] + sorted[middle])
    else
        sorted[middle];
}

fn clamp(value: f64, lower: f64, upper: f64) f64 {
    return @min(@max(value, lower), upper);
}
