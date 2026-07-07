"""py-pde JAX/MPS backend implementation for validation rollouts."""

from __future__ import annotations

import os
from typing import Any

import numpy as np
from pde.backends import get_backend

from .types import Array, RELAXATION_COEFFICIENT


def make_jax_evolution_rate(pde: Any, state: Any, backend: Any) -> Any:
    """Return a backend-native RHS for py-pde's JAX backend."""
    if backend.implementation != "jax":
        raise NotImplementedError("rho-fitting validation only implements a custom compiled RHS for the jax backend")

    import jax.numpy as jnp

    f32 = np.float32
    dx = f32(pde.dx)
    dtheta = f32(pde.dtheta)
    u0 = f32(pde.inputs.u0)
    gamma = f32(pde.inputs.gamma)
    relaxation = f32(RELAXATION_COEFFICIENT)
    rho_min = f32(pde.rho_min)
    rho_max = f32(pde.rho_max)
    p_limit = f32(pde.p_limit)
    q_limit = f32(pde.q_limit)
    rate_clip_dt = None if pde.rate_clip_dt is None else f32(pde.rate_clip_dt)
    r_centers = backend.numpy_to_native(np.asarray(pde.inputs.r_centers, dtype=np.float32))
    times = backend.numpy_to_native(np.asarray(pde.inputs.times, dtype=np.float32))
    a_cache = backend.numpy_to_native(np.asarray(pde.inputs.a, dtype=np.float32))
    y_p_cache = backend.numpy_to_native(np.asarray(pde.inputs.y_p, dtype=np.float32))
    psi6_sq_fixed = backend.numpy_to_native(np.asarray(pde.psi6_sq_fixed, dtype=np.float32))
    c_rho = backend.numpy_to_native(np.asarray(pde.inputs.y_rho_coefficients, dtype=np.float32))
    c_p = backend.numpy_to_native(np.asarray(pde.inputs.y_p_coefficients, dtype=np.float32))
    c_q = backend.numpy_to_native(np.asarray(pde.inputs.y_q_coefficients, dtype=np.float32))
    identity = backend.numpy_to_native(np.eye(3, dtype=np.float32))
    filter_kernels = jax_filter_kernels(pde, backend)

    def radial_spacing() -> Any:
        return r_centers[1] - r_centers[0]

    def radial_broadcast(ndim: int) -> Any:
        return r_centers.reshape((1, 1, r_centers.shape[0]) + (1,) * (ndim - 3))

    def gradient_scalar(values: Any) -> Any:
        dr = radial_spacing()
        grad_x = (jnp.roll(values, -1, axis=0) - jnp.roll(values, 1, axis=0)) / (f32(2.0) * dx)
        theta_derivative = (jnp.roll(values, -1, axis=1) - jnp.roll(values, 1, axis=1)) / (f32(2.0) * dtheta)
        grad_theta = theta_derivative / radial_broadcast(values.ndim)
        first = ((values[:, :, 1, ...] - values[:, :, 0, ...]) / dr)[:, :, None, ...]
        middle = (values[:, :, 2:, ...] - values[:, :, :-2, ...]) / (f32(2.0) * dr)
        last = ((values[:, :, -1, ...] - values[:, :, -2, ...]) / dr)[:, :, None, ...]
        grad_r = jnp.concatenate((first, middle, last), axis=2)
        return jnp.stack((grad_x, grad_theta, grad_r), axis=3)

    def divergence_vector(values: Any) -> Any:
        dr = radial_spacing()
        flux_x = values[:, :, :, 0, ...]
        flux_theta = values[:, :, :, 1, ...]
        flux_r = values[:, :, :, 2, ...]
        r_scale = r_centers.reshape((1, 1, r_centers.shape[0]) + (1,) * (values.ndim - 4))
        div_x = (jnp.roll(flux_x, -1, axis=0) - jnp.roll(flux_x, 1, axis=0)) / (f32(2.0) * dx)
        div_theta = (jnp.roll(flux_theta, -1, axis=1) - jnp.roll(flux_theta, 1, axis=1)) / (f32(2.0) * dtheta * r_scale)
        weighted = r_scale * flux_r
        faces = jnp.zeros(weighted.shape[:2] + (weighted.shape[2] + 1,) + weighted.shape[3:], dtype=f32)
        faces = faces.at[:, :, 1:-1, ...].set(f32(0.5) * (weighted[:, :, :-1, ...] + weighted[:, :, 1:, ...]))
        div_r = (faces[:, :, 1:, ...] - faces[:, :, :-1, ...]) / (dr * r_scale)
        return div_x + div_theta + div_r

    def laplacian_scalar(values: Any) -> Any:
        return divergence_vector(gradient_scalar(values))

    def gradient_vector(values: Any) -> Any:
        return jnp.stack([gradient_scalar(values[..., component]) for component in range(3)], axis=-1)

    def laplacian_vector(values: Any) -> Any:
        return jnp.stack([laplacian_scalar(values[..., component]) for component in range(3)], axis=-1)

    def gradient_rank2(values: Any) -> Any:
        rows = []
        for a in range(3):
            cols = []
            for b in range(3):
                cols.append(gradient_scalar(values[..., a, b]))
            rows.append(jnp.stack(cols, axis=-1))
        return jnp.stack(rows, axis=-2)

    def project_flux_directions(values: Any, mode: str) -> Any:
        out = jnp.zeros_like(values)
        if mode == "tangential":
            return out.at[..., 0, :, :].set(values[..., 0, :, :]).at[..., 1, :, :].set(values[..., 1, :, :])
        return out.at[..., 2, :, :].set(values[..., 2, :, :])

    def q_dot_grad_rho(q: Any, grad_rho: Any) -> Any:
        return jnp.einsum("...ka,...a->...k", q, grad_rho)

    def p_dot_alpha_traceless(p: Any) -> Any:
        out = jnp.zeros(p.shape[:-1] + (3, 3, 3), dtype=f32)
        for k in range(3):
            for a in range(3):
                for b in range(3):
                    value = p[..., a] * f32(k == b) + p[..., b] * f32(k == a) - f32(2.0 / 3.0) * p[..., k] * identity[a, b]
                    out = out.at[..., k, a, b].set(value)
        return out

    def estimate_ubar_jax(y_p: Any, a: Any) -> Any:
        denominator = jnp.sum(a * a, axis=(-2, -1))
        numerator = jnp.sum(y_p * a, axis=(-2, -1))
        return jnp.where(denominator > f32(0.0), numerator / denominator, f32(0.0))

    def interpolation_index_weight(t: Any) -> tuple[Any, Any]:
        index = jnp.searchsorted(times, t, side="right") - 1
        index = jnp.clip(index, 0, times.shape[0] - 2)
        t0 = times[index]
        t1 = times[index + 1]
        weight = jnp.where(t1 == t0, f32(0.0), (t - t0) / (t1 - t0))
        return index, weight

    def interpolate_time_series(values: Any, t: Any) -> Any:
        index, weight = interpolation_index_weight(t)
        return (f32(1.0) - weight) * values[index] + weight * values[index + 1]

    def filtered_fields(rho: Any, p: Any, q: Any) -> tuple[Any, Any, Any]:
        if filter_kernels is None:
            return rho, p, q
        kernel_x, kernel_theta, kernel_r = filter_kernels
        rho_out = jax_gaussian_filter(rho, kernel_x, kernel_theta, kernel_r)
        p_out = jax_gaussian_filter(p, kernel_x, kernel_theta, kernel_r)
        q_out = jax_gaussian_filter(q, kernel_x, kernel_theta, kernel_r)
        return rho_out, p_out, q_out

    def project_fields(rho: Any, p: Any, q: Any) -> tuple[Any, Any, Any]:
        rho_out = jnp.nan_to_num(rho, nan=rho_min, posinf=rho_max, neginf=rho_min)
        p_out = jnp.nan_to_num(p, nan=f32(0.0), posinf=p_limit, neginf=-p_limit)
        q_out = jnp.nan_to_num(q, nan=f32(0.0), posinf=q_limit, neginf=-q_limit)
        return (
            jnp.clip(rho_out, rho_min, rho_max),
            jnp.clip(p_out, -p_limit, p_limit),
            jnp.clip(q_out, -q_limit, q_limit),
        )

    def unpack_data(data: Any) -> tuple[Any, Any, Any]:
        rho = data[0]
        p = jnp.moveaxis(data[1:4], 0, -1)
        q = jnp.moveaxis(data[4:].reshape(3, 3, *rho.shape), (0, 1), (-2, -1))
        return rho, p, q

    def pack_data(rho: Any, p: Any, q: Any) -> Any:
        p_data = jnp.moveaxis(p, -1, 0)
        q_data = jnp.moveaxis(q, (-2, -1), (0, 1)).reshape(9, *rho.shape)
        return jnp.concatenate((rho[None, ...], p_data, q_data), axis=0)

    def closure_fields_jax(rho: Any, p: Any, q: Any, psi6_sq: Any, ubar: Any) -> tuple[Any, Any, Any, Any]:
        a = q + rho[..., None, None] * identity / f32(3.0)
        grad_rho = gradient_scalar(rho)
        grad_lap_rho = gradient_scalar(laplacian_scalar(rho))
        f_rho = (
            c_rho[0] * grad_rho
            + c_rho[1] * grad_lap_rho
            + c_rho[2] * q_dot_grad_rho(q, grad_rho)
            + c_rho[3] * p
        )

        grad_p = gradient_vector(p)
        grad_lap_p = gradient_vector(laplacian_vector(p))
        f_p = (
            c_p[0] * a
            + c_p[1] * rho[..., None, None] * a
            + c_p[2] * psi6_sq[..., None, None] * a
            + c_p[3] * grad_p
            + c_p[4] * rho[..., None, None] * grad_p
            + c_p[5] * grad_lap_p
        )

        grad_q = gradient_rank2(q)
        ubar_p_alpha = ubar[..., None, None, None] * p_dot_alpha_traceless(p)
        f_q = (
            c_q[0] * project_flux_directions(ubar_p_alpha, "tangential")
            + c_q[1] * project_flux_directions(ubar_p_alpha, "radial")
            + c_q[2] * project_flux_directions(grad_q, "tangential")
            + c_q[3] * project_flux_directions(grad_q, "radial")
        )
        s_q = c_q[4] * q + c_q[5] * psi6_sq[..., None, None] * q
        return f_rho, f_p, f_q, s_q

    def stable_rate_data(rate_data: Any) -> Any:
        if rate_clip_dt is None:
            return jnp.nan_to_num(rate_data, nan=f32(0.0), posinf=f32(0.0), neginf=f32(0.0))
        values = jnp.nan_to_num(rate_data, nan=f32(0.0), posinf=f32(0.0), neginf=f32(0.0))
        rho_limit = (rho_max - rho_min) / rate_clip_dt
        p_rate_limit = (f32(2.0) * p_limit) / rate_clip_dt
        q_rate_limit = (f32(2.0) * q_limit) / rate_clip_dt
        return jnp.concatenate(
            (
                jnp.clip(values[0:1], -rho_limit, rho_limit),
                jnp.clip(values[1:4], -p_rate_limit, p_rate_limit),
                jnp.clip(values[4:], -q_rate_limit, q_rate_limit),
            ),
            axis=0,
        )

    def rhs(data: Any, t: Any) -> Any:
        rho, p, q = unpack_data(data)
        rho, p, q = project_fields(rho, p, q)
        rho_eval, p_eval, q_eval = filtered_fields(rho, p, q)
        rho_eval, p_eval, q_eval = project_fields(rho_eval, p_eval, q_eval)
        a = interpolate_time_series(a_cache, t)
        y_p = interpolate_time_series(y_p_cache, t)
        ubar = estimate_ubar_jax(y_p, a)
        f_rho, f_p, f_q, s_q = closure_fields_jax(rho_eval, p_eval, q_eval, psi6_sq_fixed, ubar)
        rho_flux = u0 * p_eval + f_rho / gamma
        d_rho = -divergence_vector(rho_flux)
        d_p = -u0 * divergence_vector(f_p) + relaxation * p
        d_q = -divergence_vector(f_q) + s_q + relaxation * q
        return stable_rate_data(pack_data(d_rho, d_p, d_q))

    return rhs


def run_jax_py_pde_euler(
    pde: Any,
    state: Any,
    times: Array,
    dt: float,
    mode: str,
    jax_device: str,
) -> tuple[Array, Array, Array]:
    """Run explicit Euler with py-pde's JAX backend and cached-field single modes."""
    assert mode in {"full", "rho-only", "p-only", "q-only"}, f"unknown validation mode: {mode}"
    ensure_jax_mps_loaded(jax_device)
    backend_name = f"jax:{jax_device}" if jax_device else "jax"
    try:
        backend = get_backend(backend_name)
        assert_requested_jax_device(backend, jax_device)
        rhs = pde.make_pde_rhs(state, backend=backend)
        data = backend.numpy_to_native(np.asarray(state.data, dtype=np.float32))
        times_native = backend.numpy_to_native(np.asarray(times, dtype=np.float32))
        rho_cache = backend.numpy_to_native(np.asarray(pde.inputs.rho, dtype=np.float32))
        p_cache = backend.numpy_to_native(np.asarray(pde.inputs.p, dtype=np.float32))
        q_cache = backend.numpy_to_native(np.asarray(pde.inputs.q, dtype=np.float32))
        post_filter_kernels = jax_filter_kernels(pde, backend)
    except RuntimeError as err:
        raise_clear_jax_mps_error(err)

    import jax
    import jax.numpy as jnp

    f32 = np.float32
    dt = float(dt)
    assert dt > 0.0, "dt must be positive"
    rho_fit = []
    p_fit = []
    q_fit = []

    def unpack_data(native_data: Any) -> tuple[Any, Any, Any]:
        rho = native_data[0]
        p = jnp.moveaxis(native_data[1:4], 0, -1)
        q = jnp.moveaxis(native_data[4:].reshape(3, 3, *rho.shape), (0, 1), (-2, -1))
        return rho, p, q

    def pack_data(rho: Any, p: Any, q: Any) -> Any:
        p_data = jnp.moveaxis(p, -1, 0)
        q_data = jnp.moveaxis(q, (-2, -1), (0, 1)).reshape(9, *rho.shape)
        return jnp.concatenate((rho[None, ...], p_data, q_data), axis=0)

    def project_data(native_data: Any) -> Any:
        rho, p, q = unpack_data(native_data)
        rho, p, q = jax_project_fields(rho, p, q, pde.rho_min, pde.rho_max, pde.p_limit, pde.q_limit)
        return pack_data(rho, p, q)

    def filter_project_data(native_data: Any) -> Any:
        if post_filter_kernels is None:
            return project_data(native_data)
        rho, p, q = unpack_data(native_data)
        kernel_x, kernel_theta, kernel_r = post_filter_kernels
        rho = jax_gaussian_filter(rho, kernel_x, kernel_theta, kernel_r)
        p = jax_gaussian_filter(p, kernel_x, kernel_theta, kernel_r)
        q = jax_gaussian_filter(q, kernel_x, kernel_theta, kernel_r)
        rho, p, q = jax_project_fields(rho, p, q, pde.rho_min, pde.rho_max, pde.p_limit, pde.q_limit)
        return pack_data(rho, p, q)

    def interpolation_index_weight(t: Any) -> tuple[Any, Any]:
        index = jnp.searchsorted(times_native, t, side="right") - 1
        index = jnp.clip(index, 0, times_native.shape[0] - 2)
        t0 = times_native[index]
        t1 = times_native[index + 1]
        weight = jnp.where(t1 == t0, f32(0.0), (t - t0) / (t1 - t0))
        return index, weight

    def interpolate_time_series(values: Any, t: Any) -> Any:
        index, weight = interpolation_index_weight(t)
        return (f32(1.0) - weight) * values[index] + weight * values[index + 1]

    def reference_data(native_data: Any, t: Any, run_mode: str) -> Any:
        rho_ref = interpolate_time_series(rho_cache, t)
        p_ref = interpolate_time_series(p_cache, t)
        q_ref = interpolate_time_series(q_cache, t)
        rho_live, p_live, q_live = unpack_data(native_data)
        rho_eval = rho_live if run_mode == "rho-only" else rho_ref
        p_eval = p_live if run_mode == "p-only" else p_ref
        q_eval = q_live if run_mode == "q-only" else q_ref
        return pack_data(rho_eval, p_eval, q_eval)

    @jax.jit
    def full_step(native_data: Any, t: Any, step_dt: Any) -> Any:
        return filter_project_data(native_data + step_dt * rhs(native_data, t))

    @jax.jit
    def rho_step(native_data: Any, t: Any, step_dt: Any) -> Any:
        hard_data = reference_data(native_data, t, "rho-only")
        rate = rhs(hard_data, t)
        out = hard_data.at[0].set(native_data[0] + step_dt * rate[0])
        return filter_project_data(out)

    @jax.jit
    def p_step(native_data: Any, t: Any, step_dt: Any) -> Any:
        hard_data = reference_data(native_data, t, "p-only")
        out = hard_data.at[1:4].set(native_data[1:4] + step_dt * rhs(hard_data, t)[1:4])
        return filter_project_data(out)

    @jax.jit
    def q_step(native_data: Any, t: Any, step_dt: Any) -> Any:
        hard_data = reference_data(native_data, t, "q-only")
        out = hard_data.at[:4].set(hard_data[:4])
        out = out.at[4:].set(native_data[4:] + step_dt * rhs(hard_data, t)[4:])
        return filter_project_data(out)

    stepper = {
        "full": full_step,
        "rho-only": rho_step,
        "p-only": p_step,
        "q-only": q_step,
    }[mode]

    try:
        for index in range(times.size):
            rho, p, q = unpack_data(data)
            rho_fit.append(rho)
            p_fit.append(p)
            q_fit.append(q)
            if index == times.size - 1:
                break
            frame_dt = float(times[index + 1] - times[index])
            assert frame_dt > 0.0, "validation times must be strictly increasing"
            substeps = max(1, int(np.ceil(frame_dt / dt)))
            step_dt = np.float32(frame_dt / substeps)
            for substep in range(substeps):
                t = np.float32(float(times[index]) + substep * float(step_dt))
                data = stepper(data, t, step_dt)
        rho_out = np.asarray(jax.device_get(jnp.stack(rho_fit)), dtype=np.float32)
        p_out = np.asarray(jax.device_get(jnp.stack(p_fit)), dtype=np.float32)
        q_out = np.asarray(jax.device_get(jnp.stack(q_fit)), dtype=np.float32)
    except RuntimeError as err:
        raise_clear_jax_mps_error(err)
    return rho_out, p_out, q_out


def ensure_jax_mps_loaded(jax_device: str) -> None:
    """Import jax-mps before py-pde constructs an MPS JAX backend."""
    if jax_device != "mps":
        return
    os.environ.setdefault("JAX_MPS_ASYNC_DISPATCH", "1")
    try:
        import jax_plugins.mps  # noqa: F401
    except ModuleNotFoundError as err:
        raise RuntimeError("jax-mps is required for --backend jax --jax-device mps; run `pixi install`") from err


def assert_requested_jax_device(backend: Any, jax_device: str) -> None:
    """Fail early when py-pde/JAX is not using the requested accelerator."""
    if not jax_device:
        return
    requested = jax_device.split(":", 1)[0].lower()
    actual = str(getattr(backend.device, "platform", "")).lower()
    if requested == "gpu":
        requested = "cuda"
    if requested == "mps":
        requested = "mps"
    assert actual == requested, f"requested jax:{jax_device}, but py-pde selected JAX device {backend.device!r}"
    try:
        import jax.numpy as jnp

        probe = backend.numpy_to_native(np.asarray([1.0, 2.0, 3.0], dtype=np.float32))
        native_sum = jnp.sum(probe * probe)
        host_sum = float(backend.native_to_numpy(native_sum))
    except RuntimeError as err:
        raise_clear_jax_mps_error(err)
    assert abs(host_sum - 14.0) < 1.0e-5, f"JAX device probe failed on {backend.device!r}: got {host_sum}"


def raise_clear_jax_mps_error(err: RuntimeError) -> None:
    """Raise a clearer message for common sandboxed/headless Metal failures."""
    if "metal::load_device" in str(err) or "No Metal device available" in str(err):
        raise RuntimeError(
            "jax-mps is installed and registered, but Metal execution is not available "
            "in this process. Run this command from a non-sandboxed macOS session, "
            "or pass --backend py-pde to use the original CPU validation path."
        ) from err
    raise err


def jax_project_fields(
    rho: Any,
    p: Any,
    q: Any,
    rho_min: float,
    rho_max: float,
    p_limit: float,
    q_limit: float,
) -> tuple[Any, Any, Any]:
    """Clip JAX validation fields into the same bounds used by the NumPy path."""
    import jax.numpy as jnp

    f32 = np.float32
    rho_min_j = f32(rho_min)
    rho_max_j = f32(rho_max)
    p_limit_j = f32(p_limit)
    q_limit_j = f32(q_limit)
    rho_out = jnp.nan_to_num(rho, nan=rho_min_j, posinf=rho_max_j, neginf=rho_min_j)
    p_out = jnp.nan_to_num(p, nan=f32(0.0), posinf=p_limit_j, neginf=-p_limit_j)
    q_out = jnp.nan_to_num(q, nan=f32(0.0), posinf=q_limit_j, neginf=-q_limit_j)
    return (
        jnp.clip(rho_out, rho_min_j, rho_max_j),
        jnp.clip(p_out, -p_limit_j, p_limit_j),
        jnp.clip(q_out, -q_limit_j, q_limit_j),
    )


def jax_filter_kernels(pde: Any, backend: Any) -> tuple[Any, Any, Any] | None:
    """Build float32 JAX Gaussian kernels for the filtered Euler RHS."""
    if pde.filter_sigma is None or pde.filter_sigma <= 0.0:
        return None
    sx = pde.filter_sigma / pde.dx
    stheta = pde.filter_sigma / (float(np.mean(pde.inputs.r_centers)) * pde.dtheta)
    sr = pde.filter_sigma / float(np.mean(np.diff(pde.inputs.r_centers)))
    return (
        backend.numpy_to_native(gaussian_kernel_1d(sx)),
        backend.numpy_to_native(gaussian_kernel_1d(stheta)),
        backend.numpy_to_native(gaussian_kernel_1d(sr)),
    )


def gaussian_kernel_1d(sigma_cells: float) -> Array:
    """Return a normalized 1D Gaussian kernel in float32 cell units."""
    if sigma_cells <= 0.0:
        return np.ones((1,), dtype=np.float32)
    radius = max(1, int(np.ceil(3.0 * sigma_cells)))
    offsets = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (offsets / np.float32(sigma_cells)) ** 2).astype(np.float32)
    kernel /= np.sum(kernel, dtype=np.float32)
    return kernel


def jax_gaussian_filter(values: Any, kernel_x: Any, kernel_theta: Any, kernel_r: Any) -> Any:
    """Apply separable x/theta periodic and radial nearest Gaussian filtering in JAX."""
    out = jax_filter_axis(values, kernel_x, axis=0, periodic=True)
    out = jax_filter_axis(out, kernel_theta, axis=1, periodic=True)
    return jax_filter_axis(out, kernel_r, axis=2, periodic=False)


def jax_filter_axis(values: Any, kernel: Any, *, axis: int, periodic: bool) -> Any:
    """Apply one fixed 1D kernel along an axis using JAX array operations."""
    import jax.numpy as jnp

    radius = int(kernel.shape[0] // 2)
    out = jnp.zeros_like(values)
    for index, offset in enumerate(range(-radius, radius + 1)):
        if periodic:
            shifted = jnp.roll(values, offset, axis=axis)
        else:
            shifted = jax_shift_nearest(values, offset, axis=axis)
        out = out + kernel[index] * shifted
    return out


def jax_shift_nearest(values: Any, offset: int, *, axis: int) -> Any:
    """Shift an array with nearest-value extension instead of wraparound."""
    import jax.numpy as jnp

    size = values.shape[axis]
    indices = jnp.clip(jnp.arange(size) - offset, 0, size - 1)
    return jnp.take(values, indices, axis=axis)
