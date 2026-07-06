# rho_fitting Data Flow

## General rho_fitting with GPU enabled

`py.hexatic.rho_fitting.__main__.main` -> `py.hexatic.rho_fitting.__main__.build_parser`

`py.hexatic.rho_fitting.__main__.main` -> `py.hexatic.rho_fitting.fit.run`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.fit._settings`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.config.radius_from_case_id`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.io.load_active_matter_npz`

`py.hexatic.rho_fitting.io.load_active_matter_npz` -> `py.hexatic.rho_fitting.io._read_radius`

`py.hexatic.rho_fitting.io.load_active_matter_npz` -> `py.hexatic.rho_fitting.io._read_centers`

`py.hexatic.rho_fitting.io.load_active_matter_npz` -> `py.hexatic.rho_fitting.io.validate_active_matter_arrays`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.fit.coarse_grain_active_fields`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.fit._core`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.fit._settings`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.geometry.surface_lengths`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.geometry.theta_to_y`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.particles.particle_tangent_directions`

`py.hexatic.rho_fitting.particles.particle_tangent_directions` -> `py.hexatic.rho_fitting.io.load_gsd_orientations`

`py.hexatic.rho_fitting.particles.particle_tangent_directions` -> `py.hexatic.rho_fitting.io.validate_step_alignment`

`py.hexatic.rho_fitting.particles.particle_tangent_directions` -> `py.hexatic.rho_fitting.geometry.active_direction_from_quaternion`

`py.hexatic.rho_fitting.particles.particle_tangent_directions` -> `py.hexatic.rho_fitting.geometry.tangential_particle_vectors`

`py.hexatic.rho_fitting.particles.particle_surface_velocities` -> `py.hexatic.rho_fitting.geometry.tangential_particle_vectors`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.particles.particle_surface_velocities`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `py.hexatic.rho_fitting.fit._load_hexatic_abs_frames`

`py.hexatic.rho_fitting.fit.coarse_grain_active_fields` -> `rs.python.build_mechanical_fields`

`rs.python.build_mechanical_fields` <-----> `rs.coarse_grain_burn.build_mechanical_fields`

`rs.python.build_mechanical_fields` <-----> `rs.mechanics.build_mechanical_fields`

`rs.coarse_grain_burn.build_mechanical_fields` -> `rs.coarse_grain_burn.catch_burn_panic`

`rs.coarse_grain_burn.catch_burn_panic` -> `rs.coarse_grain_burn.mechanical_burn_device`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.repeated_grid_values`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.tiled_grid_values`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.frame_component`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.frame_scalar`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.frame_mask`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.combine_particle_components`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.tensor2`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.minimum_image_tensor`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.weighted_sum`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.weighted_sum_product`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.tensor_vec`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.write_chunk3`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.write_chunk4`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.write_chunk5`

`rs.coarse_grain_burn.mechanical_burn_device` -> `rs.coarse_grain_burn.write_chunk6`

`rs.mechanics.build_mechanical_fields` -> `rs.mechanics.validation.validate_particle_fields`

`rs.mechanics.build_mechanical_fields` -> `rs.geometry.minimum_image`

`rs.mechanics.build_mechanical_fields` -> `rs.geometry.gaussian_2d`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.fit.spectral_active_fields`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.fit._core`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.fit._settings`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.fit._filter_coarse_field`

`py.hexatic.rho_fitting.fit._filter_coarse_field` -> `py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative`

`py.hexatic.rho_fitting.fit._filter_coarse_field` -> `py.hexatic.rho_fitting.fit._validate_temporal_alignment`

`py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative` -> `py.hexatic.rho_fitting.basis.physical_times`

`py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative` -> `py.hexatic.rho_fitting.basis.validate_cheb_cutoff`

`py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative` -> `py.hexatic.rho_fitting.basis._scaled_times`

`py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative` -> `py.hexatic.rho_fitting.basis._fit_coefficients`

`py.hexatic.rho_fitting.basis.chebyshev_filter_and_derivative` -> `py.hexatic.rho_fitting.basis._values_at_chebyshev_lobatto_nodes`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.geometry.surface_lengths`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `rs.python.build_density_fluxes`

`rs.python.build_density_fluxes` <-----> `rs.library.build_density_fluxes`

`rs.library.build_density_fluxes` -> `rs.fft_ops.gradient_scalar`

`rs.library.build_density_fluxes` -> `rs.fft_ops.laplacian_scalar`

`rs.fft_ops.laplacian_scalar` -> `rs.fft_ops.repeated_laplacian_scalar`

`rs.fft_ops.repeated_laplacian_scalar` -> `rs.fft_ops.scalar_k_power`

`rs.fft_ops.gradient_scalar` -> `rs.fft_ops.validate_scalar`

`rs.fft_ops.gradient_scalar` -> `rs.fft_ops.wavenumbers`

`rs.fft_ops.gradient_scalar` -> `rs.fft_ops.fft2_real`

`rs.fft_ops.gradient_scalar` -> `rs.fft_ops.inverse_with_multiplier`

`rs.fft_ops.fft2_real` -> `rs.fft_ops.fft2_in_place`

`rs.fft_ops.inverse_with_multiplier` -> `rs.fft_ops.inverse_complex`

`rs.fft_ops.inverse_complex` -> `rs.fft_ops.fft2_in_place`

`rs.fft_ops.fft2_in_place` -> `rs.fft_ops.plan_fft`

`rs.library.build_density_fluxes` -> `rs.library.scalar_times_vector`

`rs.library.build_density_fluxes` -> `rs.library.cubic_gradient_flux`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `rs.python.build_mechanical_targets`

`rs.python.build_mechanical_targets` <-----> `rs.mechanics.build_targets`

`py.hexatic.rho_fitting.fit.spectral_active_fields` -> `py.hexatic.rho_fitting.basis.temporal_power_spectrum`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.plots.write_temporal_power_plots`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.fit.fit_mechanical`

`py.hexatic.rho_fitting.fit.fit_mechanical` -> `py.hexatic.rho_fitting.fit._mechanical_sample_indices`

`py.hexatic.rho_fitting.fit._mechanical_sample_indices` -> `py.hexatic.rho_fitting.fit._core`

`py.hexatic.rho_fitting.fit._mechanical_sample_indices` -> `py.hexatic.rho_fitting.fit._settings`

`py.hexatic.rho_fitting.fit._mechanical_sample_indices` -> `py.hexatic.rho_fitting.fit._mechanical_valid_mask`

`py.hexatic.rho_fitting.fit._mechanical_sample_indices` -> `rs.python.sample_rows`

`rs.python.sample_rows` <-----> `rs.sampling.sample_rows`

`rs.sampling.sample_rows` -> `rs.sampling.valid_indices`

`rs.sampling.sample_rows` -> `rs.sampling.seeded_rng`

`py.hexatic.rho_fitting.fit._mechanical_sample_indices` -> `py.hexatic.rho_fitting.fit._validate_sample_count`

`py.hexatic.rho_fitting.fit.fit_mechanical` -> `py.hexatic.rho_fitting.geometry.surface_lengths`

`py.hexatic.rho_fitting.fit.fit_mechanical` -> `py.hexatic.rho_fitting.fit._mechanical_libraries`

`py.hexatic.rho_fitting.fit._mechanical_libraries` -> `py.hexatic.rho_fitting.fit._core`

`py.hexatic.rho_fitting.fit._mechanical_libraries` -> `rs.python.build_mechanical_libraries`

`rs.python.build_mechanical_libraries` <-----> `rs.mechanics.build_mechanical_libraries`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.validation.validate_grid`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.operators.estimate_ubar`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.operators.vector_dot_alpha_traceless`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.build_y_rho_terms`

`rs.mechanics.libraries.build_y_rho_terms` -> `rs.fft_ops.laplacian_scalar`

`rs.mechanics.libraries.build_y_rho_terms` -> `rs.fft_ops.gradient_scalar`

`rs.mechanics.libraries.build_y_rho_terms` -> `rs.mechanics.operators.q_dot_grad_rho`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.build_y_p_terms`

`rs.mechanics.libraries.build_y_p_terms` -> `rs.mechanics.operators.surface_rows_rank2`

`rs.mechanics.libraries.build_y_p_terms` -> `rs.mechanics.libraries.gradient_vector3`

`rs.mechanics.libraries.build_y_p_terms` -> `rs.mechanics.libraries.laplacian_vector3`

`rs.mechanics.libraries.gradient_vector3` -> `rs.mechanics.libraries.scalar_component`

`rs.mechanics.libraries.gradient_vector3` -> `rs.fft_ops.gradient_scalar`

`rs.mechanics.libraries.laplacian_vector3` -> `rs.mechanics.libraries.scalar_component`

`rs.mechanics.libraries.laplacian_vector3` -> `rs.fft_ops.laplacian_scalar`

`rs.mechanics.libraries.build_y_p_terms` -> `rs.mechanics.operators.scalar_times_rank2`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.build_y_q_terms`

`rs.mechanics.libraries.build_y_q_terms` -> `rs.mechanics.operators.scalar_times_rank3`

`rs.mechanics.libraries.build_y_q_terms` -> `rs.mechanics.libraries.gradient_rank2`

`rs.mechanics.libraries.build_y_q_terms` -> `rs.mechanics.libraries.laplacian_rank2`

`rs.mechanics.libraries.build_y_q_terms` -> `rs.mechanics.libraries.grad_p_symmetric_traceless`

`rs.mechanics.libraries.gradient_rank2` -> `rs.mechanics.libraries.rank2_component`

`rs.mechanics.libraries.gradient_rank2` -> `rs.fft_ops.gradient_scalar`

`rs.mechanics.libraries.laplacian_rank2` -> `rs.mechanics.libraries.rank2_component`

`rs.mechanics.libraries.laplacian_rank2` -> `rs.fft_ops.laplacian_scalar`

`rs.mechanics.libraries.grad_p_symmetric_traceless` -> `rs.mechanics.libraries.gradient_vector3`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.stack_vector_terms`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.stack_rank2_terms`

`rs.mechanics.build_mechanical_libraries` -> `rs.mechanics.libraries.stack_rank3_terms`

`py.hexatic.rho_fitting.fit.fit_mechanical` -> `py.hexatic.rho_fitting.fit._fit_divergence_primary_target`

`py.hexatic.rho_fitting.fit._fit_divergence_primary_target` -> `py.hexatic.rho_fitting.fit._settings`

`py.hexatic.rho_fitting.fit._fit_divergence_primary_target` -> `py.hexatic.rho_fitting.fit._sample_component_matrix`

`py.hexatic.rho_fitting.fit._sample_component_matrix` -> `py.hexatic.rho_fitting.fit._core`

`py.hexatic.rho_fitting.fit._sample_component_matrix` -> `rs.python.sample_component_rows`

`rs.python.sample_component_rows` <-----> `rs.mechanics.sample_component_rows`

`rs.mechanics.sample_component_rows` -> `rs.mechanics.sampling.checked_index`

`rs.mechanics.sample_component_rows` -> `rs.mechanics.sampling.unravel_component`

`py.hexatic.rho_fitting.fit._fit_divergence_primary_target` -> `py.hexatic.rho_fitting.fit._sample_divergence_matrix`

`py.hexatic.rho_fitting.fit._sample_divergence_matrix` -> `py.hexatic.rho_fitting.fit._divergence_surface_flux_field`

`py.hexatic.rho_fitting.fit._sample_divergence_matrix` -> `py.hexatic.rho_fitting.fit._sample_scalar`

`py.hexatic.rho_fitting.fit._sample_divergence_matrix` -> `py.hexatic.rho_fitting.fit._sample_component_matrix`

`py.hexatic.rho_fitting.fit._fit_divergence_primary_target` -> `py.hexatic.rho_fitting.library.mechanical_labels`

`py.hexatic.rho_fitting.fit._fit_divergence_primary_target` -> `py.hexatic.rho_fitting.regression.stability_selection`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._validate_fit_inputs`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression.tau_path`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._fit_tau_path`

`py.hexatic.rho_fitting.regression._fit_tau_path` -> `py.hexatic.rho_fitting.regression._sr3_coefficients`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._coefficient_importance`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._evaluation_rows`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._auxiliary_rows`

`py.hexatic.rho_fitting.regression.stability_selection` -> `py.hexatic.rho_fitting.regression._regression_metrics`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.fit._overwrite_config`

`py.hexatic.rho_fitting.fit.run` -> `py.hexatic.rho_fitting.outputs.write_mechanical_outputs`

`py.hexatic.rho_fitting.outputs.write_mechanical_outputs` -> `py.hexatic.rho_fitting.outputs.cache_metadata`

`py.hexatic.rho_fitting.outputs.write_mechanical_outputs` -> `py.hexatic.rho_fitting.cache.write_npz_atomic`

`py.hexatic.rho_fitting.outputs.write_mechanical_outputs` -> `py.hexatic.rho_fitting.outputs.mechanical_report_lines`

`py.hexatic.rho_fitting.outputs.write_mechanical_outputs` -> `py.hexatic.rho_fitting.report.write_report`

## rho_fitting with PDE validation

`py.hexatic.rho_fitting.pde_validation.__main__.main` -> `py.hexatic.rho_fitting.pde_validation.__main__.build_parser`

`py.hexatic.rho_fitting.pde_validation.__main__.main` -> `py.hexatic.rho_fitting.pde_validation.model.run_validation_from_cache`

`py.hexatic.rho_fitting.pde_validation.model.run_validation_from_cache` -> `py.hexatic.rho_fitting.pde_validation.cache.load_validation_inputs`

`py.hexatic.rho_fitting.pde_validation.cache.load_validation_inputs` -> `py.hexatic.rho_fitting.pde_validation.cache._validate_library_names`

`py.hexatic.rho_fitting.pde_validation.cache.load_validation_inputs` -> `py.hexatic.rho_fitting.pde_validation.cache._load_field_arrays`

`py.hexatic.rho_fitting.pde_validation.cache._load_field_arrays` -> `py.hexatic.rho_fitting.pde_validation.cache._load_array`

`py.hexatic.rho_fitting.pde_validation.cache.load_validation_inputs` -> `py.hexatic.rho_fitting.pde_validation.cache._load_array`

`py.hexatic.rho_fitting.pde_validation.model.run_validation_from_cache` -> `py.hexatic.rho_fitting.pde_validation.model.run_validation`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model.make_grid`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.__init__`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.__init__` -> `py.hexatic.rho_fitting.pde_validation.model._scalar_bounds`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.__init__` -> `py.hexatic.rho_fitting.pde_validation.model._symmetric_bound`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model.pack_state`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state` -> `py.hexatic.rho_fitting.pde_validation.model.unpack_state`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_fields`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state` -> `py.hexatic.rho_fitting.pde_validation.model.pack_state`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model._run_solver`

`py.hexatic.rho_fitting.pde_validation.model._run_solver` <-----> `py.hexatic.rho_fitting.pde_validation.model._run_euler`

`py.hexatic.rho_fitting.pde_validation.model._run_solver` <-----> `py.hexatic.rho_fitting.pde_validation.model._run_scipy`

`py.hexatic.rho_fitting.pde_validation.model._run_euler` -> `py.hexatic.rho_fitting.pde_validation.model.unpack_state`

`py.hexatic.rho_fitting.pde_validation.model._run_euler` -> `py.hexatic.rho_fitting.pde_validation.model._step_full_state`

`py.hexatic.rho_fitting.pde_validation.model._step_full_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate`

`py.hexatic.rho_fitting.pde_validation.model._step_full_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.stable_rate_data`

`py.hexatic.rho_fitting.pde_validation.model._step_full_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state`

`py.hexatic.rho_fitting.pde_validation.model._run_euler` -> `py.hexatic.rho_fitting.pde_validation.model._step_single_field_state`

`py.hexatic.rho_fitting.pde_validation.model._step_single_field_state` -> `py.hexatic.rho_fitting.pde_validation.model._single_field_reference_state`

`py.hexatic.rho_fitting.pde_validation.model._single_field_reference_state` -> `py.hexatic.rho_fitting.pde_validation.model.interpolated_fields`

`py.hexatic.rho_fitting.pde_validation.model.interpolated_fields` -> `py.hexatic.rho_fitting.pde_validation.model.interpolated_cached_fields`

`py.hexatic.rho_fitting.pde_validation.model.interpolated_cached_fields` -> `py.hexatic.rho_fitting.pde_validation.model.interpolate_time_series`

`py.hexatic.rho_fitting.pde_validation.model.interpolate_time_series` -> `py.hexatic.rho_fitting.pde_validation.model.interpolation_index_weight`

`py.hexatic.rho_fitting.pde_validation.model._single_field_reference_state` -> `py.hexatic.rho_fitting.pde_validation.model.unpack_state`

`py.hexatic.rho_fitting.pde_validation.model._single_field_reference_state` -> `py.hexatic.rho_fitting.pde_validation.model.pack_state`

`py.hexatic.rho_fitting.pde_validation.model._step_single_field_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate`

`py.hexatic.rho_fitting.pde_validation.model._step_single_field_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.stable_rate_data`

`py.hexatic.rho_fitting.pde_validation.model._step_single_field_state` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_state`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.model.unpack_state`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.project_fields`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.filtered_fields`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.fit_time_ubar`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.fit_time_ubar` -> `py.hexatic.rho_fitting.pde_validation.model.interpolated_cached_fields`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.fit_time_ubar` -> `py.hexatic.rho_fitting.pde_validation.operators.estimate_ubar`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.operators.closure_fields`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_scalar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.laplacian_scalar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.q_dot_grad_rho`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_vector`

`py.hexatic.rho_fitting.pde_validation.operators.gradient_vector` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_scalar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.laplacian_vector`

`py.hexatic.rho_fitting.pde_validation.operators.laplacian_vector` -> `py.hexatic.rho_fitting.pde_validation.operators.laplacian_scalar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.estimate_ubar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.p_dot_alpha_traceless`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.grad_p_symmetric_traceless`

`py.hexatic.rho_fitting.pde_validation.operators.grad_p_symmetric_traceless` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_vector`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_rank2`

`py.hexatic.rho_fitting.pde_validation.operators.gradient_rank2` -> `py.hexatic.rho_fitting.pde_validation.operators.gradient_scalar`

`py.hexatic.rho_fitting.pde_validation.operators.closure_fields` -> `py.hexatic.rho_fitting.pde_validation.operators.laplacian_rank2`

`py.hexatic.rho_fitting.pde_validation.operators.laplacian_rank2` -> `py.hexatic.rho_fitting.pde_validation.operators.laplacian_scalar`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.operators.divergence_vector`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.operators.divergence_surface_flux`

`py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate` -> `py.hexatic.rho_fitting.pde_validation.model.pack_state`

`py.hexatic.rho_fitting.pde_validation.model._run_scipy` -> `py.hexatic.rho_fitting.pde_validation.model.RhoFitPDE.evolution_rate`

`py.hexatic.rho_fitting.pde_validation.model._run_solver` -> `py.hexatic.rho_fitting.pde_validation.model._run_scipy`

`py.hexatic.rho_fitting.pde_validation.model.run_validation` -> `py.hexatic.rho_fitting.pde_validation.model.validation_metric_arrays`

`py.hexatic.rho_fitting.pde_validation.__main__.main` -> `py.hexatic.rho_fitting.cache.write_npz_atomic`

`py.hexatic.rho_fitting.pde_validation.__main__.main` -> `py.hexatic.rho_fitting.pde_validation.__main__.write_mode_outputs`

`py.hexatic.rho_fitting.pde_validation.__main__.write_mode_outputs` -> `py.hexatic.rho_fitting.pde_validation.plot.write_rho_animation`

`py.hexatic.rho_fitting.pde_validation.__main__.write_mode_outputs` -> `py.hexatic.rho_fitting.pde_validation.plot.write_p_animation`

`py.hexatic.rho_fitting.pde_validation.__main__.write_mode_outputs` -> `py.hexatic.rho_fitting.pde_validation.plot.write_q_animation`

`py.hexatic.rho_fitting.pde_validation.plot.write_rho_animation` -> `py.hexatic.rho_fitting.pde_validation.plot.write_scalar_animation`

`py.hexatic.rho_fitting.pde_validation.plot.write_p_animation` -> `py.hexatic.rho_fitting.pde_validation.plot.write_component_animation`

`py.hexatic.rho_fitting.pde_validation.plot.write_q_animation` -> `py.hexatic.rho_fitting.pde_validation.plot.write_component_animation`

`py.hexatic.rho_fitting.pde_validation.plot.write_component_animation` -> `py.hexatic.rho_fitting.pde_validation.plot.surface_axes`

`py.hexatic.rho_fitting.pde_validation.plot.write_scalar_animation` -> `py.hexatic.rho_fitting.pde_validation.plot.surface_axes`
