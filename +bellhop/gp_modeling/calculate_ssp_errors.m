function error_metrics = calculate_ssp_errors(ssp_gp, true_ssp_grid, s)
    % CALCULATE_SSP_ERRORS - Compute estimation errors for SSP GP model
    %
    % Inputs:
    %   ssp_gp: SSPGaussianProcess object with current estimates
    %   true_ssp_grid: [Ny x Nx x Nz] true SSP field
    %   s: simulation settings structure
    %
    % Output:
    %   error_metrics: struct with various error measures
    
    % Get current SSP estimate from GP
    estimated_ssp_grid = ssp_gp.getCurrentSSPGrid();
    
    % Ensure dimensions match
    if ~isequal(size(estimated_ssp_grid), size(true_ssp_grid))
        error('SSP grid dimensions do not match: Est=%s, True=%s', ...
              mat2str(size(estimated_ssp_grid)), mat2str(size(true_ssp_grid)));
    end
    
    % Compute pointwise errors
    error_field = estimated_ssp_grid - true_ssp_grid;
    absolute_error_field = abs(error_field);
    relative_error_field = absolute_error_field ./ abs(true_ssp_grid);
    
    % Global error metrics
    error_metrics.rmse = sqrt(mean(error_field(:).^2));
    error_metrics.mae = mean(absolute_error_field(:));
    error_metrics.max_error = max(absolute_error_field(:));
    error_metrics.mean_relative_error = mean(relative_error_field(:));
    error_metrics.max_relative_error = max(relative_error_field(:));
    
    % Depth-stratified errors (often important for oceanography)
    [Ny, Nx, Nz] = size(true_ssp_grid);
    error_metrics.rmse_by_depth = zeros(Nz, 1);
    error_metrics.mae_by_depth = zeros(Nz, 1);
    
    for iz = 1:Nz
        depth_errors = error_field(:, :, iz);
        depth_abs_errors = absolute_error_field(:, :, iz);
        error_metrics.rmse_by_depth(iz) = sqrt(mean(depth_errors(:).^2));
        error_metrics.mae_by_depth(iz) = mean(depth_abs_errors(:));
    end
    
    % Spatial coherence metrics
    error_metrics.spatial_correlation = corrcoef(estimated_ssp_grid(:), true_ssp_grid(:));
    error_metrics.spatial_correlation = error_metrics.spatial_correlation(1,2);
    
    % Information metrics
    error_metrics.num_observations = ssp_gp.getNumObservations();
    
    % Get uncertainty information
    grid_uncertainty = ssp_gp.getUncertainty();
    if ~isempty(grid_uncertainty)
        Ny = length(s.Ocean_y_min:s.Ocean_step:s.Ocean_y_max);
        Nx = length(s.Ocean_x_min:s.Ocean_step:s.Ocean_x_max);
        Nz = length(0:s.Ocean_z_step:s.sim_max_depth);
        uncertainty_grid = reshape(grid_uncertainty, [Ny, Nx, Nz]);
        
        error_metrics.mean_uncertainty = mean(uncertainty_grid(:));
        error_metrics.max_uncertainty = max(uncertainty_grid(:));
        
        % Coverage metrics (how often true value falls within uncertainty bounds)
        within_1sigma = abs(error_field) <= uncertainty_grid;
        within_2sigma = abs(error_field) <= 2 * uncertainty_grid;
        error_metrics.coverage_1sigma = mean(within_1sigma(:));
        error_metrics.coverage_2sigma = mean(within_2sigma(:));
    else
        error_metrics.mean_uncertainty = NaN;
        error_metrics.max_uncertainty = NaN;
        error_metrics.coverage_1sigma = NaN;
        error_metrics.coverage_2sigma = NaN;
    end
    
    % Store iteration information
    error_metrics.iteration = ssp_gp.getNumObservations();
    
    % Print summary (optional)
    if nargout == 0 || s.verbose
        fprintf('\n=== SSP Estimation Error Summary (Obs: %d) ===\n', error_metrics.num_observations);
        fprintf('RMSE: %.3f m/s\n', error_metrics.rmse);
        fprintf('MAE: %.3f m/s\n', error_metrics.mae);
        fprintf('Max Error: %.3f m/s\n', error_metrics.max_error);
        fprintf('Mean Relative Error: %.2f%%\n', error_metrics.mean_relative_error * 100);
        fprintf('Spatial Correlation: %.3f\n', error_metrics.spatial_correlation);
        if ~isnan(error_metrics.mean_uncertainty)
            fprintf('Mean Uncertainty: %.3f m/s\n', error_metrics.mean_uncertainty);
            fprintf('Coverage (1σ): %.1f%%\n', error_metrics.coverage_1sigma * 100);
        end
        fprintf('=======================================\n');
    end
end