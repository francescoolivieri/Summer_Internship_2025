function update_visualization(figure_handles, data, errors, s, iter)
    % UPDATE_VISUALIZATION - Update real-time visualization of SSP estimation
    %
    % Inputs:
    %   figure_handles: struct with figure handles
    %   data: simulation data structure
    %   errors: array of error metrics
    %   s: simulation settings
    %   iter: current iteration number
    
    % Set current figure
    figure(figure_handles.main);
    
    % Clear and setup subplots
    clf;
    
    % Get current estimates
    true_ssp = data.true_params.get('ssp_grid');
    est_ssp = data.ssp_gp.getCurrentSSPGrid();
    uncertainty = data.ssp_gp.getUncertainty(); % Full grid uncertainty
    
    % Get observations
    [obs_pos, obs_vals] = data.ssp_gp.getObservations();
    
    % Setup grid coordinates
    x_coords = s.Ocean_x_min:s.Ocean_step:s.Ocean_x_max;
    y_coords = s.Ocean_y_min:s.Ocean_step:s.Ocean_y_max;
    z_coords = 0:s.Ocean_z_step:s.sim_max_depth;
    
    % Choose middle depth for 2D visualization
    mid_z_idx = round(length(z_coords) / 2);
    mid_depth = z_coords(mid_z_idx);
    
    % Reshape uncertainty for visualization
    if ~isempty(uncertainty)
        Ny = length(y_coords);
        Nx = length(x_coords);
        Nz = length(z_coords);
        uncertainty_grid = reshape(uncertainty, [Ny, Nx, Nz]);
    end
    
    % 1. True SSP field (2D slice)
    subplot(2, 3, 1);
    imagesc(x_coords, y_coords, true_ssp(:, :, mid_z_idx));
    colorbar;
    title(sprintf('True SSP at %.0fm depth', mid_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy; % Correct orientation
    
    % Add observation points if any
    if ~isempty(obs_pos)
        hold on;
        obs_at_depth = obs_pos(abs(obs_pos(:,3) - mid_depth) < s.Ocean_z_step, :);
        if ~isempty(obs_at_depth)
            plot(obs_at_depth(:,1), obs_at_depth(:,2), 'wo', 'MarkerSize', 8, 'LineWidth', 2);
        end
        hold off;
    end
    
    % 2. Estimated SSP field (2D slice)
    subplot(2, 3, 2);
    imagesc(x_coords, y_coords, est_ssp(:, :, mid_z_idx));
    colorbar;
    title(sprintf('Estimated SSP at %.0fm depth', mid_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    
    % Add observation points
    if ~isempty(obs_pos)
        hold on;
        obs_at_depth = obs_pos(abs(obs_pos(:,3) - mid_depth) < s.Ocean_z_step, :);
        if ~isempty(obs_at_depth)
            plot(obs_at_depth(:,1), obs_at_depth(:,2), 'wo', 'MarkerSize', 8, 'LineWidth', 2);
        end
        hold off;
    end
    
    % 3. Error field (2D slice)
    subplot(2, 3, 3);
    error_slice = est_ssp(:, :, mid_z_idx) - true_ssp(:, :, mid_z_idx);
    imagesc(x_coords, y_coords, error_slice);
    colorbar;
    title(sprintf('Error at %.0fm depth', mid_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    
    % 4. Uncertainty field (2D slice)
    subplot(2, 3, 4);
    if ~isempty(uncertainty)
        imagesc(x_coords, y_coords, uncertainty_grid(:, :, mid_z_idx));
        colorbar;
        title(sprintf('Uncertainty at %.0fm depth', mid_depth));
        xlabel('X (km)'); ylabel('Y (km)');
        axis xy;
    else
        text(0.5, 0.5, 'No uncertainty data', 'HorizontalAlignment', 'center');
        title('Uncertainty (unavailable)');
    end
    
    % 5. Error convergence plot
    subplot(2, 3, 5);
    if iter > 1
        iterations = 1:length(errors);
        rmse_vals = [errors.rmse];
        mae_vals = [errors.mae];
        
        yyaxis left;
        plot(iterations, rmse_vals, 'b-o', 'LineWidth', 2);
        ylabel('RMSE (m/s)', 'Color', 'b');
        
        yyaxis right;
        plot(iterations, mae_vals, 'r-s', 'LineWidth', 2);
        ylabel('MAE (m/s)', 'Color', 'r');
        
        xlabel('Iteration');
        title('Error Convergence');
        grid on;
    else
        text(0.5, 0.5, 'Need more iterations', 'HorizontalAlignment', 'center');
        title('Error Convergence');
    end
    
    % 6. Vertical profile comparison
    subplot(2, 3, 6);
    % Choose center point for profile
    center_x_idx = round(length(x_coords) / 2);
    center_y_idx = round(length(y_coords) / 2);
    
    true_profile = squeeze(true_ssp(center_y_idx, center_x_idx, :));
    est_profile = squeeze(est_ssp(center_y_idx, center_x_idx, :));
    
    plot(true_profile, z_coords, 'b-', 'LineWidth', 2, 'DisplayName', 'True');
    hold on;
    plot(est_profile, z_coords, 'r--', 'LineWidth', 2, 'DisplayName', 'Estimated');
    
    % Add uncertainty bounds if available
    if ~isempty(uncertainty)
        center_uncertainty = squeeze(uncertainty_grid(center_y_idx, center_x_idx, :));
        fill([est_profile - center_uncertainty; flipud(est_profile + center_uncertainty)], ...
             [z_coords'; flipud(z_coords')], 'r', 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    end
    
    hold off;
    set(gca, 'YDir', 'reverse'); % Depth increases downward
    xlabel('Sound Speed (m/s)');
    ylabel('Depth (m)');
    title('Vertical Profile (Center)');
    legend('Location', 'best');
    grid on;
    
    % Add overall title with current statistics
    if ~isempty(errors) && iter <= length(errors)
        current_error = errors(iter);
        sgtitle(sprintf('SSP Estimation - Iter %d: RMSE=%.3f m/s, Obs=%d, Corr=%.3f', ...
                       iter, current_error.rmse, current_error.num_observations, ...
                       current_error.spatial_correlation));
    else
        sgtitle(sprintf('SSP Estimation - Iteration %d', iter));
    end
    
    % Force update
    drawnow;
end