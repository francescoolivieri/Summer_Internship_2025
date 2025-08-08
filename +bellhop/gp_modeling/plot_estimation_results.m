function plot_estimation_results(errors, ssp_gp, true_ssp_grid, s)
    % PLOT_ESTIMATION_RESULTS - Create comprehensive final results visualization
    %
    % Inputs:
    %   errors: array of error metrics from all iterations
    %   ssp_gp: final SSPGaussianProcess object
    %   true_ssp_grid: true SSP field [Ny x Nx x Nz]
    %   s: simulation settings
    
    % Create new figure for final results
    fig = figure('Name', 'SSP Estimation Final Results', 'Position', [50 50 1400 1000]);
    
    % Get final estimates and observations
    est_ssp_grid = ssp_gp.getCurrentSSPGrid();
    uncertainty = ssp_gp.getUncertainty();
    [obs_pos, obs_vals] = ssp_gp.getObservations();
    
    % Setup grid coordinates
    x_coords = s.Ocean_x_min:s.Ocean_step:s.Ocean_x_max;
    y_coords = s.Ocean_y_min:s.Ocean_step:s.Ocean_y_max;
    z_coords = 0:s.Ocean_z_step:s.sim_max_depth;
    
    % Choose depths for visualization
    shallow_idx = max(1, round(length(z_coords) * 0.2));  % 20% depth
    deep_idx = round(length(z_coords) * 0.8);             % 80% depth
    
    shallow_depth = z_coords(shallow_idx);
    deep_depth = z_coords(deep_idx);
    
    % Reshape uncertainty
    if ~isempty(uncertainty)
        Ny = length(y_coords);
        Nx = length(x_coords);
        Nz = length(z_coords);
        uncertainty_grid = reshape(uncertainty, [Ny, Nx, Nz]);
    end
    
    % ======= ROW 1: Shallow depth comparison =======
    
    % True SSP at shallow depth
    subplot(3, 4, 1);
    imagesc(x_coords, y_coords, true_ssp_grid(:, :, shallow_idx));
    colorbar;
    title(sprintf('True SSP at %.0fm', shallow_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    addObservationPoints(obs_pos, shallow_depth, s.Ocean_z_step);
    
    % Estimated SSP at shallow depth
    subplot(3, 4, 2);
    imagesc(x_coords, y_coords, est_ssp_grid(:, :, shallow_idx));
    colorbar;
    title(sprintf('Estimated SSP at %.0fm', shallow_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    addObservationPoints(obs_pos, shallow_depth, s.Ocean_z_step);
    
    % Error at shallow depth
    subplot(3, 4, 3);
    error_shallow = est_ssp_grid(:, :, shallow_idx) - true_ssp_grid(:, :, shallow_idx);
    imagesc(x_coords, y_coords, error_shallow);
    colorbar;
    title(sprintf('Error at %.0fm', shallow_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    
    % Uncertainty at shallow depth
    subplot(3, 4, 4);
    if ~isempty(uncertainty)
        imagesc(x_coords, y_coords, uncertainty_grid(:, :, shallow_idx));
        colorbar;
        title(sprintf('Uncertainty at %.0fm', shallow_depth));
        xlabel('X (km)'); ylabel('Y (km)');
        axis xy;
    else
        text(0.5, 0.5, 'Uncertainty not available', 'HorizontalAlignment', 'center');
        title('Uncertainty');
    end
    
    % ======= ROW 2: Deep depth comparison =======
    
    % True SSP at deep depth
    subplot(3, 4, 5);
    imagesc(x_coords, y_coords, true_ssp_grid(:, :, deep_idx));
    colorbar;
    title(sprintf('True SSP at %.0fm', deep_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    addObservationPoints(obs_pos, deep_depth, s.Ocean_z_step);
    
    % Estimated SSP at deep depth
    subplot(3, 4, 6);
    imagesc(x_coords, y_coords, est_ssp_grid(:, :, deep_idx));
    colorbar;
    title(sprintf('Estimated SSP at %.0fm', deep_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    addObservationPoints(obs_pos, deep_depth, s.Ocean_z_step);
    
    % Error at deep depth
    subplot(3, 4, 7);
    error_deep = est_ssp_grid(:, :, deep_idx) - true_ssp_grid(:, :, deep_idx);
    imagesc(x_coords, y_coords, error_deep);
    colorbar;
    title(sprintf('Error at %.0fm', deep_depth));
    xlabel('X (km)'); ylabel('Y (km)');
    axis xy;
    
    % Uncertainty at deep depth
    subplot(3, 4, 8);
    if ~isempty(uncertainty)
        imagesc(x_coords, y_coords, uncertainty_grid(:, :, deep_idx));
        colorbar;
        title(sprintf('Uncertainty at %.0fm', deep_depth));
        xlabel('X (km)'); ylabel('Y (km)');
        axis xy;
    else
        text(0.5, 0.5, 'Uncertainty not available', 'HorizontalAlignment', 'center');
        title('Uncertainty');
    end
    
    % ======= ROW 3: Analysis plots =======
    
    % Error convergence
    subplot(3, 4, 9);
    iterations = 1:length(errors);
    rmse_vals = [errors.rmse];
    mae_vals = [errors.mae];
    max_error_vals = [errors.max_error];
    
    yyaxis left;
    plot(iterations, rmse_vals, 'b-o', 'LineWidth', 2, 'DisplayName', 'RMSE');
    hold on;
    plot(iterations, mae_vals, 'b--s', 'LineWidth', 2, 'DisplayName', 'MAE');
    ylabel('Error (m/s)', 'Color', 'b');
    
    yyaxis right;
    plot(iterations, max_error_vals, 'r-^', 'LineWidth', 2, 'DisplayName', 'Max Error');
    ylabel('Max Error (m/s)', 'Color', 'r');
    
    xlabel('Iteration');
    title('Error Convergence');
    legend('Location', 'best');
    grid on;
    
    % Error vs depth profile
    subplot(3, 4, 10);
    rmse_by_depth = [errors(end).rmse_by_depth];
    mae_by_depth = [errors(end).mae_by_depth];
    
    plot(rmse_by_depth, z_coords, 'b-', 'LineWidth', 2, 'DisplayName', 'RMSE');
    hold on;
    plot(mae_by_depth, z_coords, 'r--', 'LineWidth', 2, 'DisplayName', 'MAE');
    set(gca, 'YDir', 'reverse');
    xlabel('Error (m/s)');
    ylabel('Depth (m)');
    title('Error vs Depth');
    legend('Location', 'best');
    grid on;
    
    % Observation locations in 3D
    subplot(3, 4, 11);
    if ~isempty(obs_pos)
        scatter3(obs_pos(:,1), obs_pos(:,2), obs_pos(:,3), 50, obs_vals, 'filled');
        colorbar;
        xlabel('X (km)'); ylabel('Y (km)'); zlabel('Depth (m)');
        title('Observation Locations');
        set(gca, 'ZDir', 'reverse');
        grid on;
    else
        text(0.5, 0.5, 'No observations', 'HorizontalAlignment', 'center');
        title('Observation Locations');
    end
    
    % Final statistics and metrics
    subplot(3, 4, 12);
    axis off;
    
    % Compute final statistics
    final_error = errors(end);
    
    stats_text = {
        'Final Estimation Statistics:';
        '';
        sprintf('Total Observations: %d', final_error.num_observations);
        sprintf('Final RMSE: %.4f m/s', final_error.rmse);
        sprintf('Final MAE: %.4f m/s', final_error.mae);
        sprintf('Max Error: %.4f m/s', final_error.max_error);
        sprintf('Mean Rel. Error: %.2f%%', final_error.mean_relative_error * 100);
        sprintf('Spatial Correlation: %.4f', final_error.spatial_correlation);
        '';
    };
    
    % Conditionally add uncertainty statistics
    if ~isnan(final_error.mean_uncertainty)
        stats_text(end+1:end+3) = {
            sprintf('Mean Uncertainty: %.4f m/s', final_error.mean_uncertainty);
            sprintf('Coverage (1σ): %.1f%%', final_error.coverage_1sigma * 100);
            sprintf('Coverage (2σ): %.1f%%', final_error.coverage_2sigma * 100);
        };
    else
        stats_text{end+1} = 'Uncertainty: Not Available';
    end
    
    text(0.05, 0.95, stats_text, 'Units', 'normalized', 'VerticalAlignment', 'top', ...
         'FontSize', 10, 'FontName', 'monospace');
    
    % Add overall title
    sgtitle(sprintf('SSP Estimation Results - %d Observations, Final RMSE: %.4f m/s', ...
                   final_error.num_observations, final_error.rmse), 'FontSize', 14);
    
    % Save results
    try
        saveas(fig, 'ssp_estimation_results.png');
        fprintf('Saved results figure to: ssp_estimation_results.png\n');
    catch
        warning('Could not save results figure');
    end
end

function addObservationPoints(obs_pos, target_depth, depth_tolerance)
    % Helper function to add observation points to current plot
    if ~isempty(obs_pos)
        hold on;
        obs_at_depth = obs_pos(abs(obs_pos(:,3) - target_depth) <= depth_tolerance, :);
        if ~isempty(obs_at_depth)
            plot(obs_at_depth(:,1), obs_at_depth(:,2), 'wo', 'MarkerSize', 8, ...
                 'LineWidth', 2, 'MarkerFaceColor', 'black');
        end
        hold off;
    end
end