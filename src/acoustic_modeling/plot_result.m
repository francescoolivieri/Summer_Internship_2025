function plot_result(data, s)

idx=find(isfinite(data.x),1,'last');


% Plot parameter estimates over time
for ii=1:numel(s.mu_th)

    h=zeros(1,3);
    figure(ii+2)
    clf
        
    h(1)=plot(data.th_est(ii,:),'r');
    hold on;
    sigma=sqrt(squeeze(data.Sigma_est(ii,ii,:)))';
    h(2)=plot(data.th_est(ii,:)+3*sigma,'b');
    plot(data.th_est(ii,:)-3*sigma,'b')
    h(3)=plot(data.th(ii)*ones(size(data.th_est(ii,:))),'k');
    legend(h,{'Mean','3*Std','True'})
    set(gcf, 'Color', 'w');         % White figure background
    ax = gca;
    ax.Color = 'w';                 % White axes background
    ax.XColor = 'k';                % Black axis lines
    ax.YColor = 'k';
    ax.ZColor = 'k';                % Optional (for 3D plots)
    if ii==1
        ylabel('Speed (m/s)')
        title('alphaR')
    else
        ylabel('rho (g/cm^3)')
        title('rho')
    end
    xlabel('Time step')
    grid minor;
end

% % Create 3D grid for uncertainty visualization
% x_values = s.x_min:0.3:s.x_max;
% y_values = s.y_min:0.3:s.y_max;
% z_values = s.z_min+5:5:s.OceanDepth-5;

% % OPTION 1: Compute only 2D slice (+ EFFICIENT)
% % Create a 2D slice at a specific y-value
% y_slice = mean([s.y_min, s.y_max]); % Middle y-value
% 
% % Generate 2D grid at the y_slice
% [X_slice, Z_slice] = meshgrid(x_values, z_values);
% pos_slice = [X_slice(:), repmat(y_slice, numel(X_slice), 1), Z_slice(:)];
% 
% % Calculate prediction uncertainty for 2D slice only
% Np = 10;
% Y_slice = zeros(size(pos_slice, 1), Np);
% 
% parfor pp = 1:Np
%     % Sample parameter values from the posterior distribution
%     th_sample = data.th_est(:, idx) + chol(squeeze(data.Sigma_est(:, :, idx)), 'lower') * randn(size(data.th_est(:, idx)));
% 
%     % Create parameter map for this sample using utility function
%     param_map_sample = createParameterMapFromArray(th_sample, s, data.estimated_params);
% 
%     % Call forward model with parameter map
%     Y_slice(:, pp) = forward_model(param_map_sample, pos_slice, s);
% end
% 
% var_tl_slice = var(Y_slice, [], 2);
% 
% % Reshape for 2D plotting
% var_tl_2d = reshape(var_tl_slice, size(X_slice));
% 
% % Create the pcolor plot (uncertainty plot, 2D slice)
% figure(5);
% pcolor(X_slice, Z_slice, sqrt(var_tl_2d));
% shading interp;
% cb = colorbar;
% set(gca, 'YDir', 'Reverse')
% ylabel('Depth (m)')
% xlabel('X coordinate (m)')
% title(sprintf('Standard deviation of predicted loss (Y-slice at %.1f m)', y_slice));
% colormap jet;
% caxis([0 10])
% cb.Label.String = 'Std (dB)';


% % OPTION 2: Full 3D computation and extract slices
% Uncomment this section and comment out Option 1 above
% Full 3D grid (warning: computationally expensive)
% [X, Y, Z] = meshgrid(x_values, y_values, z_values);
% pos_all = [X(:), Y(:), Z(:)];
% 
% % Check positions are within the boundaries
% mask = sqrt(pos_all(:,1).^2 + pos_all(:,2).^2) <= s.x_max;
% pos_valid = pos_all(mask, :);
% 
% % Calculate prediction uncertainty for valid positions only
% Np = 10;
% Yp = zeros(size(pos_valid, 1), Np);
% 
% parfor pp = 1:Np
%     % Sample parameter values from the posterior distribution
%     th_sample = data.th_est(:, idx) + chol(squeeze(data.Sigma_est(:, :, idx)), 'lower') * randn(size(data.th_est(:, idx)));
%     
%     % Create parameter map for this sample using utility function
%     param_map_sample = createParameterMapFromArray(th_sample, s, data.estimated_params);
%     
%     % Call forward model with parameter map
%     Yp(:, pp) = forward_model(param_map_sample, pos_valid, s);
% end
% var_tl_valid = var(Yp, [], 2);
% 
% % Create full-size array with NaN for invalid positions
% var_tl_full = NaN(size(pos_all, 1), 1);
% var_tl_full(mask) = var_tl_valid;  % Place valid values at correct positions
% 
% % Reshape var_tl into 3D grid
% var_tl_3d = reshape(var_tl_full, size(X));
% 
% % Extract 2D slice from 3D results for visualization
% y_slice = mean([s.y_min, s.y_max]);
% [~, y_idx] = min(abs(y_values - y_slice)); % Find closest y index
% 
% % Extract slice at the chosen y-index
% X_slice_from_3d = squeeze(X(:, y_idx, :));
% Z_slice_from_3d = squeeze(Z(:, y_idx, :));
% var_tl_slice_from_3d = squeeze(var_tl_3d(:, y_idx, :));
% 
% % Plot the extracted slice
% figure(5);
% pcolor(X_slice_from_3d, Z_slice_from_3d, sqrt(var_tl_slice_from_3d));
% shading interp;
% cb = colorbar;
% set(gca, 'YDir', 'Reverse')
% ylabel('Depth (m)')
% xlabel('X coordinate (m)')
% title(sprintf('Standard deviation of predicted loss (Y-slice at %.1f m)', y_slice));
% colormap jet;
% caxis([0 10])
% cb.Label.String = 'Std (dB)';
% 
% % Additional 3D visualization options
% % figure(6);
% 
% % % Plot 3D isosurface of uncertainty
% % isosurface(X, Y, Z, sqrt(var_tl_3d), 2); % 2 dB uncertainty level
% % xlabel('X coordinate (m)')
% % ylabel('Y coordinate (m)')
% % zlabel('Depth (m)')
% % title('3D Uncertainty Isosurface (2 dB level)')
% % set(gca, 'ZDir', 'Reverse')
% % view(45, 30)
% % grid on
% 
% % Plot multiple Y-slices
% figure(7);
% num_slices = 5;
% y_slice_indices = round(linspace(1, length(y_values), num_slices));
% for i = 1:num_slices
%     subplot(1, num_slices, i);
%     slice_data = squeeze(sqrt(var_tl_3d(:, y_slice_indices(i), :)));
%     X_plot = squeeze(X(:, y_slice_indices(i), :));
%     Z_plot = squeeze(Z(:, y_slice_indices(i), :));
%     pcolor(X_plot, Z_plot, slice_data);
%     shading interp;
%     set(gca, 'YDir', 'Reverse')
%     title(sprintf('Y = %.1f m', y_values(y_slice_indices(i))));
%     xlabel('X (m)');
%     if i == 1
%         ylabel('Depth (m)');
%     end
%     colorbar;
%     caxis([0 10]);
% end


% Plot measurement trajectory in 3D
figure(100)
clf
plot3(data.x(1:idx), data.y(1:idx), data.z(1:idx), 'b-', 'LineWidth', 2);
hold on
scatter3(data.x(1:idx), data.y(1:idx), data.z(1:idx), 50, 'b', 'filled');
% Highlight current position
scatter3(data.x(idx), data.y(idx), data.z(idx), 100, 'r', 'filled', 'MarkerEdgeColor', 'k');
% Highlight source position
scatter3(s.sim_source_x, s.sim_source_y, s.sim_source_depth, 100, 'green', 'filled', 'MarkerEdgeColor', 'k');


% Set axis properties
axis([s.x_min s.x_max s.y_min s.y_max s.z_min s.z_max])
set(gca, 'ZDir', 'Reverse') % Assuming z is depth (positive downward)
xlabel('X coordinate (km)')
ylabel('Y coordinate (km)')
zlabel('Depth (m)')
grid on
title('3D Measurement Trajectory')
view(45, 30) % Set 3D viewing angle



pause(0.1)

end