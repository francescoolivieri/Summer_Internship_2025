function [Cvol] = generateSSP3D(s, scene)

%GENERATESSP Generate SSP file 

% Plot measured sound speed profile
% plot_sound_speed_profile('CTD');

% --------------------- Build a 3-D grid ---------------------------
x = s.Ocean_x_min:s.Ocean_step:s.Ocean_x_max;           % km east‑west
y = s.Ocean_y_min:s.Ocean_step:s.Ocean_y_max;           % km north‑south
z = 0:s.Ocean_z_step:s.sim_max_depth; % max_depth = max(scene.floor, [], "all");  


% x(1) = s.Ocean_x_min - 1; 
% y(1) = s.Ocean_y_min - 1; 
% x(end) = s.Ocean_x_max + 1; 
% y(end) = s.Ocean_y_max + 1; 
% %display(size(z));

[X,Y,Z] = meshgrid(x,y,z);    % Cartesian grid (size Ny×Nx×Nz)


% --------------------- Generate sound‑speed volume -----------------------
Cvol = gen_sound_speed_gp(X,Y,Z);   % use default hyper‑parameters

writeSSP3D(s.bellhop_file_name, x, y, z, Cvol);

end

% % --------------------- Extend ssp data to below the ocean floor -----------------------
% %% Is this part useful? Or the bty options cover it?
% 
% ground_ssp = 1450; %max(Cvol(:, :, end), [], 'all');
% 
% % "z" goes until the MAX depth of the real bty, we now add the ssp below that floor
% z_filled = [z(1:end), max_depth+0.1:1:s.sim_max_depth+5];
% for i = (length(z)+1):length(z_filled)
%     Cvol(:,:,i) = ground_ssp;  % Fill all new layers with the ground speed
% end
% z = z_filled;
% 
% % Put to ground speed all the points below bty
% z_depths_3D = reshape(z(1:size(Cvol,3)), 1, 1, []);
% logical_mask = z_depths_3D > scene.floor;
% Cvol(logical_mask) = ground_ssp;
% 
% 
% % plot one sound speed profile 
% sound_speed = squeeze(Cvol(1,1, :))';
% depth = z;
% figure;
% title('SSP profile at one point','FontSize',10);
% plot(sound_speed, depth, 'b-','LineWidth',1.5);
% set(gca, 'YDir','reverse'); % Depth increases downward
% xlabel('Sound Speed (m/s)');
% ylabel('Depth (m)');
% title('Sound Speed Profile');
% grid on;
% 
% 
% 
% % --------------------- Write File  ------------------------
% writeSSP3D(s.bellhop_file_name, x, y, z, Cvol);
% 
% % --------------------- Visualise  ------------------------
% % figure;
% % % Get global min/max for consistent color scale
% % vmin = min(Cvol(:));
% % vmax = max(Cvol(:));
% % 
% % 
% % for i = 1:length(x)
% %     imagesc(y, z, squeeze(Cvol(i,:,:))');
% % 
% %     set(gca, 'YDir','reverse');
% %     caxis([vmin vmax]);  % fixed color range
% %     xlabel('y (m)');
% %     ylabel('depth (m)');
% %     title(['Vertical slice at x = ', num2str(x(i)), ' m']);
% %     colorbar;
% %     pause(0.5);
% % end
% 
% end