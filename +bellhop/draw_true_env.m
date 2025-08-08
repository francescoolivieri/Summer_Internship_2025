function draw_true_env(s, scene)

global extra_output

% % Generate file for the true parameters
% create_env_file(data.th, 'true_parameters.env')

if extra_output
    figure;
    title('SSP plot','FontSize',10);
    if s.sim_use_ssp_file
        plotssp3d(s.bellhop_file_name + ".ssp" )
    else
        plotssp(s.bellhop_file_name + ".env" )
    end
end
 

% Run bellhop to get the true transmission loss
bellhop3d(s.bellhop_file_name);

figure;
title('Transmission loss','FontSize',10);
plotshd(char(s.bellhop_file_name + ".shd"));
hold on;

% title('Transmission Loss','FontSize',12,'FontWeight','bold')
% xlabel('Range (km)','FontSize',10);
% ylabel('Depth (m)','FontSize',10);
% colorbar;
% hold off;

% %% Plot rays
% figure()
% plotray3d(char(s.bellhop_file_name + ".ray"))
% %title('3D Ray Traces','FontSize',12,'FontWeight','bold');
% grid on;

end