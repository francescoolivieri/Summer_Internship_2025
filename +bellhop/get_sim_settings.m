function s = get_sim_settings()
%GET_SETTINGS Returns a structure containing the variables that control the
%simulation

%% Ocean Settings
s.OceanDepth = 50.; % Depth of the ocean in meters
s.OceanFloorType = 'flat'; % types: flat, smooth_waves, gaussian_features, fractal_noise

s.Ocean_x_min = -2;
s.Ocean_x_max = 2;
s.Ocean_y_min = -2;
s.Ocean_y_max = 2;

s.Ocean_step = 0.2;
s.Ocean_z_step = s.OceanDepth * 0.25; % Empirical rule: shd plot scattered with smaller steps


%% Bellhop simulation settings
s.sim_frequency = 1000.;
s.sim_max_depth = 80.0;
s.sim_source_x = -0.2;
s.sim_source_y = 0.0;
s.sim_source_depth = 10.0;
s.sim_range = 1.5;          %% CHECK COMPLIANT WITH SCENARIO
s.sim_num_bearing = 5;    % more accurate -> 361 but heavier simulation

% Extensions
s.sim_use_ssp_file = true;
s.sim_use_bty_file = true;
s.sim_accurate_3d = false; % May increase running time by 2/3 times
s.sim_bty_splitted = false;


% Default Parameters
s.sim_param_sp_water = 1500;
s.sim_param_sp_sediment = 1600;
s.sim_param_density_sediment = 1.5;
s.sim_param_attenuation_sediment = 0.5;

s.verbose = true;


%% External File References
s.env_file_name = "ac_env_model.env";
s.bellhop_file_name = 'ac_env_model';
s.bty_file_name = "ac_env_model.bty";


%% UAV Measurements Settings
s.z_min=0;              
s.z_max=s.sim_max_depth;

% fix some, it's actually rectangle cause bellhop3d generates sphere
s.x_min=-1.5;
s.x_max=1.5;

s.y_min=-1.5;
s.y_max=1.5;


% Sensor management
s.sm=false;                            % Sensor management on/off
s.nbv_method='tree_memoized';       % methods: tree_memoized, rrt_star, bayesian_opt, information_gain, multi_objective


%% Measurements position settings
s.d_z = 20;                            % Step distance depth [m]
s.d_x = 0.5;                           % Step distance range [km]
s.d_y = 0.5;                           % Step distance range [km]

s.z_start = 20;                        % Start depth [m]
s.x_start = 0.500;                     % Start x [km]
s.y_start = 0.500;                     % Start y [km]

s.depth=2;                             % Depth of planing tree, i.e., how many steps ahead should we plan. 


s.N=20;                                % Total number of measurements
s.sigma_tl_noise=1;                    % Variance of the measurement noise [dB]


s.Sigma_rr=1^2;                        % Filter assumed measurement noise variance [dB^2]



%% UAV Settings
s.UAVSampleTime = 0.001;
s.Gravity = 9.81;
s.DroneMass = 0.1;
assignin("base", "Gravity", s.Gravity)
assignin("base", "DroneMass", s.DroneMass)

s.InitialPosition = [0 0 -7];
s.InitialOrientation = [0 0 0];

% Proportional Gains
s.Px = 3.5;
s.Py = 3.5;
s.Pz = 4.0;

% Derivative Gains
s.Dx = 3.0;
s.Dy = 3.0;
s.Dz = 4.0;

% Integral Gains
s.Ix = 0.1;
s.Iy = 0.1;
s.Iz = 0.2;

% Filter Coefficients
s.Nx = 10;
s.Ny = 10;
s.Nz = 14.4947065605712; 

% Lidar Settings
s.AzimuthResolution = 0.5;      
s.ElevationResolution = 2;
s.MaxRange = 7;
s.AzimuthLimits = [-179 179];
s.ElevationLimits = [-15 15];

%

end