function s = acoustic_presets(preset_name)
    % ACOUSTIC_PRESETS - Predefined configurations for different applications
    %
    % Usage: s = acoustic_presets('shallow_water_comms')
    %
    % Available presets:
    %   'shallow_water_comms'  - Underwater communication systems (shallow water)
    %   'deep_ocean_sonar'     - Deep ocean sonar applications  
    %   'coastal_survey'       - Coastal geological surveys
    %   'mine_warfare'         - Naval mine detection/warfare
    %   'marine_biology'       - Marine habitat characterization
    %   'seabed_mapping'       - High-resolution seabed parameter mapping
    
    if nargin < 1
        preset_name = 'shallow_water_comms';
        fprintf('No preset specified, using default: %s\n', preset_name);
    end
    
    % Start with base settings
    s = get_sim_settings();
    
    switch lower(preset_name)
        case 'shallow_water_comms'
            % Configuration for underwater communication systems
            fprintf('🌊 Loading: Shallow Water Communications preset\n');
            
            % Environment
            s.OceanDepth = 30;           % Typical coastal depth
            s.sim_frequency = 12000;     % 12kHz communication frequency
            s.sim_source_depth = 5;      % Near-surface transmitter
            
            % Bottom parameters (sandy coastal environment)
            s.mu_th = [1650; 1.6];       % Sand/shell bottom
            s.Sigma_th = diag([30, 0.2].^2);  % Moderate uncertainty
            
            % Measurement strategy
            s.N = 12;                    % Sufficient for comm planning
            s.nbv_method = 'rrt_star_nbv';  % Good for AUV deployment
            s.sigma_tl_noise = 0.8;      % Low noise environment
            
            % Coverage area
            s.x_max = 2.0; s.x_min = -2.0;  % 4km × 4km area
            s.y_max = 2.0; s.y_min = -2.0;
            
        case 'deep_ocean_sonar'
            % Configuration for deep ocean sonar applications
            fprintf('🌊 Loading: Deep Ocean Sonar preset\n');
            
            % Environment  
            s.OceanDepth = 4000;         % Deep ocean
            s.sim_frequency = 1000;      % 1kHz sonar frequency
            s.sim_source_depth = 100;    % Submarine depth
            
            % Bottom parameters (abyssal sediments)
            s.mu_th = [1500; 1.2];       % Soft sediment
            s.Sigma_th = diag([50, 0.3].^2);  % High uncertainty
            
            % Measurement strategy
            s.N = 20;                    % More measurements needed
            s.nbv_method = 'multi_objective';  % Balance objectives
            s.sigma_tl_noise = 1.5;      % Noisy deep ocean
            
            % Large coverage area
            s.x_max = 10.0; s.x_min = -10.0;  % 20km × 20km
            s.y_max = 10.0; s.y_min = -10.0;
            s.z_max = 500;               % Deep sensor deployments
            
        case 'coastal_survey'
            % Configuration for coastal geological surveys
            fprintf('🌊 Loading: Coastal Survey preset\n');
            
            % Environment
            s.OceanDepth = 50;           % Continental shelf
            s.sim_frequency = 3500;      % 3.5kHz sub-bottom profiler
            s.sim_source_depth = 2;      % Surface vessel
            
            % Bottom parameters (variable geology)
            s.mu_th = [1600; 1.4];       % Mixed sediment/rock
            s.Sigma_th = diag([100, 0.5].^2);  % High spatial variability
            
            % High-resolution mapping
            s.N = 25;                    % Dense sampling
            s.nbv_method = 'information_gain';  % Maximize information
            s.sigma_tl_noise = 1.0;      % Moderate noise
            
            % Survey grid
            s.x_max = 3.0; s.x_min = -3.0;   % 6km × 6km survey area
            s.y_max = 3.0; s.y_min = -3.0;
            
        case 'mine_warfare'
            % Configuration for naval mine detection
            fprintf('🌊 Loading: Mine Warfare preset\n');
            
            % Environment
            s.OceanDepth = 20;           % Shallow tactical waters
            s.sim_frequency = 50000;     % 50kHz high-frequency sonar
            s.sim_source_depth = 8;      % MCM vessel sonar
            
            % Bottom parameters (sandy/rocky)
            s.mu_th = [1700; 1.8];       % Hard bottom for mine contrast
            s.Sigma_th = diag([25, 0.15].^2);  % Low uncertainty needed
            
            % Precise requirements
            s.N = 15;                    % Focused measurements
            s.nbv_method = 'uncertainty_guided';  % Minimize uncertainty
            s.sigma_tl_noise = 0.5;      % High-quality sonar
            
            % Tactical area
            s.x_max = 1.0; s.x_min = -1.0;   % 2km × 2km patrol area
            s.y_max = 1.0; s.y_min = -1.0;
            
        case 'marine_biology' 
            % Configuration for marine habitat characterization
            fprintf('🌊 Loading: Marine Biology preset\n');
            
            % Environment
            s.OceanDepth = 200;          % Marine sanctuary depth
            s.sim_frequency = 120000;    % 120kHz biological acoustic
            s.sim_source_depth = 10;     % Research vessel
            
            % Bottom parameters (biological focus)
            s.mu_th = [1580; 1.3];       % Organic-rich sediments
            s.Sigma_th = diag([40, 0.25].^2);  % Biological variability
            
            % Ecological sampling
            s.N = 18;                    % Ecosystem coverage
            s.nbv_method = 'multi_objective';  % Balance coverage & accuracy
            s.sigma_tl_noise = 1.2;      % Variable biological scattering
            
            % Habitat area
            s.x_max = 5.0; s.x_min = -5.0;   % 10km × 10km ecosystem
            s.y_max = 5.0; s.y_min = -5.0;
            
        case 'seabed_mapping'
            % Configuration for high-resolution seabed parameter mapping
            fprintf('🌊 Loading: Seabed Mapping preset\n');
            
            % Environment
            s.OceanDepth = 100;          % Mid-depth continental shelf
            s.sim_frequency = 5000;      % 5kHz mapping frequency
            s.sim_source_depth = 3;      % AUV/ROV platform
            
            % Bottom parameters (geological mapping)
            s.mu_th = [1620; 1.5];       % Typical shelf sediments
            s.Sigma_th = diag([60, 0.3].^2);  % Geological uncertainty
            
            % High-resolution mapping
            s.N = 30;                    % Dense measurements
            s.nbv_method = 'rrt_star_nbv';  % Optimal for AUV paths
            s.sigma_tl_noise = 0.8;      % Controlled platform noise
            
            % Detailed mapping area
            s.x_max = 4.0; s.x_min = -4.0;   % 8km × 8km detailed map
            s.y_max = 4.0; s.y_min = -4.0;
            s.z_max = 80;                % Near-bottom measurements
            
        otherwise
            error('Unknown preset: %s\nAvailable: shallow_water_comms, deep_ocean_sonar, coastal_survey, mine_warfare, marine_biology, seabed_mapping', preset_name);
    end
    
    % Set NBV-specific parameters based on method
    switch s.nbv_method
        case 'rrt_star_nbv'
            s.rrt_max_iter = 200;
            s.rrt_step_size = min([s.x_max-s.x_min, s.y_max-s.y_min])/20;
            s.rrt_search_radius = s.rrt_step_size * 3;
            
        case 'multi_objective'
            s.w_info = 0.5;
            s.w_coverage = 0.3;
            s.w_efficiency = 0.2;
            
        case 'information_gain'
            s.n_candidates = min(100, s.N * 8);  % Scale with measurement count
    end
    
    fprintf('Configuration loaded:\n');
    fprintf('  Environment: %.0fm depth, %.0f Hz\n', s.OceanDepth, s.sim_frequency);
    fprintf('  Parameters: [%.0f ± %.0f m/s, %.2f ± %.2f]\n', ...
        s.mu_th(1), sqrt(s.Sigma_th(1,1)), s.mu_th(2), sqrt(s.Sigma_th(2,2)));
    fprintf('  Strategy: %s with %d measurements\n', s.nbv_method, s.N);
    fprintf('  Coverage: %.1f×%.1f km area\n', s.x_max-s.x_min, s.y_max-s.y_min);
end
