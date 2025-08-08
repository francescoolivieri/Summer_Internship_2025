function [data, s, sceneFigure] = setupUnderwaterSimulation(varargin)
    % Setup underwater simulation with configurable parameters
    %
    % Optional Name-Value Parameters:
    %   'Parameters' - Struct with fields:
    %       - names: cell array of parameter names to estimate
    %       - mu: prior mean values
    %       - Sigma: prior covariance matrix
    %   'Units' - String specifying units (default: 'km')
    %   'ExtraOutput' - Boolean for additional output (default: false)
    
    % Parse input parameters
    p = inputParser;
    addParameter(p, 'Parameters', [], @(x) isempty(x) || isstruct(x));
    addParameter(p, 'Units', 'km', @ischar);
    addParameter(p, 'ExtraOutput', false, @islogical);
    parse(p, varargin{:});
    
    % Initialize global variables
    global extra_output
    extra_output = p.Results.ExtraOutput;
    
    global units
    units = p.Results.Units;
    
    % Clean any existing files
    clean_files();
    
    % Load the simulation settings
    s = get_sim_settings();

    % Create chosen scenario for the simulation
    [scene, sceneFigure] = scenarioBuilder(s);
    s.scene = scene;
    
    % Initialize parameter maps using the simplified approach
    [data, s] = initializeParameterMaps(s, p.Results.Parameters);
   
    % Generate a .ssp file if needed
    if s.sim_use_ssp_file
        data.true_params.set("ssp_grid", generateSSP3D(s, scene));
    end
    
    % Generate .env file using true parameter values
    writeENV3D(s.bellhop_file_name + ".env", s, data.true_params.getMap()); 
    fprintf('Wrote true ENV file.\n');
    
    % Generate .bty file using true parameter values
    if s.sim_use_bty_file
        writeBTY3D(s.bellhop_file_name + ".bty", data.true_params.getMap());
        figure
    end
    
    % Run bellhop and draw environment
    draw_true_env(s, scene);

    % Print polar shd
    if s.sim_accurate_3d
        figure;
        plotshdpol(s.bellhop_file_name + ".shd");
    end

    % Initialize filter
    data = init_filter(data, s);
   
end



% Helper function to display comparison
function displayParameterComparison(data)
    % Display comparison between true and estimated parameters
    
    estimation_names = data.true_params.getEstimationParameterNames();
    
    fprintf('\n=== Parameter Comparison (Iteration %d) ===\n', data.iteration_count);
    for i = 1:length(estimation_names)
        name = estimation_names{i};
        true_val = data.true_params.get(name);
        est_val = data.estimated_params.get(name);
        error_val = est_val - true_val;
        
        % Calculate improvement if history exists
        if size(data.estimation_history, 2) > 1
            initial_est = data.estimation_history(i, 1);
            initial_error = abs(initial_est - true_val);
            current_error = abs(error_val);
            improvement = ((initial_error - current_error) / initial_error) * 100;
            
            fprintf('  %-25s: True=%.4f, Est=%.4f, Error=%.4f, Improvement=%.1f%%\n', ...
                    name, true_val, est_val, error_val, improvement);
        else
            fprintf('  %-25s: True=%.4f, Est=%.4f, Error=%.4f\n', ...
                    name, true_val, est_val, error_val);
        end
    end
end

function [data, s] = initializeParameterMaps(s, param_config)
    % Initialize parameter mapping system (updated for simplified approach)
    
    % Define all possible parameters and their default values
    default_params = ParameterMap(s).getMap();
    
    % Create true parameter map
    data.true_params = ParameterMap(default_params);
    
    % Handle custom parameter configuration
    if ~isempty(param_config)

        if strcmp(param_config.names(1), 'ssp_grid')
            
            disp('Estimating ssp_grid. (Find better solution)')
            s.estimation_param_names = param_config.names;
            s.mu_th = 0.;
            s.Sigma_th = [0];

            % Initialize estimated parameter map (copy from default)
            data.estimated_params = ParameterMap(default_params, s.estimation_param_names);

        else

            % Validate parameter structure
            assert(all(isfield(param_config, {'names', 'mu', 'Sigma'})), ...
                'Parameters struct must contain fields: names, mu, Sigma');
            assert(length(param_config.mu) == size(param_config.Sigma, 1), ...
                'Parameter mean and covariance dimensions must match');
            
            % Store estimation configuration
            s.estimation_param_names = param_config.names;
            s.mu_th = param_config.mu;
            s.Sigma_th = param_config.Sigma;
            
            % Set estimation parameter names in the maps
            data.true_params.setEstimationParameterNames(param_config.names);
            
            % Sample true values for parameters being estimated
            true_values = param_config.mu + chol(param_config.Sigma,'lower')*randn(size(param_config.mu));
            data.true_params.update(true_values, param_config.names);
    
            % Initialize estimated parameter map (copy from true params)
            data.estimated_params = ParameterMap(data.true_params.getMap(), s.estimation_param_names);
            
            % Set initial estimates for parameters being estimated to prior means
            data.estimated_params.update(s.mu_th, s.estimation_param_names);
        
            % Store parameter arrays for backward compatibility
            data.th = data.true_params.asArray(s.estimation_param_names);
            data.th_est = data.estimated_params.asArray(s.estimation_param_names);
        end

        
    else
        disp("Simple run: no parameter to estimate.");
        % % Use default estimation parameters from settings
        % s.estimation_param_names = s.parameter_names;
        % data.true_params.setEstimationParameterNames(s.parameter_names);
        % 
        % % Sample true values using default priors
        % true_values = s.mu_th + chol(s.Sigma_th,'lower')*randn(size(s.mu_th));
        % data.true_params.update(true_values, s.estimation_param_names);
    end
    
    

end