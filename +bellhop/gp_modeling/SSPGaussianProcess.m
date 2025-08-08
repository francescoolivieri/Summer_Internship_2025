classdef SSPGaussianProcess < handle
    % SSPGaussianProcess - Gaussian Process for 3D Sound Speed Profile estimation
    %
    % This class manages GP-based estimation of sound speed profiles in 3D
    % underwater environments, with integration to Bellhop acoustic modeling.
    
    properties (Access = private)
        % GP hyperparameters
        ell_h           % Horizontal correlation length (m)
        ell_v           % Vertical correlation length (m)
        sigma_f         % Signal standard deviation (m/s)
        noise_std       % Measurement noise std (m/s)
        
        % Data storage
        X_obs           % Observed positions [N x 3] (x, y, z)
        y_obs           % Observed measurements [N x 1] (derived from TL)
        tl_obs          % Original TL measurements [N x 1] for reference
        
        % Grid for prediction
        grid_x          % x coordinates (km)
        grid_y          % y coordinates (km) 
        grid_z          % z coordinates (m)
        X_grid          % Grid positions [M x 3]
        
        % Mean function (from CTD data)
        mean_func       % Function handle for prior mean
        
        % SSP file settings
        filename        % Output SSP filename
        
        % Cached computations
        K_inv           % Inverse of observation covariance matrix
        alpha           % K_inv * (y_obs - mean(X_obs))
        
        % Inversion parameters
        inversion_method    % 'gradient', 'bayesian', or 'hybrid'
        max_iterations      % Maximum iterations for inversion
        convergence_tol     % Convergence tolerance
        perturbation_size   % Size of perturbations for gradient estimation
    end
    
    methods
        function obj = SSPGaussianProcess(config)
            % Constructor
            % Input: config struct with fields:
            %   - ell_h: horizontal correlation length
            %   - ell_v: vertical correlation length  
            %   - sigma_f: signal standard deviation
            %   - noise_std: measurement noise standard deviation
            %   - filename: output SSP filename
            
            obj.ell_h = config.ell_h;
            obj.ell_v = config.ell_v;
            obj.sigma_f = config.sigma_f;
            obj.noise_std = config.noise_std;
            obj.filename = config.filename;
            
            % Initialize empty observation arrays
            obj.X_obs = [];
            obj.y_obs = [];
            obj.tl_obs = [];
            
            % Set inversion parameters with defaults
            obj.inversion_method = 'hybrid';  % Default method
            obj.max_iterations = 20;
            obj.convergence_tol = 1e-3;
            obj.perturbation_size = 0.1;  % m/s perturbation for gradients
            
            % Load CTD data for mean function
            obj.setupMeanFunction();
            
            % Setup prediction grid based on simulation settings
            obj.setupPredictionGrid();
        end
        
        function setupMeanFunction(obj)
            % Setup mean function from CTD data
            try
                S = load('data/CTD.mat');
                fn = fieldnames(S);
                raw = S.(fn{1});
                z_raw = raw(:,1);
                c_raw = raw(:,2);
                
                % Remove duplicate depths and average
                [z_tr, ~, grp] = unique(z_raw, 'stable');
                c_tr = accumarray(grp, c_raw, [], @mean);
                
                % Create interpolation function
                obj.mean_func = @(z) interp1(z_tr, c_tr, z, 'linear', 'extrap');
                
            catch
                warning('Could not load CTD.mat, using constant mean function');
                obj.mean_func = @(z) 1500 * ones(size(z)); % Default ocean sound speed
            end
        end
        
        function setupPredictionGrid(obj)
            % Setup 3D prediction grid based on simulation settings
            s = get_sim_settings(); % Load simulation settings
            
            % Create coordinate vectors
            obj.grid_x = s.Ocean_x_min:s.Ocean_step:s.Ocean_x_max;  % km
            obj.grid_y = s.Ocean_y_min:s.Ocean_step:s.Ocean_y_max;  % km  
            obj.grid_z = 0:s.Ocean_z_step:s.sim_max_depth;          % m
            
            % Create full 3D meshgrid
            [X, Y, Z] = meshgrid(obj.grid_x, obj.grid_y, obj.grid_z);
            obj.X_grid = [X(:), Y(:), Z(:)]; % [M x 3] matrix
        end
        
        function update(obj, pos, measurement, parameter_map, varargin)
            % Update GP with new observation using sophisticated inversion
            % Inputs:
            %   pos: [1 x 3] position vector [x_km, y_km, z_m]
            %   measurement: scalar transmission loss measurement [dB]
            %   'parameter_map': ParameterMap class for ray-tracing analysis
            %   varargin: optional parameters
            %     - 'method': inversion method ('gradient', 'bayesian', 'hybrid')
            
            % Parse optional inputs
            p = inputParser;
            addParameter(p, 'method', obj.inversion_method, @ischar);
            parse(p, varargin{:});
            
            method = p.Results.method;
            
            % Perform sophisticated inversion
            [c_estimate, confidence] = obj.acousticInversion(measurement, pos, parameter_map, ...
                'method', method);
            
            % Add to observation data
            obj.X_obs = [obj.X_obs; pos];
            obj.y_obs = [obj.y_obs; c_estimate];
            obj.tl_obs = [obj.tl_obs; measurement];
            
            % Update cached computations
            obj.updateCachedValues();
            
            % Optional: print inversion diagnostics
            if length(obj.X_obs) <= 10 || mod(length(obj.X_obs), 10) == 0
                fprintf('  Inversion at [%.1f,%.1f,%.1f]: TL=%.1fdB -> c=%.1fm/s (conf=%.3f)\n', ...
                    pos, measurement, c_estimate, confidence);
            end

            parameter_map.set('ssp_grid', obj.getCurrentSSPGrid());
        end
        
        function [c_est, confidence] = acousticInversion(obj, tl_obs, pos, parameter_map, varargin)
            % Sophisticated acoustic inversion to estimate sound speed from TL
            % Inputs:
            %   tl_obs: observed transmission loss [dB]
            %   pos: receiver position [1x3] [x_km, y_km, z_m]
            %   varargin: optional parameters
            % Outputs:
            %   c_est: estimated sound speed [m/s]
            %   confidence: confidence measure [0-1]
            
            % Parse inputs
            p = inputParser;
            addParameter(p, 'method', obj.inversion_method, @ischar);
            parse(p, varargin{:});
            
            method = p.Results.method;
            
            switch lower(method)
                case 'gradient'
                    [c_est, confidence] = obj.gradientBasedInversion(tl_obs, pos, parameter_map);
                case 'bayesian'
                    [c_est, confidence] = obj.bayesianInversion(tl_obs, pos);
                case 'hybrid'
                    [c_est, confidence] = obj.hybridInversion(tl_obs, pos);
                otherwise
                    error('Unknown inversion method: %s', method);
            end
        end
        
        function [c_est, confidence] = gradientBasedInversion(obj, tl_obs, pos, parameter_map)
            % Gradient-based inversion using local linearization
            
            % Get prior estimate
            c_prior = obj.mean_func(pos(3));
            
            % Compute acoustic sensitivity (Jacobian)
            J = obj.computeAcousticJacobian(pos, c_prior, parameter_map);
            
            if abs(J) < 1e-10
                % Low sensitivity - fall back to prior
                c_est = c_prior;
                confidence = 0.1;
                return;
            end
            
            % Get expected TL at prior
            try
                % Create temporary SSP with prior estimate
                temp_ssp = obj.createLocalSSP(pos, c_prior);
                tl_prior = obj.forwardModelLocal(temp_ssp, pos, parameter_map);
                
                % Gradient-based update
                residual = tl_obs - tl_prior;
                step_size = 0.5;  % Adaptive step size
                delta_c = -step_size * residual / J;
                
                % Apply update with bounds
                c_est = c_prior + delta_c;
                c_est = max(1400, min(1600, c_est));
                
                % Estimate confidence based on gradient magnitude and residual
                confidence = min(1.0, abs(J) * exp(-abs(residual)/10));
                
            catch ME
                warning('Forward model failed in inversion: %s', ME.message);
                c_est = c_prior;
                confidence = 0.1;
            end
        end
        
        function [c_est, confidence] = bayesianInversion(obj, tl_obs, pos, source_pos, frequency)
            % Bayesian inversion with uncertainty quantification
            
            % Prior parameters
            c_prior = obj.mean_func(pos(3));
            sigma_prior = obj.sigma_f;  % Prior uncertainty
            
            % Likelihood parameters (measurement noise model)
            sigma_likelihood = 2.0;  % TL measurement uncertainty [dB]
            
            % Sample around prior for Bayesian update
            n_samples = 15;
            c_samples = linspace(c_prior - 2*sigma_prior, c_prior + 2*sigma_prior, n_samples);
            c_samples = max(1400, min(1600, c_samples));
            
            log_likelihood = zeros(size(c_samples));
            
            for i = 1:length(c_samples)
                try
                    temp_ssp = obj.createLocalSSP(pos, c_samples(i));
                    tl_pred = obj.forwardModelLocal(temp_ssp, pos);
                    
                    % Gaussian likelihood
                    log_likelihood(i) = -0.5 * ((tl_obs - tl_pred) / sigma_likelihood)^2;
                catch
                    log_likelihood(i) = -inf;
                end
            end
            
            % Convert to probabilities
            max_ll = max(log_likelihood(isfinite(log_likelihood)));
            if isempty(max_ll)
                c_est = c_prior;
                confidence = 0.1;
                return;
            end
            
            log_posterior = log_likelihood - max_ll;
            posterior = exp(log_posterior);
            posterior = posterior / sum(posterior);
            
            % Compute posterior mean and confidence
            c_est = sum(c_samples .* posterior);
            posterior_var = sum((c_samples - c_est).^2 .* posterior);
            confidence = exp(-posterior_var / (2 * sigma_prior^2));
        end
        
        function [c_est, confidence] = hybridInversion(obj, tl_obs, pos, source_pos, frequency)
            % Hybrid approach: Bayesian for exploration, gradient for refinement
            
            % First pass: Bayesian inversion for robust estimate
            [c_bayes, conf_bayes] = obj.bayesianInversion(tl_obs, pos, source_pos, frequency);
            
            % Second pass: Gradient refinement if confidence is reasonable
            if conf_bayes > 0.3
                % Use Bayesian estimate as starting point for gradient method
                c_prior_orig = obj.mean_func(pos(3));
                obj.mean_func = @(z) c_bayes * ones(size(z));  % Temporarily update prior
                
                [c_grad, conf_grad] = obj.gradientBasedInversion(tl_obs, pos, source_pos);
                
                % Restore original prior
                obj.mean_func = @(z) c_prior_orig * ones(size(z));
                
                % Weighted combination
                w_bayes = 0.3;
                w_grad = 0.7;
                c_est = w_bayes * c_bayes + w_grad * c_grad;
                confidence = w_bayes * conf_bayes + w_grad * conf_grad;
            else
                c_est = c_bayes;
                confidence = conf_bayes;
            end
        end
        
        function updateCachedValues(obj)
            % Update cached covariance matrix inverse and alpha values
            if isempty(obj.X_obs)
                return;
            end
            
            % Compute observation covariance matrix
            mu_obs = obj.mean_func(obj.X_obs(:,3));
            K_obs = obj.kernelMatrix(obj.X_obs, obj.X_obs);
            K_stable = K_obs + (obj.noise_std^2 + 1e-6) * eye(size(obj.X_obs, 1));
            
            obj.alpha = K_stable \ (obj.y_obs - mu_obs);
            obj.K_inv = inv(K_stable); % Only compute if needed for variance
        end
        
        function K = kernelMatrix(obj, X1, X2)
            % Compute kernel matrix between two sets of points
            % Inputs:
            %   X1: [N1 x 3] positions
            %   X2: [N2 x 3] positions
            % Output:
            %   K: [N1 x N2] kernel matrix
            
            N1 = size(X1, 1);
            N2 = size(X2, 1);
            K = zeros(N1, N2);
            
            for i = 1:N1
                % Compute squared distances
                dx = X1(i,1) - X2(:,1); % x difference (km)
                dy = X1(i,2) - X2(:,2); % y difference (km)  
                dz = X1(i,3) - X2(:,3); % z difference (m)
                
                % Convert x,y to meters for consistent units
                dx_m = dx * 1000;
                dy_m = dy * 1000;
                
                % Anisotropic squared exponential kernel
                r2_h = dx_m.^2 + dy_m.^2;
                r2_v = dz.^2;
                
                K(i,:) = obj.sigma_f^2 * exp(-0.5 * r2_h / obj.ell_h^2 - 0.5 * r2_v / obj.ell_v^2);
            end
        end
        
        function [mu_pred, var_pred] = predict(obj, X_test)
            % Predict mean and variance at test points
            % Input:
            %   X_test: [N_test x 3] test positions
            % Outputs:
            %   mu_pred: [N_test x 1] predicted means
            %   var_pred: [N_test x 1] predicted variances
            
            if isempty(obj.X_obs)
                % No observations yet, return prior
                mu_pred = obj.mean_func(X_test(:,3));
                var_pred = obj.sigma_f^2 * ones(size(mu_pred));
                return;
            end
            
            % Cross-covariance between test and observation points
            K_cross = obj.kernelMatrix(X_test, obj.X_obs);
            
            % Prior mean at test points
            mu_prior = obj.mean_func(X_test(:,3));
            
            % Posterior mean
            mu_pred = mu_prior + K_cross * obj.alpha;
            
            % Posterior variance (if requested)
            if nargout > 1
                K_test = obj.kernelMatrix(X_test, X_test);
                var_pred = diag(K_test) - diag(K_cross * obj.K_inv * K_cross');
                var_pred = max(var_pred, 1e-8); % Ensure non-negative
            end
        end
        
        function ssp_grid = getCurrentSSPGrid(obj)
            % Get current SSP estimate on the full 3D grid
            [mu_pred, ~] = obj.predict(obj.X_grid);
            
            % Reshape to grid format [Ny x Nx x Nz]
            Ny = length(obj.grid_y);
            Nx = length(obj.grid_x);
            Nz = length(obj.grid_z);
            
            ssp_grid = reshape(mu_pred, [Ny, Nx, Nz]); %% [Ny, Nx, Nz]
        end
        
        function writeSSPFile(obj)
            % Write current SSP estimate to file for Bellhop
            ssp_grid = obj.getCurrentSSPGrid();
            writeSSP3D(obj.filename, obj.grid_x, obj.grid_y, obj.grid_z, ssp_grid);
        end
        
        function uncertainty = getUncertainty(obj, positions)
            % Get prediction uncertainty at specified positions
            % Input:
            %   positions: [N x 3] query positions (optional, default: grid)
            
            if nargin < 2
                positions = obj.X_grid;
            end
            
            [~, var_pred] = obj.predict(positions);
            uncertainty = sqrt(var_pred); % Standard deviation
        end
        
        function n_obs = getNumObservations(obj)
            % Get number of observations
            n_obs = size(obj.X_obs, 1);
        end
        
        function [pos, measurements] = getObservations(obj)
            % Get all observations
            pos = obj.X_obs;
            measurements = obj.y_obs;
        end
        
        % =============== SUPPORTING METHODS FOR INVERSION ===============
        
        function J = computeAcousticJacobian(obj, pos, c_nominal, parameter_map)
            % Compute acoustic Jacobian (sensitivity of TL to sound speed changes)
            % Uses finite differences to approximate dTL/dc
            
            try
                % Create SSPs with small perturbations
                ssp_plus = obj.createLocalSSP(pos, c_nominal + obj.perturbation_size);
                ssp_minus = obj.createLocalSSP(pos, c_nominal - obj.perturbation_size);
                
                % disp(sum(abs(ssp_plus.ssp_grid - ssp_minus.ssp_grid), 'all'));

                % Compute forward models
                tl_plus = obj.forwardModelLocal(ssp_plus, pos, parameter_map);
                disp(tl_plus)
                tl_minus = obj.forwardModelLocal(ssp_minus, pos, parameter_map);
                disp(tl_minus)
                
                % Finite difference approximation
                J = (tl_plus - tl_minus) / (2 * obj.perturbation_size);
                
            catch ME
                warning('Jacobian computation failed: %s. Using approximate value.', ME.message);
                % Approximate Jacobian based on acoustic theory
                % Deeper receivers typically have higher sensitivity
                depth_factor = min(2.0, 1.0 + pos(3)/1000);  % Increase with depth
                J = -0.1 * depth_factor;  % Negative: higher c typically reduces TL
            end
        end
        
        function ssp_data = createLocalSSP(obj, pos, c_value)
            % Create local SSP structure for forward modeling
            % This creates a simplified SSP around the measurement location
            
            % Get current GP prediction as baseline
            if ~isempty(obj.X_obs)
                [mu_baseline, ~] = obj.predict(obj.X_grid);
                ssp_grid_baseline = reshape(mu_baseline, [length(obj.grid_y), length(obj.grid_x), length(obj.grid_z)]);
            else
                % Use prior if no observations yet
                mu_baseline = obj.mean_func(obj.X_grid(:,3));
                ssp_grid_baseline = reshape(mu_baseline, [length(obj.grid_y), length(obj.grid_x), length(obj.grid_z)]);
            end
            
            % Apply local perturbation around measurement location
            influence_radius_h = 2 * obj.ell_h;  % Convert to km
            influence_radius_v = 2 * obj.ell_v;         % Already in meters
            
            ssp_grid_modified = ssp_grid_baseline;
            
            % Find grid points within influence radius
            for i = 1:length(obj.grid_y)
                for j = 1:length(obj.grid_x)
                    for k = 1:length(obj.grid_z)
                        % Distance to measurement point
                        dx_km = obj.grid_x(j) - pos(1);
                        dy_km = obj.grid_y(i) - pos(2);
                        dz_m = obj.grid_z(k) - pos(3);
                        
                        dist_h = sqrt(dx_km^2 + dy_km^2) * 1000;  % Convert to meters
                        dist_v = abs(dz_m);
                        
                        % Apply Gaussian influence function
                        if dist_h <= influence_radius_h && dist_v <= influence_radius_v
                            weight_h = exp(-0.5 * (dist_h / obj.ell_h)^2);
                            weight_v = exp(-0.5 * (dist_v / obj.ell_v)^2);
                            weight = weight_h * weight_v;
                            
                            % Blend between baseline and target value
                            ssp_grid_modified(i,j,k) = (1-weight) * ssp_grid_baseline(i,j,k) + weight * c_value;
                        end
                    end
                end
            end
            
            % Return structure compatible with forward model
            ssp_data.grid_x = obj.grid_x;
            ssp_data.grid_y = obj.grid_y;
            ssp_data.grid_z = obj.grid_z;
            ssp_data.ssp_grid = ssp_grid_modified;
        end
        
        function tl = forwardModelLocal(obj, ssp_data, pos, parameter_map)
            % Forward model for inversion
            map = ParameterMap(parameter_map.getMap(), parameter_map.getEstimationParameterNames());
            map.set('ssp_grid', ssp_data.ssp_grid);
            writeSSP3D(obj.filename, ssp_data.grid_x, ssp_data.grid_y, ssp_data.grid_z, ssp_data.ssp_grid);
            tl = forward_model(map, pos, get_sim_settings());
            
        end
        
        function c = interpolateSSP(obj, ssp_data, pos)
            % Interpolate sound speed at given position
            
            % Clamp position to grid bounds
            x_clamp = max(min(pos(1), max(ssp_data.grid_x)), min(ssp_data.grid_x));
            y_clamp = max(min(pos(2), max(ssp_data.grid_y)), min(ssp_data.grid_y));
            z_clamp = max(min(pos(3), max(ssp_data.grid_z)), min(ssp_data.grid_z));
            
            % 3D interpolation
            c = interp3(ssp_data.grid_x, ssp_data.grid_y, ssp_data.grid_z, ...
                       permute(ssp_data.ssp_grid, [2,1,3]), ...
                       x_clamp, y_clamp, z_clamp, 'linear', 'extrap');
        end
        
        function setInversionMethod(obj, method)
            % Set the inversion method
            % method: 'gradient', 'bayesian', or 'hybrid'
            if ismember(lower(method), {'gradient', 'bayesian', 'hybrid'})
                obj.inversion_method = lower(method);
            else
                error('Invalid inversion method. Use: gradient, bayesian, or hybrid');
            end
        end
        
        function setInversionParameters(obj, varargin)
            % Set inversion parameters
            % Usage: setInversionParameters('max_iterations', 30, 'convergence_tol', 1e-4)
            
            p = inputParser;
            addParameter(p, 'max_iterations', obj.max_iterations, @isnumeric);
            addParameter(p, 'convergence_tol', obj.convergence_tol, @isnumeric);
            addParameter(p, 'perturbation_size', obj.perturbation_size, @isnumeric);
            parse(p, varargin{:});
            
            obj.max_iterations = p.Results.max_iterations;
            obj.convergence_tol = p.Results.convergence_tol;
            obj.perturbation_size = p.Results.perturbation_size;
        end
    end
end