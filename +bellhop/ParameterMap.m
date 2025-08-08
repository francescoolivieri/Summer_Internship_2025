classdef ParameterMap < handle
    % ParameterMap - Represents a collection of underwater simulation parameters
    % 
    % This class encapsulates a single parameter map with methods for
    % accessing, updating, and converting parameters.
    
    properties (Access = private)
        param_map
        estimation_param_names
    end
    
    properties (Dependent)
        all_keys    % Get all parameter names
    end
    
    methods
        function obj = ParameterMap(initial_params, estimation_names)
            % Constructor
            %
            % Inputs:
            %   initial_params - containers.Map or struct with initial values
            %   estimation_names - cell array of parameter names for estimation
            
            if isa(initial_params, 'containers.Map')
                obj.param_map = containers.Map(initial_params.keys, initial_params.values);
            else
                % Constructor with default values (defined in sim_settings)
                obj.param_map = containers.Map();
                
                obj.param_map('source_frequency') = initial_params.sim_frequency;

                % Acoustic parameters
                obj.param_map('sound_speed_water') = initial_params.sim_param_sp_water; % m/s
                obj.param_map('sound_speed_sediment') = initial_params.sim_param_sp_sediment; % m/s
                obj.param_map('density_sediment') = initial_params.sim_param_density_sediment; % g/cm³
                obj.param_map('attenuation_sediment') = initial_params.sim_param_attenuation_sediment; % dB/λ
                
                obj.param_map('ssp_grid') = nan(length(initial_params.Ocean_y_min:initial_params.Ocean_step:initial_params.Ocean_y_max), ...
                                                length(initial_params.Ocean_x_min:initial_params.Ocean_step:initial_params.Ocean_x_max), ...
                                                length(0:initial_params.Ocean_z_step:initial_params.sim_max_depth));              % dB/λ
                
                % Geometric parameters
                obj.param_map('water_depth') = initial_params.sim_max_depth; % m   % NOT TESTED
                obj.param_map('source_x') = initial_params.sim_source_x; % m
                obj.param_map('source_y') = initial_params.sim_source_y; % m
                obj.param_map('source_depth') = initial_params.sim_source_depth; % m

                obj.param_map('scene') = initial_params.scene; % m
    
                obj.estimation_param_names = {};
            
            end
            
            if nargin > 1
                obj.estimation_param_names = estimation_names;
            else
                obj.estimation_param_names = {};
            end
        end

        
        
        function keys = get.all_keys(obj)
            % Get all parameter names
            keys = obj.param_map.keys;
        end
        
        function value = get(obj, param_name)
            % Get parameter value
            if obj.param_map.isKey(param_name)
                value = obj.param_map(param_name);
            else
                error('Parameter "%s" not found', param_name);
            end
        end
        
        function set(obj, param_name, value)
            % Set parameter value
            obj.param_map(param_name) = value;

            if obj.isSedimentParameter(param_name)
                obj.applySedimentPadding(obj.param_map);  % Add padding
            end
        end
        
        function param_array = asArray(obj, param_names)
            % Get specified parameters as array
            if nargin < 2
                param_names = obj.estimation_param_names;
            end
            param_array = obj.getParameterArray(param_names);
        end
        
        function update(obj, param_values, param_names)
            % Update parameters with new values
            %
            % Inputs:
            %   param_values - array of new parameter values
            %   param_names - optional, parameter names to update (default: estimation params)
            
            if nargin < 3
                param_names = obj.estimation_param_names;
            end
            
            cont = 1;
            unique_names = unique(param_names, 'stable');
            
            for i = 1:length(unique_names)
                param_name = unique_names{i};
                % Count how many times this parameter appears
                count = sum(strcmp(param_names, param_name));
                
                if count == 1
                    obj.set(param_name, param_values(cont));
                    cont = cont + 1;
                else
                    % Handle multi-valued parameters
                    values = param_values(cont:cont+count-1);
              
                    obj.set(param_name, values);
                    cont = cont + count;
                end
            end
        end
        
        
        function map = getMap(obj)
            % Get copy of internal map
            map = containers.Map(obj.param_map.keys, obj.param_map.values);
            %map = obj.param_map;
        end
        
        function has = hasParameter(obj, param_name)
            % Check if parameter exists
            has = obj.param_map.isKey(param_name);
        end
        
        function copyFrom(obj, other_param_map)
            % Copy values from another ParameterMap or containers.Map
            if isa(other_param_map, 'ParameterMap')
                source_map = other_param_map.getMap();
            else
                source_map = other_param_map;
            end
            
            keys = source_map.keys;
            for i = 1:length(keys)
                obj.set(keys{i}, source_map(keys{i}));
            end
        end
        
        function applySedimentPadding(obj, default_param_map)
            % Apply sediment parameter padding using the standalone function
            obj.param_map = paddingSedimentParams(obj.param_map, default_param_map);
        end
        
        function print(obj, title_str)
            % Display parameters
            if nargin < 2
                title_str = 'Parameters';
            end
            
            fprintf('\n=== %s ===\n', title_str);
            T = table(obj.param_map.keys', obj.param_map.values', 'VariableNames', {'Key', 'Value'});
            disp(T);
        end
        
        function diff = compareWith(obj, other_param_map, param_names)
            % Compare with another parameter map
            if nargin < 3
                param_names = obj.estimation_param_names;
            end
            
            diff = struct();
            for i = 1:length(param_names)
                name = param_names{i};
                if obj.hasParameter(name)
                    if isa(other_param_map, 'ParameterMap') && other_param_map.hasParameter(name)
                        diff.(name) = obj.get(name) - other_param_map.get(name);
                    elseif isa(other_param_map, 'containers.Map') && other_param_map.isKey(name)
                        diff.(name) = obj.get(name) - other_param_map(name);
                    end
                end
            end
        end
        
        function setEstimationParameterNames(obj, param_names)
            % Set which parameters are used for estimation
            obj.estimation_param_names = param_names;
        end
        
        function names = getEstimationParameterNames(obj)
            % Get estimation parameter names
            names = obj.estimation_param_names;
        end

        % function printEstimationParameters()
        % 
        % 
        % end

    end
    
    methods (Access = private)
        function param_array = getParameterArray(obj, param_names)
            % Convert specified parameters to array
            param_array = zeros(length(param_names), 1);
            cont = 1;
            
            unique_names = unique(param_names, 'stable');
            for i = 1:length(unique_names)
                param_name = unique_names{i};
                if obj.param_map.isKey(param_name)
                    values = obj.param_map(param_name);
                    for j = 1:length(values)
                        param_array(cont) = values(j);
                        cont = cont + 1;
                    end
                else
                    error('Parameter "%s" not found in parameter map', param_name);
                end
            end
        end

        function is_sediment = isSedimentParameter(obj, param_name)
            sediment_keys = {'density_sediment', 'sound_speed_sediment', 'attenuation_sediment'};
            is_sediment = any(strcmp(param_name, sediment_keys));
        end
    end
end

% =========================================================================
% STANDALONE FUNCTIONS
% =========================================================================


function param_map = paddingSedimentParams(param_map, default_param_map)
    % Harmonizes sediment-related parameters to have equal lengths
    % If one or more sediment parameters have fewer elements, they are padded
    % with their default values.
    
    sediment_keys = {'density_sediment', 'sound_speed_sediment', 'attenuation_sediment'};
    lengths = zeros(1, numel(sediment_keys));
    
    % Get current lengths
    for i = 1:numel(sediment_keys)
        key = sediment_keys{i};
        if isKey(param_map, key)
            value = param_map(key);
            if isnumeric(value)
                lengths(i) = length(value);
            else
                error('Parameter "%s" must be numeric.', key);
            end
        else
            warning('Parameter "%s" not found in parameter map.', key);
            lengths(i) = 0;
        end
    end
    
    max_len = max(lengths);
    
    % Pad each key to match max_len
    for i = 1:numel(sediment_keys)
        key = sediment_keys{i};
        if isKey(param_map, key)
            val = param_map(key);
            n_missing = max_len - length(val);
            if n_missing > 0
                default_val = default_param_map(key);
                pad = repmat(default_val, n_missing, 1);
                param_map(key) = [val; pad];
            end
        end
    end
end

