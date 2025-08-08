classdef Estimation < handle
    % ESTIMATION Manages parameter estimation tasks.
    
    properties
        ParametersToEstimate (1,:) cell = {}
        PriorMean (1,:) double = []
        PriorCovariance (:,:) double = []
        
        EstimatedMean (1,:) double
        EstimatedCovariance (:,:) double
        
        FilterType (1,1) string = "UKF" % Unscented Kalman Filter
    end
    
    methods
        function obj = Estimation(paramNames, priorMean, priorCov)
            % ESTIMATION Construct a new Estimation object.
            %
            % est = bellhop.Estimation(...
            %         {'Bottom.SoundSpeed', 'Bottom.Density'}, ...
            %         [1650, 1.6], ...
            %         diag([50^2, 0.2^2]) ...
            % );
            
            if nargin > 0
                obj.ParametersToEstimate = paramNames;
            end
            if nargin > 1
                obj.PriorMean = priorMean;
                obj.EstimatedMean = priorMean;
            end
            if nargin > 2
                obj.PriorCovariance = priorCov;
                obj.EstimatedCovariance = priorCov;
            end
        end
        
        function setup(obj, simulation)
            % SETUP Initializes the estimation process for a given simulation.
            % It samples the "true" parameters from the prior distribution 
            % and dynamically assigns them to the simulation object.

            if isempty(obj.ParametersToEstimate)
                disp('No parameters to estimate.');
                return;
            end

            disp('Setting up estimation task: sampling true parameters...');
            
            % Sample the "true" values from the prior distribution
            true_values = mvnrnd(obj.PriorMean, obj.PriorCovariance);
            
            % Assign these true values to the simulation object
            for i = 1:length(obj.ParametersToEstimate)
                param_path = obj.ParametersToEstimate{i};
                param_value = true_values(i);
                
                try
                    % This uses a helper function to set nested properties
                    simulation = obj.setNestedProperty(simulation, param_path, param_value);
                catch e
                    error('Failed to set parameter ''%s''. Error: %s', param_path, e.message);
                end
            end
            
            disp('True parameters have been set in the simulation object.');
        end
        
        function update(obj, measurement)
            % UPDATE Updates the parameter estimates based on a new measurement.
            
            % This is where the UKF (or other filter) logic will go.
            % It will take the measurement, run the forward model, and
            % update obj.EstimatedMean and obj.EstimatedCovariance.
            
            disp('Updating estimates with new measurement... (placeholder)');
        end
    end

    methods (Access = private)
        function obj_out = setNestedProperty(~, obj_in, field_path, value)
            % SETNESTEDPROPERTY Sets a property value in a nested object structure.
            %
            % obj_out = setNestedProperty(obj_in, 'Prop1.NestedProp.Value', 123);

            path_parts = strsplit(field_path, '.');
            
            % This is a bit tricky with value classes (like most MATLAB
            % objects). We need to get the nested object, modify it, and
            % then set it back at each level.
            
            s = struct();
            s.type = '.';
            s.subs = path_parts;
            
            obj_out = subsasgn(obj_in, s, value);
        end
    end
end

