classdef Environment
    % ENVIRONMENT Defines the ocean environment for the simulation.
    
    properties
        SSP          % Sound Speed Profile object
        Bottom       % Bottom properties object
        SeaSurface   % Sea surface properties (e.g., wave height)
        
        Depth (1,1) double = 100 % meters
    end
    
    methods
        function obj = Environment()
            % ENVIRONMENT Construct a new Environment object.
            
            % Initialize with default components
            obj.SSP = bellhop.SSP();
            obj.Bottom = bellhop.Bottom();
        end
    end
end

