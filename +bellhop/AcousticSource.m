classdef AcousticSource
    % ACOUSTICSOURCE Defines the acoustic source for the simulation.
    
    properties
        Frequency (1,1) double = 3000 % Hz
        Depth (1,1) double = 50     % meters
        
        % Can add directivity, etc. later
    end
    
    methods
        function obj = AcousticSource(freq, depth)
            % ACOUSTICSOURCE Construct a new AcousticSource object.
            
            if nargin > 0
                obj.Frequency = freq;
            end
            if nargin > 1
                obj.Depth = depth;
            end
        end
        
        function writeSourceInfo(obj, fid)
            % WRITESOURCEINFO Writes source information to the .env file.
            fprintf(fid, '%f  ! Source Frequency (Hz)\n', obj.Frequency);
            fprintf(fid, '1    ! Number of sources\n');
            fprintf(fid, '0.0 / %f\n', obj.Depth);
        end
    end
end

