classdef SSP
    % SSP Defines the Sound Speed Profile.
    
    properties
        ProfileType (1,1) string = "ISOVELOCITY" % e.g., "MUNK", "FILE", "TABULAR"
        
        % For isovelocity
        SoundSpeed (1,1) double = 1500 % m/s
        
        % For tabular data
        DepthZ = []
        SoundSpeedV = []
    end
    
    methods
        function writeSspInfo(obj, fid)
            % WRITESSPINFO Writes SSP information to the .env file.
            
            fprintf(fid, '''V''  ! SSP Type (isovelocity)\n');
            fprintf(fid, '1 0.0\n');
            fprintf(fid, '0.0 %f\n', obj.SoundSpeed);
            fprintf(fid, '5000.0 %f\n', obj.SoundSpeed); % Bellhop needs a bottom value
        end
        
        function writeSspFile(obj, filePath)
            % WRITESSPFILE Writes a .ssp file for tabular profiles.
            
            if obj.ProfileType ~= "TABULAR"
                return;
            end
            
            % Placeholder for writing the SSP data to a file
            disp(['Writing .ssp file to: ' filePath ' (placeholder)']);
        end
    end
end

