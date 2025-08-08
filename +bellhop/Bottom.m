classdef Bottom
    % BOTTOM Defines the ocean bottom properties.
    
    properties
        Type (1,1) string = "ACOUSTO-ELASTIC" % Or "FLAT", "HALF-SPACE", etc.
        
        % Properties for a half-space bottom
        SoundSpeed (1,1) double = 1600 % m/s
        Density (1,1) double = 1.5   % g/cm^3
        Attenuation (1,1) double = 0.5 % dB/wavelength
        
        % For non-flat bottoms, you'd store bathymetry data here
        BathymetryGrid = []
    end
    
    methods
        function flat = isFlat(obj)
            % ISFLAT Returns true if the bottom is flat.
            flat = isempty(obj.BathymetryGrid);
        end
        
        function writeBottomInfo(obj, fid)
            % WRITEBOTTOMINFO Writes bottom info to the .env file.
            
            % This is a simplified example for an acousto-elastic half-space
            fprintf(fid, '''A*''  ! Bottom type\n');
            fprintf(fid, '0.0 %f %f %f\n', obj.SoundSpeed, obj.Density, obj.Attenuation);
        end
        
        function writeBtyFile(obj, filePath)
            % WRITEBTYFILE Writes a .bty file for non-flat bottoms.
            if obj.isFlat()
                return;
            end
            
            % Placeholder for writing the bathymetry grid to a file
            disp(['Writing .bty file to: ' filePath ' (placeholder)']);
        end
    end
end

