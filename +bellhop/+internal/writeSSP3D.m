function writeSSP3D(filename, x, y, z, c)
% WRITE_SSP3D writes a 3D sound speed profile to a file for Bellhop3D
% Inputs:
%   filename : output .ssp file name
%   x        : vector of x-coordinates [in km]
%   y        : vector of y-coordinates [in km]
%   z        : vector of z-coordinates [in meters, increasing downward]
%   c        : 3D matrix of sound speeds [Nx x Ny x Nz] in m/s

    c = permute(c, [3 2 1]); % reshape to [Nz x Ny x Nx]

    % Sanity checks
    Nx = length(x);
    Ny = length(y);
    Nz = length(z);

    assert(isequal(size(c), [Nz, Ny, Nx]), 'c must be [Nz x Ny x Nx]');

    % Open file
    fid = fopen(filename + ".ssp", 'w');
    if fid == -1
        error('Could not open file for writing: %s', filename);
    end

    % Write x-axis
    fprintf(fid, '%d\n', Nx);
    fprintf(fid, '%.6f ', x );  % Convert to km
    fprintf(fid, '\n');

    % Write y-axis
    fprintf(fid, '%d\n', Ny);
    fprintf(fid, '%.6f ', y );  % Convert to km
    fprintf(fid, '\n');

    % Write z-axis
    fprintf(fid, '%d\n', Nz);
    fprintf(fid, '%.6f ', z);  % Keep in meters
    fprintf(fid, '\n');

    % Write sound speed matrices per z-layer
    for iz = 1:Nz
        c2d = squeeze(c(iz, :, :));  % size [Ny x Nx]
        for iy = 1:Ny
            fprintf(fid, '%.6f ', c2d(iy, :));  % row of Nx values
            fprintf(fid, '\n');
        end
    end

    fclose(fid);
end



% Nx = length(x);
% Ny = length(y); 
% Nz = length(z);
% 
% % Verify input dimensions
% c = permute(c, [3 2 1]);
% assert(isequal(size(c), [Nz, Ny, Nx]), 'c must be [Nz x Ny x Nx]');
% 
% fid = fopen(filename + ".ssp", 'w');
% if fid == -1
%     error('Could not open file for writing: %s', filename);
% end
% 
% % Write dimensions and coordinates
% fprintf(fid, '%d\n', Nx);
% fprintf(fid, '%.6f ', x);
% fprintf(fid, '\n');
% fprintf(fid, '%d\n', Ny);
% fprintf(fid, '%.6f ', y);
% fprintf(fid, '\n');
% fprintf(fid, '%d\n', Nz);
% fprintf(fid, '%.6f ', z);
% fprintf(fid, '\n');
% 
% % Write sound speed data
% for iz = 1:Nz
%     c2d = squeeze(c(iz, :, :));  % [Ny x Nx]
%     c2d = c2d';                  % Transpose to [Nx x Ny] for output
%     for ix = 1:Nx
%         fprintf(fid, '%.6f ', c2d(ix, :)); % Write Nx rows of Ny values
%         fprintf(fid, '\n');
%     end
% end
% 
% fclose(fid);
% end