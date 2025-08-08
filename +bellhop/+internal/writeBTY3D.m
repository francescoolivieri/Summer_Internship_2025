function writeBTY3D(name_btyfil, parameter_map)
%createbtyfil_3D Creates a bathymetry file from a scene structure.
%   This function extracts grid and depth data from a 'scene' struct
%   and writes it to a bathymetry file using 'writebdry3d'.
%
% Syntax:
%   createbtyfil_3d(name_btyfil, scene)
%
% Inputs:
%   bty_filename - The name for the output .bty file (e.g., 'MyBathy').
%   parameter_map        - A ParameterMap class containing 'scene': X, Y and floor(Z) attributes.
%   plot         - A boolean, true -> plots the bathymetry.

global extra_output

interp_type = 'R';

% Create Bathy structure

scene = parameter_map('scene');
original_X = scene.X(1, :);
original_Y = scene.Y(:, 1)';
original_depth = scene.floor;

% write bathymetry
try
    original_depth( isnan( original_depth ) ) = 0.0; % remove NaNs

    nx_original = length( original_X );
    ny_original = length( original_Y );

    num_sediment_types = size(parameter_map('sound_speed_sediment'), 1);

    if num_sediment_types > 1000 % NOT ACCURATE
        
        % Calculate where transitions should occur
        segment_width = nx_original / num_sediment_types;
        
        transition_step = 0.1; % Very small step for sharp transition

        % Find transition indices (where sediment type changes)
        transition_indices = [];
        for i = 1:(num_sediment_types-1)
            transition_idx = round(i * segment_width);
            if transition_idx > 1 && transition_idx <= nx_original
                transition_indices = [transition_indices, transition_idx];
            end
        end
        
        % Create new X coordinates with additional points at transitions
        new_X = [];
        new_X_indices = []; % Track which original indices correspond to new X
        
        for ix = 1:nx_original
            new_X = [new_X, original_X(ix)];
            new_X_indices = [new_X_indices, ix];
            
            % If this is a transition point, add a very close additional point
            if ismember(ix, transition_indices)
                new_X = [new_X, original_X(ix) + transition_step];
                new_X_indices = [new_X_indices, ix]; % Same original index for interpolation
            end
        end
        
        % Do the same for Y coordinates
        transition_indices_y = [];
        for i = 1:(num_sediment_types-1)
            transition_idx = round(i * (ny_original / num_sediment_types));
            if transition_idx > 1 && transition_idx <= ny_original
                transition_indices_y = [transition_indices_y, transition_idx];
            end
        end
        
        new_Y = [];
        new_Y_indices = [];
        
        for iy = 1:ny_original
            new_Y = [new_Y, original_Y(iy)];
            new_Y_indices = [new_Y_indices, iy];
            
            if ismember(iy, transition_indices_y)
                new_Y = [new_Y, original_Y(iy) + transition_step];
                new_Y_indices = [new_Y_indices, iy];
            end
        end
        
        % Interpolate depth values for new grid points
        new_depth = zeros(length(new_Y), length(new_X));
        for iy = 1:length(new_Y)
            orig_iy = new_Y_indices(iy);
            for ix = 1:length(new_X)
                orig_ix = new_X_indices(ix);
                new_depth(iy, ix) = original_depth(orig_iy, orig_ix);
            end
        end
        
        Bdry.X = new_X;
        Bdry.Y = new_Y;
        Bdry.depth = new_depth;
    else
        Bdry.X = original_X;
        Bdry.Y = original_Y;
        Bdry.depth = original_depth;
    end

    nx = length( Bdry.X );
    ny = length( Bdry.Y );
    
    fid = fopen( name_btyfil, 'wt' );

    fprintf( fid, '''RL'' \n' );
    
    fprintf( fid, '%i \n', nx );
    %fprintf( fid, '%f %f /', Bathy.X( 1 ), Bathy.X( end ) );
    
    fprintf( fid, '%f ', Bdry.X( 1 : end ) );
    fprintf( fid, '\n');
    
    fprintf( fid, '%i \n', ny );
    %fprintf( fid, '%f %f /', Bathy.Y( 1 ), Bathy.Y( end ) );
    fprintf( fid, '%f ', Bdry.Y( 1 : end ) );
    fprintf( fid, '\n');
    
    for iy = 1 : ny
        fprintf( fid, '%9.3f ', Bdry.depth( iy, : ) );
        fprintf( fid, '\n');
    end
    
    % if 'long' format append a matrix with province types
    
    if ( length( interp_type ) > 1 )
        if ( interp_type( 2 : 2 ) == 'L' )
            for iy = 1 : ny
                fprintf( fid, '%3i ', Bdry.province( iy, : ) );
                fprintf( fid, '\n');
            end
    
            NProvinces = max( max( Bdry.province ) );
            fprintf( fid, '%i \n', NProvinces );
            for iProv = 1 : NProvinces
                fprintf( fid, '%f ', Bdry.geotype( iProv, : ) );
                fprintf( fid, '\n' );
            end
        end
    end
    
    
    sediment_type_counter = 1;
    for iy = 1 : ny
        for ix = 1 : nx

            if num_sediment_types > 1 

                if rem(ix, round(nx/num_sediment_types)) ~= 0
                    fprintf( fid, '%d ', sediment_type_counter);
                else
                    fprintf( fid, '%d ', sediment_type_counter);
                    sediment_type_counter = sediment_type_counter + 1;
                end 

            else
                fprintf( fid, '1 ');
            end

        end
        sediment_type_counter = 1;
        fprintf( fid, '\n');
    end

    fprintf( fid, '%d \n', num_sediment_types);
    sp_arr = parameter_map('sound_speed_sediment');
    rho_arr = parameter_map('density_sediment');
    att_arr = parameter_map('attenuation_sediment');
    for i = 1:num_sediment_types
        
        fprintf( fid, '%f 0 %f %f 0 \n',  sp_arr(i), rho_arr(i), att_arr(i));
    end

    fclose( fid );

catch ME
    error('Failed to write bathymetry file. Error: %s', ME.message);
end

if extra_output
    figure
    title('BTY plot','FontSize',10);
    plotbdry3d(name_btyfil)
end

end


