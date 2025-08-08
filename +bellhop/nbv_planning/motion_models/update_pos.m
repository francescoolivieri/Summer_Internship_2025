function [x_child, y_child, z_child] = update_pos(x_parent, y_parent, z_parent, s, action)
    % Enhanced motion model with 27 possible actions (3x3x3 grid)
    % Original 9 actions only considered movement in XY plane for each Z level
    % This considers all combinations of {-1, 0, 1} for each axis
    
    % Define all 27 movement combinations
    moves = [];
    for dx = -1:1
        for dy = -1:1
            for dz = -1:1
                moves = [moves; dx, dy, dz];
            end
        end
    end
    
    % Apply the selected movement
    if action < 1 || action > 27
        error('Action must be between 1 and 27');
    end
    
    move = moves(action, :);
    x_child = x_parent + move(1) * s.d_x;
    y_child = y_parent + move(2) * s.d_y;
    z_child = z_parent + move(3) * s.d_z;
    
    % Check bounds and constraints
    if z_child > s.z_max || z_child < s.z_min || ...
       x_child < s.x_min || x_child > s.x_max || ...
       y_child < s.y_min || y_child > s.y_max || ...
       sqrt(x_child^2 + y_child^2) > s.x_max
        x_child = nan;
        y_child = nan;
        z_child = nan;
    end
end 
