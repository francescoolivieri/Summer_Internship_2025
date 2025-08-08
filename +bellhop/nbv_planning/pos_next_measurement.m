function data = pos_next_measurement(data, s)
    % State-of-the-art NBV planning with multiple approaches
    
    % Get index of last state
    idx = find(isfinite(data.x), 1, 'last');
    
    % Check if sensor management or not 
    if s.sm
        
        % Select method based on settings
        switch s.nbv_method
            case 'tree_memoized'
                % Optimized tree-based approach with memoization
                data = tree_memoized_nbv(data, s, idx);

            case 'rrt_star'
                % RRT* based NBV planning
                data = rrt_star_based_nbv(data, s, idx);
                
            case 'information_gain' 
                % Information gain-based approach 
                data = information_gain_nbv(data, s, idx);

            case 'multi_objective' % NOT TESTED
                % Multi-objective optimization approach
                data = multi_objective_nbv(data, s, idx);
                
            case 'bayesian_opt' % NOT TESTED
                % Bayesian optimization approach
                data = bayesian_optimization_nbv(data, s, idx);
                
            otherwise
                % Move as lawn mower
                data = lawnmower_pattern(data, s, idx);
        end
        
    else

        % Move as lawn mower (original behavior)
        data = lawnmower_pattern(data, s, idx);
    end
end








function data = lawnmower_pattern(data, s, idx)
    % Circular lawnmower pattern: sqrt(x^2 + y^2) < s.x_max

    temp_x = data.x(idx);
    temp_y = data.y(idx);
    temp_z = data.z(idx);
    
    % Loop indefinitely until a valid point within the circle is found.
    while true
        % Propose the next point using the original rectangular lawnmower logic.
        if (temp_y + s.d_y) < s.y_max
            % Move along the y-axis (primary scan direction).
            temp_y = temp_y + s.d_y;
        else
            % Reached the end of a y-scan. Reset y and move along the x-axis.
            temp_y = s.y_min;
            if (temp_x + s.d_x) < s.x_max
                % Move along the x-axis (secondary scan direction).
                temp_x = temp_x + s.d_x;
            else
                % Reached the end of an x-scan. Reset x and move along the z-axis.
                temp_x = s.x_min;
                if (temp_z + s.d_z) < s.z_max
                    % Move along the z-axis (tertiary scan direction).
                    temp_z = temp_z + s.d_z;
                else
                    % Reached the end of the z-scan, so reset z. This causes
                    % the pattern to wrap around.
                    temp_z = s.z_min;
                end
            end
        end
    
        % Check if the newly proposed point (temp_x, temp_y) is inside the circle.
        if sqrt(temp_x^2 + temp_y^2) < s.x_max
            % The point is valid. Assign it to the output structure.
            data.x(idx+1) = temp_x;
            data.y(idx+1) = temp_y;
            data.z(idx+1) = temp_z;
            
            % Exit the function since we have found the next valid point.
            return;
        end
        
        % If the point is outside the circle, the loop continues. The updated
        % temporary coordinates will be used to calculate the next point in the
        % sequence, effectively skipping the invalid ones.
    end
end
