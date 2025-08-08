function data = multi_objective_nbv(data, s, idx)
    % Multi-objective optimization considering exploration vs exploitation
    candidate_positions = generate_candidate_positions(data.x(idx), data.y(idx), data.z(idx), s);
    
    best_score = -inf;
    best_pos = [data.x(idx), data.y(idx), data.z(idx)];
    
    for i = 1:size(candidate_positions, 1)
        pos = candidate_positions(i, :);
        if is_valid_position(pos, s)
            % Combine multiple objectives
            info_gain = calculate_information_gain(pos, data.th_est(:, idx), data.Sigma_est(:, :, idx), s);
            coverage = calculate_coverage_score(pos, s);
            efficiency = calculate_efficiency_score(pos, [data.x(idx), data.y(idx), data.z(idx)], s);
            
            % Weighted combination
            score = s.w_info * info_gain + s.w_coverage * coverage + s.w_efficiency * efficiency;
            
            if score > best_score
                best_score = score;
                best_pos = pos;
            end
        end
    end
    
    data.x(idx+1) = best_pos(1);
    data.y(idx+1) = best_pos(2);
    data.z(idx+1) = best_pos(3);
end



function coverage = calculate_coverage_score(pos, s)
    % Simple coverage score based on distance from explored areas
    % This is a placeholder - could be enhanced with actual coverage maps
    coverage = sqrt(pos(1)^2 + pos(2)^2 + pos(3)^2) / sqrt(s.x_max^2 + s.y_max^2 + s.z_max^2);
end

function efficiency = calculate_efficiency_score(pos, current_pos, s)
    % Efficiency score based on movement cost
    distance = norm(pos - current_pos);
    max_distance = sqrt((2*s.d_x)^2 + (2*s.d_y)^2 + (2*s.d_z)^2);
    efficiency = 1 - (distance / max_distance); % Closer positions are more efficient
end
