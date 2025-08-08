function data = information_gain_nbv(data, s, idx)
    % Information gain-based approach (efficient sampling)
    candidate_positions = generate_candidate_positions(data.x(idx), data.y(idx), data.z(idx), s);
    
    %candidate_positions = [-0.2 1 20; -0.2 -1 20; 1 0 20; -1 0 20; 1 1 20; -1 -1 20; 1 -1 20; -1 1 20];
    %candidate_positions = [-0.2 1 20]

    best_info_gain = -inf;
    best_pos = [data.x(idx), data.y(idx), data.z(idx)];
    
    for i = 1:size(candidate_positions, 1)
        pos = candidate_positions(i, :);
        if is_valid_position(pos, s)
            info_gain = calculate_information_gain(pos, data.th_est(:, idx), data.Sigma_est(:, :, idx), s);
            if info_gain > best_info_gain
                best_info_gain = info_gain;
                best_pos = pos;
            end
        end
    end
    
    data.x(idx+1) = best_pos(1);
    data.y(idx+1) = best_pos(2);
    data.z(idx+1) = best_pos(3);
end
