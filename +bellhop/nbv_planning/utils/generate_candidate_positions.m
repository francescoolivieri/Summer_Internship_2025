function positions = generate_candidate_positions(x, y, z, s)
    % Generate candidate positions in a sphere around current position
    n_candidates = 10; % Adjustable parameter
    positions = [];
    
    for i = 1:n_candidates
        % Random position within movement constraints
        dx = (rand - 0.5) * 2 * s.d_x * 3; % Within 3 steps
        dy = (rand - 0.5) * 2 * s.d_y * 3;
        dz = (rand - 0.5) * 2 * s.d_z * 3;
        
        new_pos = [x + dx, y + dy, z + dz];
        positions = [positions; new_pos];
    end
end
