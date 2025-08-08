function data = bayesian_optimization_nbv(data, s, idx)
    % Enhanced Bayesian optimization approach
    pos_x = optimizableVariable('x', [s.x_min, s.x_max], 'Type', 'real');
    pos_y = optimizableVariable('y', [s.y_min, s.y_max], 'Type', 'real');
    pos_z = optimizableVariable('z', [s.z_min, s.z_max], 'Type', 'real');
    
    % Enhanced objective function
    objectiveFcn = @(pos) enhanced_prediction_error_loss([pos.x, pos.y, pos.z], ...
        data.th_est(:,idx), squeeze(data.Sigma_est(:,:,idx)), s);
    
    results = bayesopt(objectiveFcn, [pos_x, pos_y, pos_z], ...
        'IsObjectiveDeterministic', false, ...
        'AcquisitionFunctionName', 'expected-improvement-per-second-plus', ...
        'GPActiveSetSize', 300, ...
        'NumSeedPoints', 20, ...
        'MaxObjectiveEvaluations', 10, ...
        'ExplorationRatio', 0.1, ...
        'Verbose', 0);
    
    data.x(idx+1) = results.XAtMinObjective.x;
    data.y(idx+1) = results.XAtMinObjective.y;
    data.z(idx+1) = results.XAtMinObjective.z;
end

function loss = enhanced_prediction_error_loss(pos, mu_th, Sigma_thth, s)
    % Enhanced version of prediction error loss with additional considerations
    if sqrt(pos(1)^2 + pos(2)^2) > s.x_max
        loss = 1e6;
        return
    end
    
    % Original loss calculation
    base_loss = prediction_error_loss(pos, mu_th, Sigma_thth, s);
    
    % Add exploration bonus for unvisited areas
    exploration_bonus = calculate_exploration_bonus(pos, s);
    
    % Combine with weights
    loss = base_loss - 0.1 * exploration_bonus; % Negative because we want to minimize loss
end

function bonus = calculate_exploration_bonus(pos, s)
    % Simple exploration bonus - could be enhanced with actual visited position tracking
    % For now, bonus increases with distance from origin
    bonus = sqrt(pos(1)^2 + pos(2)^2 + pos(3)^2) / sqrt(s.x_max^2 + s.y_max^2 + s.z_max^2);
end 