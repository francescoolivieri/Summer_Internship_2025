function info_gain = calculate_information_gain(pos, mu_th, Sigma_thth, s)
    % Calculate information gain using entropy reduction
    try
        fprintf("Position %f %f %f\n", pos);
        [~, Sigma_new] = step_ukf_filter(nan, @(ParameterMap)forward_model(ParameterMap, pos, s), mu_th, Sigma_thth, s.Sigma_rr, s);
        info_gain = sum(diag(Sigma_thth)) - sum(diag(Sigma_new));
        fprintf("Information Gain for %f %f %f is: %f \n", pos, info_gain);
        
    catch ME
        disp('Error in calculate_information_gain:');
        disp(getReport(ME));  % This gives full stack trace and message
        info_gain = -inf;
    end
end