function data = ukf(data, s)

% Get index of last state
idx=find(isfinite(data.m),1,'last');

% Get the position of the sensor
pos=[data.x(idx) data.y(idx) data.z(idx)];
    
% Get measurement
m=data.m(:, idx);

% Update the filter state given the measurements
 [data.th_est(:, idx), data.Sigma_est(:, :, idx)]= step_ukf_filter( ...
                m, @(ParameterMap)forward_model(ParameterMap, pos, s), ...
                data.th_est(:, idx-1), data.Sigma_est(:, :, idx-1), s.Sigma_rr, s);
 
 data.estimated_params.update(data.th_est(:, idx)) 

end
