function data=generate_data(data,s)

% Get index of last state
idx=find(isfinite(data.x),1,'last');

% Get the position of the sensor
pos=[data.x(idx) data.y(idx) data.z(idx)];

% Generate data
data.m(idx) = forward_model( data.true_params , pos, s) + s.sigma_tl_noise*randn;

end