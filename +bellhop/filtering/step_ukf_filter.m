function [mu_th,Sigma_thth]=step_ukf_filter(y,f,mu_th,Sigma_thth,Sigma_rr, s)


    % Apply unscented transform to estimate mean and covariance
    [mu_tl, Sigma_tltl, Sigma_thtl] = unscented_transform(f, mu_th, Sigma_thth, s);

    % Compute Kalman gain
    K = Sigma_thtl / (Sigma_tltl+Sigma_rr);

    % Update mean
    mu_th = mu_th + K * (y - mu_tl);

    % Update covariance
    Sigma_thth = Sigma_thth - K * (Sigma_tltl+Sigma_rr) * K';

end