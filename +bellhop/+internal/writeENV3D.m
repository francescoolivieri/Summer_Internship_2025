function writeENV3D(filename, s, map)
%WRITEENV3D Summary of this function goes here
%   Detailed explanation goes here

% Open environment file
fid = fopen(filename, "w");

if fid == -1
    error('Could not open file %s for writing.', filename);
end

% Update content
fprintf(fid, '''True 3D profile''        ! TITLE\n');
fprintf(fid, '%.2f                  ! FREQ (Hz)\n', map('source_frequency')); % Insert new_frequency here
fprintf(fid, '1                     ! NMEDIA\n');
if s.sim_use_ssp_file

    fprintf(fid, '''HVW''               ! SSPOPT (Analytic or C-linear interpolation)\n');
    fprintf(fid, '51 0.0  %.2f 		! DEPTH of bottom (m)\n', s.sim_max_depth);
    fprintf(fid, '   0.0  /\n');
    fprintf(fid, '   10.0  /\n');
    fprintf(fid, '   20.0  /\n');
    fprintf(fid, '   30.0  /\n');
    fprintf(fid, '   40.0  /\n');
    fprintf(fid, '   %.2f  /\n', s.sim_max_depth);


else
    fprintf(fid, '''CVW''				! SSPOPT (Analytic or C-linear interpolation)\n');
    fprintf(fid, '51 0.0  %.2f 		! DEPTH of bottom (m)\n', s.sim_max_depth);
    fprintf(fid, '   0.0  1535.52  /\n');
    fprintf(fid, '   10.0  1530.29  /\n');
    fprintf(fid, '   20.0  1526.69  /\n');
    fprintf(fid, '   40.0  1530.69  /\n');
    fprintf(fid, '   %.2f  1600.30  /\n', s.sim_max_depth);

end
    
if s.sim_use_bty_file
    fprintf(fid, '''A~'' 0.0 \n');
else
    fprintf(fid, '''A'' 0.0 \n');
end

ss_sediment = map('sound_speed_sediment');
rho_sediment = map('density_sediment');
att_sediment = map('attenuation_sediment');

fprintf(fid, ' %.2f   %f 0 %f %f 0 /\n', s.sim_max_depth, ss_sediment(1) , rho_sediment(1), att_sediment(1));   %fprintf(fid, ' %.2f   %.2f 0.0 1.5 0.5 /\n', s.sim_max_depth, s.bottom_ssp);
fprintf(fid, '1                 ! NSx number of source coordinates in x\n');
fprintf(fid, '%.2f /             ! x coordinate of source (km)\n', map('source_x'));
fprintf(fid, '1                 ! NSy number of source coordinates in y\n');
fprintf(fid, '%.2f /             ! y coordinate of source (km)\n', map('source_y'));
fprintf(fid, '1                 ! NSz\n');
fprintf(fid, '%.2f /          ! Sz(1 : NSz) (m)\n', map('source_depth')); 
fprintf(fid, '200               ! NRz\n');
fprintf(fid, '0 %.2f /          ! Rz(1 : NRz) (m)\n', s.sim_max_depth);
fprintf(fid, '1000              ! NRr\n');
fprintf(fid, '0.0  %.2f /      ! Rr(1 : NRr ) (km)\n', s.sim_range);
fprintf(fid, '%d                 ! Ntheta (number of bearings)\n', s.sim_num_bearing);
fprintf(fid, '0.0 360.0 /         ! bearing angles (degrees)\n');
if s.sim_accurate_3d
    fprintf(fid, '''CG   3''          ! ''R/C/I/S''\n');
else
    fprintf(fid, '''CG   2''          ! ''R/C/I/S''\n');
end
fprintf(fid, '1001              ! Nalpha\n');
fprintf(fid, '-89 89 /    ! alpha1, 2 (degrees) Elevation/declination angle fan\n');
fprintf(fid, '10            ! Nbeta\n');
fprintf(fid, '0  360 /           ! beta1, beta2 (degrees) bearine angle fan\n');
fprintf(fid, '0.0  1.55 1.55 %.2f ! STEP (m), Box%%x (km) Box%%y (km) Box%%z (m)\n', s.sim_max_depth+.5); % Use %% for literal %

% Close the file.
fclose(fid);



end