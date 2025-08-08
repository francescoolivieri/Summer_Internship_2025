classdef Simulation
    % SIMULATION The main class for orchestrating a Bellhop simulation.
    
    properties
        Environment  % bellhop.Environment object
        Source       % bellhop.AcousticSource object
        Estimation   % bellhop.Estimation object (optional)
        
        Title (1,1) string = "Untitled"
        RunType (1,1) string = 'R' % Ray trace
        
        % Other top-level .env file parameters can be added here
    end
    
    methods
        function obj = Simulation(title)
            % SIMULATION Construct a new simulation object.
            %
            % sim = bellhop.Simulation("My First Simulation");
            
            if nargin > 0
                obj.Title = title;
            end
            
            % Initialize components with default objects
            obj.Environment = bellhop.Environment();
            obj.Source = bellhop.AcousticSource();
        end
        
        function run(obj)
            % RUN Executes the Bellhop simulation.
            
            disp("Starting Bellhop simulation...");
            
            % 1. Write the .env file
            obj.writeEnvFile('temp.env');
            
            % 2. Write the .bty file (if needed)
            if obj.Environment.Bottom.isFlat()
                disp('Flat bottom, no .bty file needed.');
            else
                obj.Environment.Bottom.writeBtyFile('temp.bty');
            end
            
            % 3. Write the .ssp file (if needed)
            obj.Environment.SSP.writeSspFile('temp.ssp');
            
            % 4. Run Bellhop executable (platform-dependent)
            % This is a placeholder for the actual system call
            disp('Running Bellhop... (placeholder)');
            % [status, cmdout] = system('bellhop.exe temp.env');
            
            disp("Simulation complete.");
        end
        
        function writeEnvFile(obj, filePath)
            % WRITEENVFILE Writes the main .env file for the simulation.
            
            fid = fopen(filePath, 'w');
            if fid == -1
                error('Cannot open file for writing: %s', filePath);
            end
            
            fprintf(fid, '''%s''\n', obj.Title);
            % Add other top-level settings here...
            
            % Write source info
            obj.Source.writeSourceInfo(fid);
            
            % Write SSP info
            obj.Environment.SSP.writeSspInfo(fid);
            
            % Write bottom info
            obj.Environment.Bottom.writeBottomInfo(fid);
            
            % Write run type
            fprintf(fid, '''%s''\n', obj.RunType);
            
            fclose(fid);
            
            disp(['.env file written to: ' filePath]);
        end
    end
end

