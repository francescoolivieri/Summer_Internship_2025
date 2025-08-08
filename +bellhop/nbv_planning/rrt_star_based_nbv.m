function data = rrt_star_based_nbv(data, s, idx)
    % RRT* based NBV planning with path optimization
    
    % Initialize RRT* parameters
    if ~isfield(s, 'rrt_max_iter'), s.rrt_max_iter = 10; end
    if ~isfield(s, 'rrt_step_size'), s.rrt_step_size = 1; end
    if ~isfield(s, 'rrt_goal_bias'), s.rrt_goal_bias = 0.05; end
    if ~isfield(s, 'rrt_search_radius'), s.rrt_search_radius = 8.0; end
    
    % Current position
    current_pos = [data.x(idx), data.y(idx), data.z(idx)];
    
    % Build RRT* tree with path optimization
    rrt_star_tree = build_rrt_star_exploration_tree(current_pos, s);
    
    % Evaluate nodes considering both information gain and path cost
    best_score = -inf;
    best_pos = current_pos;
    
    for i = 1:length(rrt_star_tree.nodes)
        node = rrt_star_tree.nodes(i);
        pos = node.position;    
        
        if is_valid_position(pos, s)
            info_gain = calculate_information_gain(pos, data.th_est(:, idx), data.Sigma_est(:, :, idx), s);
            path_cost = node.cost; % Cost from RRT* optimization
            
            % Combine information gain with path efficiency
            score = info_gain - 0.0 * path_cost; % Prefer high info gain, low path cost
            
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


function rrt_tree = build_rrt_star_exploration_tree(start_pos, s)
    % Build RRT* tree with path optimization
    
    % Initialize tree
    rrt_tree.nodes = struct('position', {}, 'parent', {}, 'children', {}, 'cost', {});
    rrt_tree.nodes(1).position = start_pos;
    rrt_tree.nodes(1).parent = -1;
    rrt_tree.nodes(1).children = [];
    rrt_tree.nodes(1).cost = 0;
    
    for iter = 1:s.rrt_max_iter
        % Sample random point
        if rand < s.rrt_goal_bias
            rand_point = sample_biased_position(s);
        else
            rand_point = sample_random_position(s);
        end
        
        fprintf("RRT* rand_point: %f %f %f", rand_point);

        % Find nearest node
        [nearest_idx, nearest_node] = find_nearest_node(rrt_tree, rand_point);
        
        % Extend toward random point
        new_pos = extend_toward_point(nearest_node.position, rand_point, s.rrt_step_size, s);
        
        fprintf("\t RRT* node_ext_point : %f %f %f", new_pos);
        
        if is_valid_position(new_pos, s)
            % Find nodes within search radius
            near_indices = find_near_nodes(rrt_tree, new_pos, s.rrt_search_radius);
            
            % Choose parent with minimum cost
            min_cost = inf;
            best_parent = nearest_idx;
            
            for near_idx = near_indices
                edge_cost = norm(new_pos - rrt_tree.nodes(near_idx).position);
                total_cost = rrt_tree.nodes(near_idx).cost + edge_cost;
                if total_cost < min_cost
                    min_cost = total_cost;
                    best_parent = near_idx;
                end
            end

            fprintf("\t RRT* best_parent : %f %f %f \n", rrt_tree.nodes(best_parent).position);
            
            % Add new node
            new_idx = length(rrt_tree.nodes) + 1;
            rrt_tree.nodes(new_idx).position = new_pos;
            rrt_tree.nodes(new_idx).parent = best_parent;
            rrt_tree.nodes(new_idx).children = [];
            rrt_tree.nodes(new_idx).cost = min_cost;
            rrt_tree.nodes(best_parent).children(end+1) = new_idx;
            
            % Rewire tree (RRT* optimization)
            for near_idx = near_indices
                if near_idx ~= best_parent
                    edge_cost = norm(rrt_tree.nodes(near_idx).position - new_pos);
                    new_cost = min_cost + edge_cost;
                    if new_cost < rrt_tree.nodes(near_idx).cost
                        % Rewire
                        old_parent = rrt_tree.nodes(near_idx).parent;
                        rrt_tree.nodes(old_parent).children = ...
                            rrt_tree.nodes(old_parent).children(rrt_tree.nodes(old_parent).children ~= near_idx);
                        
                        rrt_tree.nodes(near_idx).parent = new_idx;
                        rrt_tree.nodes(near_idx).cost = new_cost;
                        rrt_tree.nodes(new_idx).children(end+1) = near_idx;
                        
                        % Update costs of descendants
                        update_descendants_cost(rrt_tree, near_idx);
                    end
                end
            end
        else
            fprintf("New pos: %f %f %f", new_pos);
            disp("INVALID");
        end
    end
end

function pos = sample_random_position(s)
    % Sample random position within bounds
    pos = [
        s.x_min + rand * (s.x_max - s.x_min);
        s.y_min + rand * (s.y_max - s.y_min);
        s.z_min + rand * (s.OceanDepth - s.z_min)
    ]';
end

function pos = sample_biased_position(s)
    % Sample position biased toward unexplored regions
    % This is a simplified version - could be enhanced with actual coverage maps
    
    % Sample with higher probability in outer regions
    radius = s.x_max * (0.5 + 0.5 * rand);
    theta = 2 * pi * rand;
    phi = pi * rand;
    
    pos = [
        radius * sin(phi) * cos(theta);
        radius * sin(phi) * sin(theta);
        s.z_min + rand * (s.z_max - s.z_min)
    ]';
    
    % Clamp to bounds
    pos(1) = max(s.x_min, min(s.x_max, pos(1)));
    pos(2) = max(s.y_min, min(s.y_max, pos(2)));
end

function [nearest_idx, nearest_node] = find_nearest_node(rrt_tree, point)
    % Find nearest node in RRT tree
    min_dist = inf;
    nearest_idx = 1;
    
    for i = 1:length(rrt_tree.nodes)
        dist = norm(rrt_tree.nodes(i).position - point);
        if dist < min_dist
            min_dist = dist;
            nearest_idx = i;
        end
    end
    
    nearest_node = rrt_tree.nodes(nearest_idx);
end

function near_indices = find_near_nodes(rrt_tree, point, radius)
    % Find all nodes within radius of point
    near_indices = [];
    
    for i = 1:length(rrt_tree.nodes)
        if norm(rrt_tree.nodes(i).position - point) <= radius
            near_indices(end+1) = i;
        end
    end
end

function new_pos = extend_toward_point(from_pos, to_pos, step_size, s)
    % Extend from one point toward another with step size limit

    direction = to_pos(1:2) - from_pos(1:2);
    distance = norm(direction);
    
    if distance <= step_size
        new_pos = to_pos;
    else
        direction = direction / distance; % Normalize
        new_pos = [(from_pos(1:2) + step_size * direction) to_pos(3)];
    end
    
    % Ensure within bounds
    new_pos(1) = max(s.x_min, min(s.x_max, new_pos(1)));
    new_pos(2) = max(s.y_min, min(s.y_max, new_pos(2)));
    new_pos(3) = max(s.z_min, min(s.z_max, new_pos(3)));
    
    if sqrt(new_pos(1)^2 + new_pos(2)^2) > s.x_max 
        v = [new_pos(1)-s.sim_source_x, new_pos(2)-s.sim_source_y];
        v_unit = v / norm(v);
        q = [s.sim_source_x, s.sim_source_y] + s.sim_range * v_unit;

        new_pos(1) = q(1);
        new_pos(2) = q(2);

        fprintf('Closest point on the circle: (%.2f, %.2f)\n', new_pos(1), new_pos(2));
    end
end


function update_descendants_cost(rrt_tree, node_idx)
    % Recursively update costs of all descendants after rewiring
    node = rrt_tree.nodes(node_idx);
    
    for child_idx = node.children
        edge_cost = norm(rrt_tree.nodes(child_idx).position - node.position);
        rrt_tree.nodes(child_idx).cost = node.cost + edge_cost;
        update_descendants_cost(rrt_tree, child_idx);
    end
end