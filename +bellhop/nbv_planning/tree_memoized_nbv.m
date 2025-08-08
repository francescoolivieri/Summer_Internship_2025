function data = tree_memoized_nbv(data, s, idx)
    % Tree approach 
    tree = generateTree(s.depth);
    position_cache = containers.Map('KeyType', 'char', 'ValueType', 'any');
    
    tic
    tree = traverseTreeMemoized(tree, data.th_est(:, idx), data.Sigma_est(:, :, idx), ...
        data.x(idx), data.y(idx), data.z(idx), s, position_cache);
    toc
    
    best_action = getBestActionSequence(tree);
    [data.x(idx+1), data.y(idx+1), data.z(idx+1)] = update_pos(data.x(idx), data.y(idx), data.z(idx), s, best_action(1));
end