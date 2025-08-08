function leafData = extractLeafData(tree)

    % Initialize leafData only at the top-level call
    leafData.actions = {};
    leafData.x = [];
    leafData.y = [];
    leafData.z = [];
    leafData.mu = {};
    leafData.Sigma = {};
    
    % Call recursive extraction
    leafData = recursiveExtract(tree, leafData);

end

% Recursive helper function
function leafData = recursiveExtract(node, leafData)
    if isfield(node, 'branch')
        % Internal node: recurse through children
        for k = 1:numel(node.branch)
            leafData = recursiveExtract(node.branch(k), leafData);
        end
    else
        % Leaf node: store data
        leafData.actions{end+1} = node.actions;
        leafData.x(end+1) = node.x;
        leafData.y(end+1) = node.y;
        leafData.z(end+1) = node.z;
        leafData.mu{end+1} = node.mu;
        leafData.Sigma{end+1} = node.Sigma;
    end
end
