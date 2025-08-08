function best_action = getBestActionSequence(tree)
    % Extract all leaf data
    leafData = extractLeafData(tree);
    
    % Number of leaves
    numLeaves = numel(leafData.Sigma);
    
    % Preallocate trace values array
    V = zeros(1, numLeaves);
    
    % Compute the cost associated with each leaf
    for ii = 1:numLeaves
        V(ii) = sum(diag((leafData.Sigma{ii})));
    end
    
    % Find index of leaf with minimum trace (Sigma)
    [~, idxMin] = min(V);
    
    % Retrieve corresponding action sequence
    best_action = leafData.actions{idxMin};
end
