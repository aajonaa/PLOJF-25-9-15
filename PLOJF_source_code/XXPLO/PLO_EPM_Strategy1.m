function [best_pos, Convergence_curve] = PLO_EPM_Strategy1(N, MaxFEs, lb, ub, dim, fobj)
% -----------------------------------------------------------------------------------------
% PLO with EPM Strategy 1: Separated Parent-Child Populations
% This version implements the parent-child separation strategy from EPM
% while keeping all PLO mechanisms intact.
% -----------------------------------------------------------------------------------------

%% Initialization
FEs = 0;
it = 1;
bestFitness = inf;
best_pos = zeros(1, dim);
Convergence_curve = [];

% Initialize parent population
parent_X = initialization(N, dim, ub, lb);
parent_Fitness = inf * ones(N, 1);
parent_V = ones(N, dim);

% Evaluate parent population
for i = 1:N
    parent_Fitness(i) = fobj(parent_X(i, :));
    FEs = FEs + 1;
end

% Sort and find initial best
[parent_Fitness, SortOrder] = sort(parent_Fitness);
parent_X = parent_X(SortOrder, :);
parent_V = parent_V(SortOrder, :);
best_pos = parent_X(1, :);
bestFitness = parent_Fitness(1);

Convergence_curve(it) = bestFitness;

% PLO-specific parameters
phi_qKR = 0.25 + 0.55 * ((0 + ((1:N)/N)) .^ 0.5);

%% Main loop
while FEs < MaxFEs
    
    % PLO calculations
    X_sum = sum(parent_X, 1);
    X_mean = X_sum / N;
    w1 = tansig((FEs/MaxFEs)^4);
    w2 = exp(-(2*FEs/MaxFEs)^3);
    
    % --- EPM STRATEGY 1: Generate separate child population ---
    child_X = zeros(N, dim);
    child_Fitness = inf * ones(N, 1);
    child_V = ones(N, dim);
    
    %% Generate child population using PLO's update rules
    for i = 1:N
        % PLO's local search component
        a = rand()/2 + 1;
        child_V(i, :) = 1 * exp((1-a)/100 * FEs);
        LS = child_V(i, :);
        
        % PLO's global search component
        GS = Levy(dim) .* (X_mean - parent_X(i, :) + (lb + rand(1, dim) * (ub - lb))/2);
        
        % Migration behavior
        ori_value = rand(1, dim);
        cauchy_value = tan((ori_value - 0.5) * pi);
        if rand < 0.05
            child_X(i, :) = parent_X(i, :) + (parent_X(i, :) - best_pos) .* cauchy_value;
        else
            child_X(i, :) = parent_X(i, :) + (w1 * LS + w2 * GS) .* ori_value;
        end
    end
    
    %% Apply PLO's migration behavior to children
    E = sqrt(FEs/MaxFEs);
    A = randperm(N);
    [~, ind] = sort(parent_Fitness);
    bestInd = ind(1);
    
    lamda_t = 0.1 + (0.518 * ((1-(FEs/MaxFEs)^0.5)));
    
    for i = 1:N
        eta_qKR_i = (round(rand * phi_qKR(i)) + (rand <= phi_qKR(i)))/2;
        omega_it = rand();
        
        % Select random indices
        while true, r1 = randi(N); if r1 ~= i && r1 ~= bestInd, break, end, end
        while true, r2 = randi(N); if r2 ~= i && r2 ~= bestInd && r2 ~= r1, break, end, end
        
        for j = 1:dim
            if rand <= eta_qKR_i
                child_X(i, j) = best_pos(j) + ((parent_X(r2, j) - parent_X(i, j)) * lamda_t) + ((parent_X(r1, j) - parent_X(i, j)) * omega_it);
            else
                child_X(i, j) = parent_X(i, j) + sin(rand * pi) * (parent_X(i, j) - parent_X(A(i), j));
            end
        end
        
        % Boundary control
        Flag4ub = child_X(i, :) > ub;
        Flag4lb = child_X(i, :) < lb;
        child_X(i, :) = (child_X(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        % Evaluate child
        if FEs < MaxFEs
            child_Fitness(i) = fobj(child_X(i, :));
            FEs = FEs + 1;
        end
    end
    
    %% EPM Strategy 1: Survivor selection from parent+child populations
    unified_X = [parent_X; child_X];
    unified_Fitness = [parent_Fitness; child_Fitness];
    unified_V = [parent_V; child_V];
    
    % Sort and select best N individuals
    [sorted_unified_Fitness, sort_idx] = sort(unified_Fitness);
    survivor_indices = sort_idx(1:N);
    
    % Update parent population for next iteration
    parent_X = unified_X(survivor_indices, :);
    parent_Fitness = unified_Fitness(survivor_indices, :);
    parent_V = unified_V(survivor_indices, :);
    
    % Update global best
    if sorted_unified_Fitness(1) < bestFitness
        bestFitness = sorted_unified_Fitness(1);
        best_pos = unified_X(survivor_indices(1), :);
    end
    
    %% Record convergence
    it = it + 1;
    Convergence_curve(it) = bestFitness;
    fprintf('PLO-EPM-S1 Iteration %d, FEs %d, Best Fitness = %e\n', it-1, FEs, bestFitness);
end

end

%% Helper Functions
function X = initialization(N, dim, ub, lb)
    if isscalar(lb)
        lb = repmat(lb, 1, dim);
    end
    if isscalar(ub)
        ub = repmat(ub, 1, dim);
    end
    X = rand(N, dim) .* (ub - lb) + lb;
end

function o = Levy(d)
    beta = 1.5;
    sigma = (gamma(1+beta)*sin(pi*beta/2)/(gamma((1+beta)/2)*beta*2^((beta-1)/2)))^(1/beta);
    u = randn(1, d) * sigma;
    v = randn(1, d);
    step = u ./ abs(v).^(1/beta);
    o = step;
end