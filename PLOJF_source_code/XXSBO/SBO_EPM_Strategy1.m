function [best_pos, Convergence_curve] = SBO_EPM_Strategy1(N, MaxFEs, lb, ub, dim, fobj)
% -----------------------------------------------------------------------------------------
% SBO with EPM Strategy 1 ONLY: Separated Parent-Child Populations
% This version only implements the parent-child separation strategy from EPM
% while keeping all other SBO mechanisms intact.
% -----------------------------------------------------------------------------------------

    %% INITIALIZATION
    FEs = 0;
    bestFitness = inf; 
    best_pos = zeros(1, dim);
    Convergence_curve = [];
    iter = 1;

    % Initialize parent population (equivalent to SBO's current_X)
    parent_X = initialization(N, dim, ub, lb);
    parent_Fitness = inf * ones(N, 1);
    
    % Initialize local elite population
    localElite_X = initialization(N, dim, ub, lb);
    localElite_Fitness = inf * ones(N, 1);
    
    % Social success flag
    flag = ones(N, 1);
    
    % Initialize fitness for both populations
    for i = 1:N
        parent_Fitness(i, 1) = fobj(parent_X(i, :));
        FEs = FEs + 1;
        fitness = fobj(localElite_X(i, :));
        FEs = FEs + 1;
        
        if parent_Fitness(i, 1) < fitness
            localElite_X(i, :) = parent_X(i, :);
            localElite_Fitness(i, 1) = parent_Fitness(i, 1);
        else
            localElite_Fitness(i, 1) = fitness;
        end
        
        if parent_Fitness(i, 1) < bestFitness
            bestFitness = parent_Fitness(i, 1);
            best_pos = parent_X(i, :);
        end
    end
    
    %% MAIN LOOP
    while FEs < MaxFEs
        
        % Sort the local elite fitness for Roulette Wheel Selection
        [sorted_localElite_Fitness, idx] = sort(localElite_Fitness);
        
        % --- EPM STRATEGY 1: Generate separate child population ---
        child_X = zeros(N, dim);
        child_Fitness = inf * ones(N, 1);
        
        %% Generate child population using SBO's update rule
        for i = 1:N
            % Select individual from local elite population
            Roulette_index = RouletteWheelSelection(1./(sorted_localElite_Fitness + eps));
            if Roulette_index == -1  
                Roulette_index = 1;
            end
            
            w1 = randn;
            w2 = randn;
            w3 = tanh((sqrt(abs(MaxFEs - randn * FEs))/i)^(FEs/MaxFEs));
            w4 = unifrnd(-w3, w3);
            
            if rand < w3
                for j = 1:dim
                    child_X(i, j) = (1 - w1 - w2) * parent_X(i, j) + w1 * localElite_X(Roulette_index, j) + w2 * best_pos(j);
                end
            else
                for j = 1:dim
                    child_X(i, j) = w4 * ((1 - w1 - w2) * parent_X(i, j) + w1 * localElite_X(Roulette_index, j) + w2 * best_pos(j));
                end
            end
        end
        
        child_X = BoundaryControl(child_X, lb, ub);
        
        %% Apply SBO's social strategy to children
        social_X = child_X;
        
        % One-dimension exchange for successful individuals
        for i = 1:N
            if flag(i) == 1
                social_X1_val = localElite_X(i, randi(dim));
                social_X2_val = best_pos(randi(dim));
                social_X(i, randi(dim)) = (social_X1_val + social_X2_val) / 2;
            end
        end
        
        % Multi-dimension exchange for failed individuals
        m = zeros(1, dim);
        u = randperm(dim);
        m(u(1:ceil(rand * dim))) = 1;
        for i = 1:N
            if flag(i) == 0
                for j = 1:dim
                    if m(j)
                        social_X(i, j) = localElite_X(i, j);
                    end
                end
            end
        end
        
        social_X = BoundaryControl(social_X, lb, ub);
        
        %% Evaluate children and apply greedy selection
        for i = 1:N
            if FEs < MaxFEs
                child_Fitness(i, 1) = fobj(child_X(i, :));
                FEs = FEs + 1;
            end
            if FEs < MaxFEs
                social_Fitness = fobj(social_X(i, :));
                FEs = FEs + 1;
                
                % Greedy selection and flag update
                if social_Fitness < child_Fitness(i, 1)
                    child_X(i, :) = social_X(i, :);
                    child_Fitness(i, 1) = social_Fitness;
                    flag(i, 1) = 1; % Social success
                else
                    flag(i, 1) = 0; % Social fail
                end
            end
        end
        
        %% Update local elite population
        for i = 1:N
            if child_Fitness(i, 1) < localElite_Fitness(i, 1)
                localElite_Fitness(i, 1) = child_Fitness(i, 1);
                localElite_X(i, :) = child_X(i, :);
            end
        end
        
        %% Simple survivor selection (best N from parent+child)
        % This is the key difference - we select survivors from parent+child
        unified_X = [parent_X; child_X];
        unified_Fitness = [parent_Fitness; child_Fitness];
        
        % Sort and select best N individuals
        [sorted_unified_Fitness, sort_idx] = sort(unified_Fitness);
        survivor_indices = sort_idx(1:N);
        
        % Update parent population for next iteration
        parent_X = unified_X(survivor_indices, :);
        parent_Fitness = unified_Fitness(survivor_indices, :);
        
        % Update global best
        if sorted_unified_Fitness(1) < bestFitness
            bestFitness = sorted_unified_Fitness(1);
            best_pos = unified_X(survivor_indices(1), :);
        end
        
        %% Record convergence
        Convergence_curve(iter) = bestFitness;
        fprintf('SBO-EPM-S1 Iteration %d, FEs %d, Best Fitness = %e\n', iter, FEs, bestFitness);
        iter = iter + 1;
    end
end

%% Helper Functions
function X = initialization(N, dim, up, low)
    if isscalar(low)
        low = repmat(low, 1, dim);
    end
    if isscalar(up)
        up = repmat(up, 1, dim);
    end
    X = rand(N, dim) .* (up - low) + low;
end

function X = BoundaryControl(X, low, up)
    [N, dim] = size(X);
    if isscalar(low)
        low = repmat(low, 1, dim);
    end
    if isscalar(up)
        up = repmat(up, 1, dim);
    end
    for i = 1:N
        for j = 1:dim                
            k = rand < rand;
            if X(i,j) < low(j) 
                if k, X(i,j) = low(j); else, X(i,j) = rand * (up(j) - low(j)) + low(j); end 
            end        
            if X(i,j) > up(j)  
                if k, X(i,j) = up(j); else, X(i,j) = rand * (up(j) - low(j)) + low(j); end 
            end
        end
    end
end

function choice = RouletteWheelSelection(weights)
    if any(weights<0), weights = weights + min(weights); end
    accumulation = cumsum(weights);
    if accumulation(end) == 0, choice = -1; return; end
    p = rand() * accumulation(end);
    chosen_index = -1;
    for index = 1:length(accumulation)
        if (accumulation(index) > p), chosen_index = index; break; end
    end
    choice = chosen_index;
end