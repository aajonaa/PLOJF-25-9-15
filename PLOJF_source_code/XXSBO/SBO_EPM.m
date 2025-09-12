function [best_pos, Convergence_curve] = SBO_EPM(N, MaxFEs, lb, ub, dim, fobj)
% -----------------------------------------------------------------------------------------
% This is a modified version of Status-based Optimization (SBO) integrated with
% key strategies from Evolutionary Population Management (EPM).
%
% EPM Strategy 1: Separated Parent-Child Populations.
% EPM Strategy 2: Fitness-based and FDB Mating for Survivor Selection.
%
% Original SBO Author: Jian Wang et al. (2025)
% EPM Inspiration from: Üstünsoy et al. (2025) 
% Integration by: Gemini AI
% -----------------------------------------------------------------------------------------

    %% INITIALIZATION
    FEs = 0;
    bestFitness = inf; 
    best_pos = zeros(1, dim);
    Convergence_curve = [];
    iter = 1;

    % In EPM, the 'localElite' acts as the PARENT population
    parent_X = initialization(N, dim, ub, lb);
    parent_Fitness = inf * ones(N, 1);
    
    % Initialize fitness for the parent population and find the initial global best
    for i = 1:N
        parent_Fitness(i, 1) = fobj(parent_X(i, :));
        FEs = FEs + 1;
        
        if parent_Fitness(i, 1) < bestFitness
            bestFitness = parent_Fitness(i, 1);
            best_pos = parent_X(i, :);
        end
    end
    
    % Initialize local elite population (same as parent initially)
    localElite_X = parent_X;
    localElite_Fitness = parent_Fitness;
    
    % Social success flag for tracking individual performance
    flag = ones(N, 1);
    
    %% MAIN LOOP (Each loop is one EPM Epoch)
    while FEs < MaxFEs
        
        % Sort the local elite fitness for Roulette Wheel Selection
        [sorted_localElite_Fitness, idx] = sort(localElite_Fitness);
        
        % --- EPM STRATEGY 1: GENERATE A SEPARATE CHILD POPULATION ---
        child_X = zeros(N, dim);
        child_Fitness = inf * ones(N, 1);
        social_X = zeros(N, dim);
        social_Fitness = inf * ones(N, 1);
        
        %% Generate child population using SBO's update rule
        for i = 1:N
            % Select individual from local elite population for each child
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
        
        %% Apply SBO's Upward Social Strategy to children
        social_X = child_X;
        
        % Apply one-dimension exchange for successful individuals
        for i = 1:N
            if flag(i) == 1
                social_X1_val = localElite_X(i, randi(dim));
                social_X2_val = best_pos(randi(dim));
                social_X(i, randi(dim)) = (social_X1_val + social_X2_val) / 2;
            end
        end
        
        % Apply multi-dimension exchange for failed individuals
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
        
        %% Greedy selection between child and social positions
        for i = 1:N
            if FEs < MaxFEs
                child_Fitness(i, 1) = fobj(child_X(i, :));
                FEs = FEs + 1;
            end
            if FEs < MaxFEs
                social_Fitness(i, 1) = fobj(social_X(i, :));
                FEs = FEs + 1;
            end
            
            % Keep the better solution and update flag
            if social_Fitness(i, 1) < child_Fitness(i, 1)
                child_X(i, :) = social_X(i, :);
                child_Fitness(i, 1) = social_Fitness(i, 1);
                flag(i, 1) = 1; % Social success
            else
                flag(i, 1) = 0; % Social fail
            end
        end
        
        %% Update local elite population
        for i = 1:N
            if child_Fitness(i, 1) < localElite_Fitness(i, 1)
                localElite_Fitness(i, 1) = child_Fitness(i, 1);
                localElite_X(i, :) = child_X(i, :);
            end
        end
        
        % --- EPM STRATEGY 2: SURVIVOR SELECTION AT END OF EPOCH ---
        % Merge parent and child populations
        unified_X = [parent_X; child_X];
        unified_Fitness = [parent_Fitness; child_Fitness];
        
        % Update global best from the entire pool
        [min_unified_Fitness, min_idx] = min(unified_Fitness);
        if min_unified_Fitness < bestFitness
            bestFitness = min_unified_Fitness;
            best_pos = unified_X(min_idx, :);
        end
        
        % Select survivors using EPM's production rule
        survivor_indices = pop_productions(unified_X, unified_Fitness, N);
        
        % Form new parent population for next epoch
        parent_X = unified_X(survivor_indices, :);
        parent_Fitness = unified_Fitness(survivor_indices, :);
        
        % Update local elite to match new parents
        localElite_X = parent_X;
        localElite_Fitness = parent_Fitness;
    
        %% Record convergence
        Convergence_curve(iter) = bestFitness;
        fprintf('SBO-EPM Iteration %d, FEs %d, Best Fitness = %e\n', iter, FEs, bestFitness);
        iter = iter + 1;
    end
end  

%% Helper Functions (Original SBO + EPM)

% Initialize a population
function X = initialization(N, dim, up, low)
    if isscalar(low)
        low = repmat(low, 1, dim);
    end
    if isscalar(up)
        up = repmat(up, 1, dim);
    end
    X = rand(N, dim) .* (up - low) + low;
end

% Enforce boundary constraints
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
    
% Roulette wheel selection
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

% --- EPM SURVIVOR SELECTION FUNCTIONS (Adapted from provided EPM-PSO code) ---
function [idx_newpop] = pop_productions(Bpop, Fpop, N)
    % This function selects N survivors from a 2N population pool.
    % It uses fitness-based selection for the first N/2 and FDB for the second N/2. 
    
    [total_pop_size, ~] = size(Bpop);
    [~, indexes] = sort(Fpop);
    forbidden_list = false(1, total_pop_size);
    idx_newpop = zeros(1, N);
    
    % Step 1: Fitness-based selection for the first half of the new population
    sayac = 0;
    for i = 1:total_pop_size        
        if ~forbidden_list(indexes(i))
            sayac = sayac + 1;
            idx_newpop(sayac) = indexes(i);
            
            % Forbid this individual and its parent/child pair from being selected again
            forbidden_list(indexes(i)) = true;
            if indexes(i) > N % It's a child, forbid its parent
               if indexes(i) - N > 0
                   forbidden_list(indexes(i) - N) = true;
               end
            else  % It's a parent, forbid its child
               if indexes(i) + N <= total_pop_size
                   forbidden_list(indexes(i) + N) = true;
               end
            end
        end
        if sayac >= floor(N/2), break; end
    end
    
    % Step 2: FDB-based mating for the remaining positions
    for i = 1:sayac
        if sayac >= N, break; end
        
        % For each of the fitness-selected members, find its best mate from the pool
        fdb_indexes = fitnessDistanceBalanceIndexes(Bpop, Fpop, Bpop(idx_newpop(i),:));
        
        for j = 1:total_pop_size
            if ~forbidden_list(fdb_indexes(j))
                sayac = sayac + 1;
                idx_newpop(sayac) = fdb_indexes(j);
                
                % Forbid this individual and its pair
                forbidden_list(fdb_indexes(j)) = true;
                if fdb_indexes(j) > N
                   if fdb_indexes(j) - N > 0
                       forbidden_list(fdb_indexes(j) - N) = true;
                   end
                else
                   if fdb_indexes(j) + N <= total_pop_size
                       forbidden_list(fdb_indexes(j) + N) = true;
                   end
                end
                break; % Move to the next member once a mate is found
            end
        end
        if sayac >= N, break; end
    end
    
    % Fill remaining positions with best available individuals if needed
    if sayac < N
        for i = 1:total_pop_size
            if ~forbidden_list(indexes(i)) && sayac < N
                sayac = sayac + 1;
                idx_newpop(sayac) = indexes(i);
            end
        end
    end
end

function fdb_indexes = fitnessDistanceBalanceIndexes(population, fitness, partner)
    % Calculates the Fitness-Distance Balance to find suitable mates. 
    [populationSize, ~] = size(population);
    distances = zeros(1, populationSize); 

    if min(fitness) == max(fitness)
        fdb_indexes = randperm(populationSize);
    else
        for i = 1:populationSize
            % Manhattan distance
            distances(i) = sum(abs(partner - population(i, :)));
        end

        minFitness = min(fitness); 
        maxMinFitness = max(fitness) - minFitness;
        minDistance = min(distances); 
        maxMinDistance = max(distances) - minDistance;

        % Avoid division by zero
        if maxMinFitness == 0
            normFitness = zeros(size(fitness));
        else
            normFitness = 1 - ((fitness - minFitness) / maxMinFitness);
        end
        
        if maxMinDistance == 0
            normDistances = zeros(size(distances));
        else
            normDistances = (distances - minDistance) / maxMinDistance;
        end
        
        % FDB score is the sum of normalized fitness and distance
        divDistances = normFitness(:) + normDistances(:);
        
        [~, fdb_indexes] = sort(divDistances, 'descend'); 
    end
end