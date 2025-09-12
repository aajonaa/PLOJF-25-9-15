function [best_pos, Convergence_curve] = SBO_EPM_Strategy2(N, MaxFEs, lb, ub, dim, fobj)
% -----------------------------------------------------------------------------------------
% SBO with EPM Strategy 2 ONLY: Fitness-Distance Balance (FDB) Survivor Selection
% This version keeps the original SBO structure but replaces the simple survivor
% selection with EPM's FDB-based selection strategy.
% -----------------------------------------------------------------------------------------

    %% INITIALIZATION (Same as original SBO)
    FEs = 0;
    bestFitness = inf; 
    best_pos = zeros(1, dim);
    Convergence_curve = [];
    iter = 1;

    current_X = initialization(N, dim, ub, lb);
    localElite_X = initialization(N, dim, ub, lb);
        
    current_Fitness = inf * ones(N, 1);
    localElite_Fitness = inf * ones(N, 1);
    social_Fitness = inf * ones(N, 1);
    
    % Social success flag
    flag = ones(N, 1);
    
    for i = 1:N
        current_Fitness(i, 1) = fobj(current_X(i, :));
        FEs = FEs + 1;
        fitness = fobj(localElite_X(i, :));
        FEs = FEs + 1;
        if current_Fitness(i, 1) < fitness
            localElite_X(i, :) = current_X(i, :);
            localElite_Fitness(i, 1) = current_Fitness(i, 1);
        else
            localElite_Fitness(i, 1) = fitness;
        end
        
        if current_Fitness(i, 1) < bestFitness
            bestFitness = current_Fitness(i, 1);
            best_pos = current_X(i, :);
        end
    end
    
    %% MAIN LOOP (Original SBO with FDB selection)
    while FEs < MaxFEs
        
        % Sort the local elite fitness for Roulette Wheel Selection
        [sorted_localElite_Fitness, idx] = sort(localElite_Fitness);
        
        %% Standard SBO update (no parent-child separation)
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
                    current_X(i, j) = (1 - w1 - w2) * current_X(i, j) + w1 * localElite_X(Roulette_index, j) + w2 * best_pos(j);
                end
            else
                for j = 1:dim
                    current_X(i, j) = w4 * ((1 - w1 - w2) * current_X(i, j) + w1 * localElite_X(Roulette_index, j) + w2 * best_pos(j));
                end
            end
        end
        
        current_X = BoundaryControl(current_X, lb, ub);
        
        %% Apply SBO's social strategy
        social_X = current_X;
        
        for i = 1:N
            if flag(i) == 1
                social_X1_val = localElite_X(i, randi(dim));
                social_X2_val = best_pos(randi(dim));
                social_X(i, randi(dim)) = (social_X1_val + social_X2_val) / 2;
            end
        end
        
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
        
        %% Greedy selection
        for i = 1:N
            current_Fitness(i, 1) = fobj(current_X(i, :));
            FEs = FEs + 1;
            social_Fitness(i, 1) = fobj(social_X(i, :));
            FEs = FEs + 1;
            
            if social_Fitness(i, 1) < current_Fitness(i, 1)
                flag(i, 1) = 1; % Social success
                current_X(i, :) = social_X(i, :);
                current_Fitness(i, 1) = social_Fitness(i, 1);
            else
                flag(i, 1) = 0; % Social fail
            end
        end
        
        %% Update local elite population
        for i = 1:N
            if current_Fitness(i, 1) < localElite_Fitness(i, 1)
                localElite_Fitness(i, 1) = current_Fitness(i, 1);
                localElite_X(i, :) = current_X(i, :);
            end
        end
        
        % --- EPM STRATEGY 2: FDB-based survivor selection ---
        % Create a doubled population by duplicating current solutions
        % (This simulates having parent and child populations for FDB selection)
        doubled_X = [current_X; current_X];
        doubled_Fitness = [current_Fitness; current_Fitness];
        
        % Apply FDB-based survivor selection
        survivor_indices = fdb_survivor_selection(doubled_X, doubled_Fitness, N);
        
        % Update current population
        current_X = doubled_X(survivor_indices, :);
        current_Fitness = doubled_Fitness(survivor_indices, :);
        
        % Update global best
        [min_fitness, min_idx] = min(current_Fitness);
        if min_fitness < bestFitness
            bestFitness = min_fitness;
            best_pos = current_X(min_idx, :);
        end
        
        %% Record convergence
        Convergence_curve(iter) = bestFitness;
        fprintf('SBO-EPM-S2 Iteration %d, FEs %d, Best Fitness = %e\n', iter, FEs, bestFitness);
        iter = iter + 1;
    end
end

%% EPM Strategy 2: FDB-based survivor selection
function [idx_newpop] = fdb_survivor_selection(Bpop, Fpop, N)
    [total_pop_size, ~] = size(Bpop);
    [~, indexes] = sort(Fpop);
    forbidden_list = false(1, total_pop_size);
    idx_newpop = zeros(1, N);
    
    % Step 1: Fitness-based selection for the first half
    sayac = 0;
    for i = 1:total_pop_size        
        if ~forbidden_list(indexes(i)) && sayac < floor(N/2)
            sayac = sayac + 1;
            idx_newpop(sayac) = indexes(i);
            forbidden_list(indexes(i)) = true;
        end
        if sayac >= floor(N/2), break; end
    end
    
    % Step 2: FDB-based selection for the remaining positions
    for i = 1:sayac
        if sayac >= N, break; end
        
        fdb_indexes = fitnessDistanceBalanceIndexes(Bpop, Fpop, Bpop(idx_newpop(i),:));
        
        for j = 1:total_pop_size
            if ~forbidden_list(fdb_indexes(j)) && sayac < N
                sayac = sayac + 1;
                idx_newpop(sayac) = fdb_indexes(j);
                forbidden_list(fdb_indexes(j)) = true;
                break;
            end
        end
    end
    
    % Fill remaining positions if needed
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
    [populationSize, ~] = size(population);
    distances = zeros(1, populationSize); 

    if min(fitness) == max(fitness)
        fdb_indexes = randperm(populationSize);
    else
        for i = 1:populationSize
            distances(i) = sum(abs(partner - population(i, :)));
        end

        minFitness = min(fitness); 
        maxMinFitness = max(fitness) - minFitness;
        minDistance = min(distances); 
        maxMinDistance = max(distances) - minDistance;

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
        
        divDistances = normFitness(:) + normDistances(:);
        [~, fdb_indexes] = sort(divDistances, 'descend'); 
    end
end

%% Helper Functions (same as original SBO)
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