function [best_pos, Convergence_curve] = PLO_Simple(N, MaxFEs, lb, ub, dim, fobj)
% Simplified PLO for testing - guaranteed to terminate
fprintf('Starting PLO_Simple with MaxFEs = %d\n', MaxFEs);

%% Initialization
FEs = 0;
X = lb + (ub - lb) .* rand(N, dim);
Fitness = inf * ones(N, 1);

% Initial evaluation
for i = 1:N
    if FEs >= MaxFEs, break; end
    Fitness(i) = fobj(X(i, :));
    FEs = FEs + 1;
end

[Fitness, idx] = sort(Fitness);
X = X(idx, :);
best_pos = X(1, :);
bestFitness = Fitness(1);

Convergence_curve = bestFitness;
iter = 1;

%% Main loop with strict FEs control
while FEs < MaxFEs
    fprintf('Iteration %d, FEs = %d/%d\n', iter, FEs, MaxFEs);
    
    % Simple PLO-like update
    for i = 1:N
        if FEs >= MaxFEs, break; end
        
        % Simple movement
        newX = X(i, :) + 0.1 * randn(1, dim);
        
        % Boundary control
        newX = max(newX, lb);
        newX = min(newX, ub);
        
        % Evaluate
        newFitness = fobj(newX);
        FEs = FEs + 1;
        
        % Update if better
        if newFitness < Fitness(i)
            X(i, :) = newX;
            Fitness(i) = newFitness;
        end
        
        if FEs >= MaxFEs, break; end
    end
    
    % Sort and update best
    [Fitness, idx] = sort(Fitness);
    X = X(idx, :);
    if Fitness(1) < bestFitness
        bestFitness = Fitness(1);
        best_pos = X(1, :);
    end
    
    iter = iter + 1;
    Convergence_curve(iter) = bestFitness;
    
    if FEs >= MaxFEs, break; end
end

fprintf('PLO_Simple completed with %d FEs\n', FEs);
end