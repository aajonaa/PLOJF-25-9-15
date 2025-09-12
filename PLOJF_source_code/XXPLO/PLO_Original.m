function [best_pos, Convergence_curve] = PLO_Original(N, MaxFEs, lb, ub, dim, fobj)
% -----------------------------------------------------------------------------------------
% Original PLO Algorithm (cleaned version)
% Based on PLOmod2.m with fixes and improvements
% -----------------------------------------------------------------------------------------

%% Initialization
FEs = 0;
it = 1;
AllFitness = inf * ones(N, 1);
newFitness = inf * ones(N, 1);

X = initialization(N, dim, ub, lb);
V = ones(N, dim);
newX = zeros(N, dim);

for i = 1:N
    AllFitness(i) = fobj(X(i, :));
    FEs = FEs + 1;
end

[AllFitness, SortOrder] = sort(AllFitness);
X = X(SortOrder, :);
Best_pos = X(1, :);
bestFitness = AllFitness(1);

Convergence_curve = [];
Convergence_curve(it) = bestFitness;

phi_qKR = 0.25 + 0.55 * ((0 + ((1:N)/N)) .^ 0.5);

%% Main loop
while FEs < MaxFEs
    
    X_sum = sum(X, 1);
    X_mean = X_sum / N;
    w1 = tansig((FEs/MaxFEs)^4);
    w2 = exp(-(2*FEs/MaxFEs)^3);

    for i = 1:N
        a = rand()/2 + 1;
        V(i, :) = 1 * exp((1-a)/100 * FEs);
        LS = V(i, :);

        GS = Levy(dim) .* (X_mean - X(i, :) + (lb + rand(1, dim) * (ub - lb))/2);
        
        % Migration behavior
        r = rand();
        ori_value = rand(1, dim);
        if rand < 0.05
            newX(i, :) = X(i, :) + (X(i, :) - Best_pos) .* tan((ori_value - 0.5) * pi);
        else
            newX(i, :) = X(i, :) + (w1 * LS + w2 * GS) .* ori_value;
        end
    end
    
    E = sqrt(FEs/MaxFEs);
    A = randperm(N);
    [~, ind] = sort(AllFitness);
    bestInd = ind(1);
    
    lamda_t = 0.1 + (0.518 * ((1-(FEs/MaxFEs)^0.5)));
    
    for i = 1:N
        eta_qKR_i = (round(rand * phi_qKR(i)) + (rand <= phi_qKR(i)))/2;
        omega_it = rand();
        
        while true, r1 = randi(N); if r1 ~= i && r1 ~= bestInd, break, end, end
        while true, r2 = randi(N); if r2 ~= i && r2 ~= bestInd && r2 ~= r1, break, end, end
        
        for j = 1:dim
            if rand <= eta_qKR_i
                newX(i, j) = Best_pos(j) + ((X(r2, j) - X(i, j)) * lamda_t) + ((X(r1, j) - X(i, j)) * omega_it);
            else
                newX(i, j) = X(i, j) + sin(rand * pi) * (X(i, j) - X(A(i), j));
            end
        end

        Flag4ub = newX(i, :) > ub;
        Flag4lb = newX(i, :) < lb;
        newX(i, :) = (newX(i, :) .* (~(Flag4ub + Flag4lb))) + ub .* Flag4ub + lb .* Flag4lb;
        
        if FEs < MaxFEs
            newFitness(i) = fobj(newX(i, :));
            FEs = FEs + 1;
        else
            break;
        end

        if newFitness(i) < AllFitness(i)
            X(i, :) = newX(i, :);
            AllFitness(i) = newFitness(i);
        end
    end

    [AllFitness, SortOrder] = sort(AllFitness);
    X = X(SortOrder, :);
    if AllFitness(1) < bestFitness
        Best_pos = X(1, :);
        bestFitness = AllFitness(1);
    end

    it = it + 1;
    Convergence_curve(it) = bestFitness;
    best_pos = Best_pos;
    fprintf('PLO-Original Iteration %d, FEs %d, Best Fitness = %e\n', it-1, FEs, bestFitness);
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