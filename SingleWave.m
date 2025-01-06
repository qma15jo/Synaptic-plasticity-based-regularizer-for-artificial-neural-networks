function SingleWave()
% Kindly follow the latest Release... This code was developed from scratch
% to illustrate the Single Wave Optimality Principle with a randomly generated
% dataset. This code is intended solely to
% facilitate the reproducibility of the method.
% 
% If you utilize this code, please cite the following paper:
% 
% Qais Yousef, Pu Li. Synaptic plasticity-based regularizer for artificial 
% neural networks, 29 March 2024, PREPRINT (Version 1) available at 
% Research Square [https://doi.org/10.21203/rs.3.rs-4114689/v1]
% 
% Additionally, please note that a comprehensive, tested package will be
% released soon.
% 
% Qais Yousef 
% 06.01.2025

    % Parameters
    m = 5; % Number of rows
    n = 5; % Number of columns
    c = 0.5; % Discount factor, 0 < c < 1
    maxGenerations = 100;
    populationSize = 50;
    
    % Initialize matrix M with random values
    M = rand(m, n);
    
    % Define uncertainty function
    p = @(x) abs(x - 0.5); % Example uncertainty function
    
    % Store convergence information
    convergence = zeros(maxGenerations, 1);

    % Evolutionary Algorithm
    for j = 2:n
        % Initialize population for column j
        P = rand(populationSize, m);
        
        for gen = 1:maxGenerations
            % Evaluate fitness
            fitness = zeros(populationSize, 1);
            for k = 1:populationSize
                fitness(k) = sum(p(c * P(k, :)' + (1 - c) * M(:, j-1)));
            end
            
            % Record the best fitness for convergence plot
            if j == 2
                convergence(gen) = min(fitness);
            else
                convergence(gen) = convergence(gen) + min(fitness);
            end
            
            % Selection
            [~, idx] = sort(fitness);
            selectedP = P(idx(1:populationSize/2), :);
            
            % Crossover and Mutation
            newP = selectedP;
            for k = 1:(populationSize/2)
                % Crossover
                if rand < 0.8 % Crossover probability
                    parent1 = selectedP(randi([1, populationSize/2]), :);
                    parent2 = selectedP(randi([1, populationSize/2]), :);
                    crossPoint = randi([1, m]);
                    newP(end+k, :) = [parent1(1:crossPoint), parent2(crossPoint+1:end)];
                end
                % Mutation
                if rand < 0.2 % Mutation probability
                    newP(end+k, randi([1, m])) = rand;
                end
            end
            P = [selectedP; newP((populationSize/2)+1:end, :)];
        end
        
        % Update matrix M with optimal solutions
        M(:, j) = P(1, :)';
    end
    
    % Calculate overall uncertainty
    totalUncertainty = 0;
    for j = 2:n
        for i = 1:m
            totalUncertainty = totalUncertainty + p(c * M(i, j) + (1 - c) * M(i, j-1));
        end
    end
    
    % Display results
    disp('Optimized Matrix M:');
    disp(M);
    disp('Total Uncertainty:');
    disp(totalUncertainty);
    
    % Plot convergence curve
    figure;
    plot(1:maxGenerations, convergence, 'LineWidth', 2);
    title('Convergence Curve');
    xlabel('Generation');
    ylabel('Total Uncertainty');
    grid on;
end
