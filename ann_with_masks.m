function ann_with_masks(num_masks, num_hidden_layers, neurons_per_layer)

    % Kindly follow the latest Release... This code was developed from scratch
    % to illustrate the Prospect Certainty method for data-driven models. It
    % features a simple Multi-Layer Perceptron (MLP) with a randomly generated
    % dataset. The final results reflect the model's simplicity and the
    % dataset's lack of coherence. However, this code is intended solely to
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
    % 21.12.2024
    
    % Example Call:
    % num_masks = [3, 3, 3, 3];
    % num_hidden_layers = 3;
    % neurons_per_layer = [4, 3, 3];
    % ann_with_masks(num_masks, num_hidden_layers, neurons_per_layer);
    
    
    
    % Define the number of masks for the hidden and output layers. 0: no masks.
    % The length of the vector must be num_hidden_layers + 1.
    
    % Define input and output sizes
    inputSize = 3;
    outputSize = 2;
    
    % Number of neurons in all the layers
    neurons_per_H_O = [neurons_per_layer, outputSize];
    neurons_per_I_H_O = [inputSize, neurons_per_layer, outputSize];
    
    % Ensure neurons_per_layer is a row vector
    neurons_per_layer = neurons_per_layer(:)';
    
    % Initialize cell arrays to hold weights and biases
    W = cell(1, num_hidden_layers + 1);
    b = cell(1, num_hidden_layers + 1);
    
    % Initialize cell arrays to hold weights and biases
    W = cell(1, num_hidden_layers + 1);
    b = cell(1, num_hidden_layers + 1);
    
    % Input layer to first hidden layer
    W{1} = rand(neurons_per_layer(1), inputSize);
    b{1} = rand(neurons_per_layer(1), 1);
    
    % Hidden layers
    for i = 2:num_hidden_layers
        W{i} = rand(neurons_per_layer(i), neurons_per_layer(i - 1));
        b{i} = rand(neurons_per_layer(i), 1);
    end
    
    % Last hidden layer to output layer
    W{num_hidden_layers + 1} = rand(outputSize, neurons_per_layer(end));
    b{num_hidden_layers + 1} = rand(outputSize, 1);
    
    
    for l = 1 : num_hidden_layers+1
        for i = 1 : neurons_per_H_O(1,l)
            wm=[];
            bm=[];
            for j = 1 : num_masks(1,l)
                wm=[wm;rand(1, neurons_per_I_H_O(1,l))];
                bm=[bm;rand(1, neurons_per_I_H_O(1,l))];
            end
            W_m{i,l} = wm;
            b_m{i,l} = bm;
        end
    end
    
    % Singlewave to determine the connections for all layers.
    for l = 1 : num_hidden_layers+1
        for i = 1 : neurons_per_H_O(1,l)
            wm=[];
            bm=[];
            risk=[];
            for j=1:num_masks(1,l)
                [m,r]=SingleWave(W_m{i,l}(j,:));
                risk=[risk;r];
                wm=[wm;m];
                [m,~]=SingleWave(b_m{i,l}(j,:));
                bm=[wm;m];
            end
            W_m{i,l} = wm;
            b_m{i,l} = bm;
            uncertainty{i,l}=risk;
        end
    end
    % Refine neuronal  value.
    for l = 1 : num_hidden_layers+1
        for i = 1 : neurons_per_H_O(1,l)
            [r,c]=find(uncertainty{1,1}==min(uncertainty{1,1}));
            W{1,l}(i,:)=W_m{i,l}(r,:);
        end
    end
    
    
    % Activation function (sigmoid)
    sigmoid = @(x) 1./(1 + exp(-x));
    
    % Forward pass function
        function output = forward_pass(input, W, b, sigmoid)
            a = input;
            for i = 1:length(W)
                a = sigmoid(W{i} * a + b{i});
            end
            output = a;
        end
    
    % Sample input
    input = [1; 2; 3];
    
    % Compute the network output
    output = forward_pass(input, W, b, sigmoid);
    
    % Display the output
    disp('Output:');
    disp(output);
    assumedBehaviours_o=ones(size(output)); % this value is assumed here, while in
    % real examples should be taken from Wasserstein Distance function to test
    % the previous output training dataset distribution with the previous
    % outputs distribution, with that after adding the node value.
    Pr_outputs = weightedProbability(output(1,1), output(2,1));
    best_alternative_idx = prospectCertainty(output(1,1), output(2,1));
    
    % % Visualization
    % figure;
    % hold on;
    % layer_positions = [0, cumsum(neurons_per_layer), outputSize];
    % for l = 1:length(layer_positions)
    %     for n = 1:layer_positions(l)
    %         plot(l, -n, 'o', 'MarkerSize', 10, 'MarkerFaceColor', 'auto');
    %     end
    % end
    % for l = 1:length(W)
    %     for i = 1:size(W{l}, 2)
    %         for j = 1:size(W{l}, 1)
    %             plot([l, l + 1], [-i, -j], 'k-');
    %         end
    %     end
    % end
    % xlim([0, length(W) + 1]);
    % ylim([-max(layer_positions) - 1, 1]);
    % xticks(0:length(W) + 1);
    % xticklabels(['Input', arrayfun(@(n) sprintf('Hidden %d', n), 1:num_hidden_layers, 'UniformOutput', false), 'Output']);
    % ylabel('Neurons');
    % title('Flexible ANN Visualization');
    % grid on;
    % hold off;
end
