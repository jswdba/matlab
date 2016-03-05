function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1)); %25 x 401  
Theta2_grad = zeros(size(Theta2)); %10 x 26

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% initialize variables


s2 = size(Theta1, 1); % s2 = number of units in layer 2
%Theta1 =  400x25; add a bias unit, theta1= 401x25

a1 = [ones(m,1) X];    %add base unkt 1 column to X

z2 = a1 * Theta1';    % a2 =  5000x401 *  401x25 = 5000x25;  

a2 = sigmoid(z2);   % z2  = 5000x25; 


a2 = [ones(m,1) a2];  % a2 add 1 bias unit, now becomes 5000x26


z3 = a2 * Theta2';   % h3 = 5000x26 * 26x10  = 5000x10

a3 = sigmoid(z3);   % 5000 x 10

% Part 1:


for t = 1:num_labels   % convert y to logical arrays
    
    y1(:,t) = y == t;  %y1 = 5000 x 10
    
end

J =  1/m * sum(-y1 .* log(a3)  - (1-y1) .* log(1-a3));  % sum all the rows --> 1 * 10

J = sum(J)  %  sum all the columns --> 1 x 1

J = J + lambda /2/m * (sum(sum(Theta1(:,2:end).^2)) + sum(sum(Theta2(:,2:end).^2)) )

%-----------------------------------------------------------------
    
delta3 = a3- y1 ; % 5000 x 10
   
    
delta2 = delta3 * Theta2(:,2:end) .* sigmoidGradient(z2);  % 5000x 10 * 10x 25 =  5000x25


Theta2_grad =  1/m* (Theta2_grad + delta3' * a2 );  % 10x26 + 10 x 5000 * 5000 * 26 = 10x26
    
Theta1_grad =  1/m *( Theta1_grad+ delta2' * a1 );  % 25x401 + 25 x 5000 * 5000 * 401 = 25x401


Theta1_grad(:, 2:end) = Theta1_grad(:, 2:end) + lambda / m * Theta1(:, 2:end); %trim bias unit

Theta2_grad(:, 2:end) =  Theta2_grad(:, 2:end) + lambda / m * Theta2(:, 2:end); %trim bias unit





% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
