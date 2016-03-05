function p = predict(Theta1, Theta2, X)
%PREDICT Predict the label of an input given a trained neural network
%   p = PREDICT(Theta1, Theta2, X) outputs the predicted label of X given the
%   trained weights of a neural network (Theta1, Theta2)

% Useful values
m = size(X, 1);
num_labels = size(Theta2, 1);

% You need to return the following variables correctly 
p = zeros(size(X, 1), 1);

% ====================== YOUR CODE HERE ======================
% Instructions: Complete the following code to make predictions using
%               your learned neural network. You should set p to a 
%               vector containing labels between 1 to num_labels.
%
% Hint: The max function might come in useful. In particular, the max
%       function can also return the index of the max element, for more
%       information see 'help max'. If your examples are in rows, then, you
%       can use max(A, [], 2) to obtain the max for each row.
%
X1 = zeros(m,size(Theta1, 2));   %X1 = 5000x401
a2 = zeros(m, size(Theta1, 1));    %a2 = 5000x25
a3 = zeros(m,  size(Theta2, 1));    %a3 = 5000*10

%add bias unit 1s to matrix X,  X1 = 5000 x 401
X1 = [ones(m,1) X(:,:)];   
%z1 = sigmoid(X1 * Theta1');    h(X) = 5000x401 * 401x25 = 5000x25
a2 = sigmoid(X1 * Theta1');     %a2's dimension 5000x22
a2 = [ones(m,1) a2(:,:)];    % add column of 1s to a2
 
a3 = sigmoid(a2 * Theta2');
 
[M,num_labels] = max(a3, [], 2);


p=num_labels;










% =========================================================================


end
