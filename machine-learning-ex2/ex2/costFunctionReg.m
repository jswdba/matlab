function [J, grad] = costFunctionReg(theta, X, y, lambda)
%COSTFUNCTIONREG Compute cost and gradient for logistic regression with regularization
%   J = COSTFUNCTIONREG(theta, X, y, lambda) computes the cost of using
%   theta as the parameter for regularized logistic regression and the
%   gradient of the cost w.r.t. to the parameters. 

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
i = 0;
n =  size(X, 1);
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% Hint: The computation of the cost function and gradients can be
%       efficiently vectorized. For example, consider the computation
%
%           sigmoid(X * theta)
%
%       Each row of the resulting matrix will contain the value of the
%       prediction for that example. You can make use of this to vectorize
%       the cost function and gradient computations. 
%
% Hint: When computing the gradient of the regularized cost function, 
%       there're many possible vectorized solutions, but one solution
%       looks like:
%           grad = (unregularized gradient for logistic regression)
%           temp = theta; 
%           temp(1) = 0;   % because we don't add anything for j = 0  
%           grad = grad + YOUR_CODE_HERE (using the temp variable)
%


% calculate hypothesis
h = sigmoid(X*theta);

% regularize theta by removing first value
% theta_reg = [0;theta(2:end, :);];
theta_reg = [0;theta(2:end)];

J = (1/m)*sum(-y'* log(h) - (1 - y)'*log(1-h)) + lambda * 1/2 * 1/m * (theta_reg'*theta_reg);

grad = X' * (h - y)/m + lambda * theta_reg /m;
% grad = (1/m)*(X'*(h-y)+lambda*theta_reg);



%h = sigmoid(X * theta);

%J = sum( -y' *  log(h)   -   (1 - y') * log( 1 - h) )/m + lambda * 1/2 * 1/m * (theta' * theta);

%grad = X' * (h - y)/m;
%temp = theta;
%temp(1) = 0
%grad = grad + temp

%for i = 1:n
%   if i = 1 
%         grad = X' * (h - y)/m;
%    else
%         grad = X' * (h - y)/m - lambda * theta(i) /m;
% 
% 
% end
    
% X' is n x m; (h-y) is m x 1, grad is n x 1 column vector




% =============================================================

end
