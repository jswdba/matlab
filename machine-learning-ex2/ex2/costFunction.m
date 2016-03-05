function [J, grad] = costFunction(theta, X, y)
%COSTFUNCTION Compute cost and gradient for logistic regression
%   J = COSTFUNCTION(theta, X, y) computes the cost of using theta as the
%   parameter for logistic regression and the gradient of the cost
%   w.r.t. to the parameters.

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;
grad = zeros(size(theta));
h =  zeros(size(m));
%theta is nx1 column vector. theta' is a row vector
% grad has the same dimension as theta, is also a nx1 column vector, for
% each feature of theta
% X is m by n 2-dim matrix, X dot theta = m x 1 column vector
% X dot theta - y is a mx1 column vector, tranpose to 1 x m row vector
%   multiply by X (mXn) ---> 1 x n row vector, same dimension as theta' 
%
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

h = sigmoid(X * theta);

grad = X' * (h - y)/m;


J = sum( -y' *  log(h)   -   (1 - y') * log( 1 - h) )/m ;

% y: m x1 
% X * theta is mx1, sigmoid(X * theta) is mx1, log(h) is mx1
% y is mx1, y * log(h)' = scalar

% =============================================================

end
