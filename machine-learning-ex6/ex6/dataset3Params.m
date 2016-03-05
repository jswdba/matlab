function [C, sigma] = dataset3Params(X, y, Xval, yval)
%EX6PARAMS returns your choice of C and sigma for Part 3 of the exercise
%where you select the optimal (C, sigma) learning parameters to use for SVM
%with RBF kernel
%   [C, sigma] = EX6PARAMS(X, y, Xval, yval) returns your choice of C and 
%   sigma. You should complete this function to return the optimal C and 
%   sigma based on a cross-validation set.
%

% You need to return the following variables correctly.
C = 1;
sigma = 0.3;

% ====================== YOUR CODE HERE ======================
% Instructions: Fill in this function to return the optimal C and sigma
%               learning parameters found using the cross validation set.
%               You can use svmPredict to predict the labels on the cross
%               validation set. For example, 
%                   predictions = svmPredict(model, Xval);
%               will return the predictions on the cross validation set.
%
%  Note: You can compute the prediction error using 
%        mean(double(predictions ~= yval))
%



a = [0.01, 0.03, 0.1, 0.3, 1, 3, 10, 30];
error_matrix = zeros(length(a),length(a));
v=0;
v1 = 0;

for c = 1:length(a)
    for s = 1:length(a)
        
        model= svmTrain(X, y, a(c), @(x1, x2) gaussianKernel(x1, x2, a(s)));
        
        predictions = svmPredict(model, Xval);
        
        
        error_matrix(c,s) = mean(double(predictions ~= yval));
         
        
    end
    
end

[v,ind] = min(error_matrix);

[v1,ind1]=min(min(error_matrix));



C = a(ind(ind1));
sigma = a(ind1);



% =========================================================================

end
