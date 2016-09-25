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

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta


hthetatemp = X * theta;
htheta = sigmoid(hthetatemp);
theta1 = [0 ; theta(2:size(theta), :)];

sumall_1 = -y .* log(htheta) -  (1-y) .* log(1-htheta);
sumall_2 = (lambda / (2*m)) * sum (theta1 .^ 2);
%sumall = (htheta - y) .^ 2;
J = ((1/m) * sum(sumall_1)) + sumall_2;

sumgrad = X' * (htheta - y);
%grad = ((1/m) * sum(sumgrad) ) + (lambda/m) .* theta1;
grad = ( sumgrad + lambda*theta1 ) ./ m;

% =============================================================

end
