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

for i = 1 : m;
    h = sigmoid(X(i, :) * theta);
    J = J + (-1 * y(i) * log(h) - (1 - y(i)) * log(1 - h));
    for j = 1 : length(theta);
        grad(j) = grad(j) + (h - y(i)) * X(i, j);
    end;
end;

% Calculate J.
addition_j = 0;
for j = 2 : length(theta);
    addition_j = addition_j + theta(j)^2;
end;
addition_j = addition_j * lambda / 2;
J = (J + addition_j) / m;

% Calculate grad.
for j = 1 : length(theta);
    if j ~= 1;
        grad(j) = grad(j) + lambda * theta(j);
    end;
    grad(j) = grad(j) / m;
end;




% =============================================================

end
