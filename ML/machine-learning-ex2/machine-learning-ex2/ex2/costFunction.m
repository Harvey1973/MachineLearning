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
sum = 0;
sum_2 = 0;
% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta
%
% Note: grad should have the same dimensions as theta
%

%compute J first 
for i =1:m 
    sum = sum + (-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
endfor
J = sum/m;
for j = 1:size(theta)
    sum_2 = 0;
    for i = 1:m
        sum_2 = sum_2+(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
    endfor        
grad(j)=sum_2/m;
endfor 







% =============================================================

end
