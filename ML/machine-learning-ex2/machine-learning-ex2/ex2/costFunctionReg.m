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
sum_1 = 0;
sum_2 = 0;
sum_3 = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta.
%               You should set J to the cost.
%               Compute the partial derivatives and set grad to the partial
%               derivatives of the cost w.r.t. each parameter in theta

% cost function with penalizing term 
for i =1:m 
    sum_1 = sum_1 + (-y(i)*log(sigmoid(X(i,:)*theta))-(1-y(i))*log(1-sigmoid(X(i,:)*theta)));
endfor
%penalizing term 
for j = 2:size(theta)
    sum_2 = sum_2+(theta(j)**2);
endfor
J=sum_1/m+lambda*sum_2/(2*m);

for i = 1:m
    sum_3 = sum_3+(sigmoid(X(i,:)*theta)-y(i))*X(i,1);
endfor  
grad(1) = sum_3/m ;

for j = 2:size(theta)
    sum_2 = 0;
    for i = 1:m
        sum_2 = sum_2+(sigmoid(X(i,:)*theta)-y(i))*X(i,j);
    endfor        
grad(j)=sum_2/m+lambda*theta(j)/m;
endfor 




% =============================================================

end
