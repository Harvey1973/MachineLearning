function J = computeCost(X, y, theta)
%COMPUTECOST Compute cost for linear regression
%   J = COMPUTECOST(X, y, theta) computes the cost of using theta as the
%   parameter for linear regression to fit the data points in X and y

% Initialize some useful values
m = length(y); % number of training examples

% You need to return the following variables correctly 
J = 0;

% ====================== YOUR CODE HERE ======================
% Instructions: Compute the cost of a particular choice of theta
%               You should set J to the cost.

theta0 = theta(1,:);  % first row of theta 
theta1 = theta(2,:);  % second row of theta
% caculate the cost using given theta values 
temp = 0  % variabels to store (h(x)-y)^2
for i = 1:m 
    temp = temp + ((theta0+theta1*X(i,2))-y(i))^2;
endfor 

J=temp/(2*m)




% =========================================================================

end
