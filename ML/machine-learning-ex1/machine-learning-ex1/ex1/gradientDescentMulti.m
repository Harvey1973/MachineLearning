function [theta, J_history] = gradientDescentMulti(X, y, theta, alpha, num_iters)
%GRADIENTDESCENTMULTI Performs gradient descent to learn theta
%   theta = GRADIENTDESCENTMULTI(x, y, theta, alpha, num_iters) updates theta by
%   taking num_iters gradient steps with learning rate alpha

% Initialize some useful values
m = length(y); % number of training examples
J_history = zeros(num_iters, 1);

temp = zeros(size(X,2),1);  % initilize temporary array of 0s 
for iter = 1:num_iters

    % ====================== YOUR CODE HERE ======================
    % Instructions: Perform a single gradient step on the parameter vector
    %               theta. 
    %
    % Hint: While debugging, it can be useful to print out the values
    %       of the cost function (computeCostMulti) and gradient here.
    %

for  j = 1:size(X,2)    % this loop is for updating all the theta(j) value
sum = 0;
for  i = 1:m     % this loop is for computing the sum of h(x)-y
sum = sum + ((X*theta)(i)-y(i))*X(i,j);
endfor
temp(j) = theta(j)-(alpha/m)*sum;   % simultaneously update all theta(j)
endfor 

theta = temp;










    % ============================================================

    % Save the cost J in every iteration    
    J_history(iter) = computeCostMulti(X, y, theta);

end

end
