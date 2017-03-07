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

%a_1 initial 
a_1 = zeros(m,size(X,2));

%add bias term for a(1)
a_1 = [ones(m, 1) X];

% hidden layer expression 
Z_2 = a_1*transpose(Theta1);

%a_2 initial 

a_2 = sigmoid(Z_2);

%add bias term for a_2 
a_2 = [ones(m,1) a_2];

%Z_3
Z_3 = a_2*transpose(Theta2);

%a_3 final output layer 
a_3 = sigmoid(Z_3);

% make prediction 

for i =1:m 
    [v,index] = max(a_3,[],2);
    p(i) = index(i);
endfor





% =========================================================================


end
