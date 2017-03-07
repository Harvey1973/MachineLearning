function [J grad] = nnCostFunction(nn_params, ...
                                   input_layer_size, ...
                                   hidden_layer_size, ...
                                   num_labels, ...
                                   X, y, lambda)
%NNCOSTFUNCTION Implements the neural network cost function for a two layer
%neural network which performs classification
%   [J grad] = NNCOSTFUNCTON(nn_params, hidden_layer_size, num_labels, ...
%   X, y, lambda) computes the cost and gradient of the neural network. The
%   parameters for the neural network are "unrolled" into the vector
%   nn_params and need to be converted back into the weight matrices. 
% 
%   The returned parameter grad should be a "unrolled" vector of the
%   partial derivatives of the neural network.
%

% Reshape nn_params back into the parameters Theta1 and Theta2, the weight matrices
% for our 2 layer neural network
Theta1 = reshape(nn_params(1:hidden_layer_size * (input_layer_size + 1)), ...
                 hidden_layer_size, (input_layer_size + 1));

Theta2 = reshape(nn_params((1 + (hidden_layer_size * (input_layer_size + 1))):end), ...
                 num_labels, (hidden_layer_size + 1));

% Setup some useful variables
% m is the number of examples 
m = size(X, 1);
         
% You need to return the following variables correctly 
J = 0;
Theta1_grad = zeros(size(Theta1));
Theta2_grad = zeros(size(Theta2));

% ====================== YOUR CODE HERE ======================
% Instructions: You should complete the code by working through the
%               following parts.
%
% Part 1: Feedforward the neural network and return the cost in the
%         variable J. After implementing Part 1, you can verify that your
%         cost function computation is correct by verifying the cost
%         computed in ex4.m
%
% Part 2: Implement the backpropagation algorithm to compute the gradients
%         Theta1_grad and Theta2_grad. You should return the partial derivatives of
%         the cost function with respect to Theta1 and Theta2 in Theta1_grad and
%         Theta2_grad, respectively. After implementing Part 2, you can check
%         that your implementation is correct by running checkNNGradients
%
%         Note: The vector y passed into the function is a vector of labels
%               containing values from 1..K. You need to map this vector into a 
%               binary vector of 1's and 0's to be used with the neural network
%               cost function.
%
%         Hint: We recommend implementing backpropagation using a for-loop
%               over the training examples if you are implementing it for the 
%               first time.
%
% Part 3: Implement regularization with the cost function and gradients.
%
%         Hint: You can implement this around the code for
%               backpropagation. That is, you can compute the gradients for
%               the regularization separately and then add them to Theta1_grad
%               and Theta2_grad from Part 2.
%

% Part1

a_1 = [ones(m,1) X];
z_2 = a_1*transpose(Theta1);
a_2 = sigmoid(z_2); 
a_2 = [ones(size(a_2,1),1) a_2]; 
z_3 = a_2*transpose(Theta2);
a_3 = sigmoid(z_3);
h_theta=a_3;

for i = 1:m
y_temp = zeros(1,num_labels);
y_temp(y(i)) = 1 ; 
J= J + sum(-y_temp.*log(h_theta(i,:))-(1-y_temp).*log(1-h_theta(i,:)));
endfor

%J=1/m*J;

%regularization terms 
% Theta_1
Theta1_reg = 0 ; 
for j = 1:size(Theta1,1)
for k = 2:size(Theta1,2)
Theta1_reg = Theta1_reg+(Theta1(j,k)**2);
endfor
endfor

%Theta_2 
Theta2_reg = 0;
for j = 1:size(Theta2,1)
for k = 2:size(Theta2,2)
Theta2_reg = Theta2_reg + (Theta2(j,k)**2);
endfor
endfor

reg_term = lambda*(Theta2_reg+Theta1_reg)/(2*m);

J = 1/m*J+reg_term;
% REMEMBER TO PUT THIS OUTSIDE OF MAIN LOOP 
DELTA_L1 = zeros(size(Theta1));
DELTA_L2 =  zeros(size(Theta2));
% Part2 backpropagation algorithm
for i =1:m 
a1 = [1 X(i,:)];
z2 = a1*transpose(Theta1);
a2 = sigmoid(z2); 
a2 = [1 a2]; 
z3 = a2*transpose(Theta2);
a3 = sigmoid(z3);
htheta=a3;
y_temp2 = zeros(1,num_labels);
y_temp2(y(i)) = 1 ;
delta_3 = (a3-y_temp2)(:);
delta_2 = transpose(Theta2)*delta_3.*transpose(sigmoidGradient([1 z2]));
delta_2 = delta_2(2:end);

DELTA_L1= DELTA_L1 +delta_2*a1;

DELTA_L2 = DELTA_L2 + delta_3*a2;
endfor
 
%Theta1_grad = 1/m*DELTA_L1;
%Theta2_grad = 1/m*DELTA_L2;

%add regularization 
Theta1_grad(:,1) = 1/m*DELTA_L1(:,1);
Theta1_grad(:,2:end) = 1/m*DELTA_L1(:,2:end)+lambda*Theta1(:,2:end)/m;
Theta2_grad(:,1) = 1/m*DELTA_L2(:,1);
Theta2_grad(:,2:end) = 1/m*DELTA_L2(:,2:end)+lambda*Theta2(:,2:end)/m;












% -------------------------------------------------------------

% =========================================================================

% Unroll gradients
grad = [Theta1_grad(:) ; Theta2_grad(:)];


end
