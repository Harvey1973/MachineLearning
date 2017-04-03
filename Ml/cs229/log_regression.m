%logistic regression using newton's method

%this function will return 2 paramters ,first one is model for parameters , l1 is the loss

function [theta,l1] = log_regression(X,Y,max_iter)
%rows of X are training examples
%rows of Y are corresponding -1/1 value
%newton's method  theta = theta - inverse(H)*grad(log_liklihood)

%number of examples 
mm = size(X,1);
%number of labels 
nn = size(X,2);
%size of theta will match the number of features
theta = zeros(nn,1);

%%%return the loss (log likelihood)‰ for each iteration
l1 = zeros(max_iter,1);

for ii = 1 : max_iter
	margins = Y .*(X*theta);
	l1(ii) =(1/mm)*sum(log(1+exp(-margins)));
	probs = 1 ./ (1+exp(margins));
	grad = (-1/mm)*(transpose(X)*(probs .* Y));
	H = (1/mm)*(transpose(X)*diag(probs.*(1-probs))*X);
	theta = theta - pinv(H)*grad;
end
