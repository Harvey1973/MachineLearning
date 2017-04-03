
function yhat = local_linear_regression(x,y,tau)
% local_linear_regression , takes an input vector x and y , where X is the design matrix
% each row of X has an example 
%y is the target value
% local_linear_regression takes an input vector x and y , at each point x 

nn = length(x);
X = [ones(nn,1) x];
yhat = zeros(nn,1);


%for ii = 1 :nn
%	w = exp(-(x-x(ii)).^2/(2*tau^2));
	%at each point we fit a theta to it , the points closest will be assigned weights , the points further will be assigned weight of 0
	% because we have added a bias term , we need to apply the weights to the bias term as well , thus the [w,w]
%	XWX = transpose(X)*([w,w].*X);
%	XtWy = transpose(X)*(w.*y);
%	theta = XWX\XtWy;
%	yhat(ii) = [1 x(ii)]*theta;
%end


%alternative implementaion by calculating the weight matrix first 
%for every point in vector x , there will be a weight vector and the normal equation will use that weight vector to try to fit a local_linear_regression
%base on only that point
for ii = 1:nn
	for jj = 1:nn
		W(jj,jj) = exp(-(x(ii)-x(jj))^2/(2*tau*tau));
	end
	theta = pinv(transpose(X)*W*X)*transpose(X)*W*y;
	yhat(ii) = [1 x(ii)]*theta;
end
