function y = lwlr(X_train, y_train, x, tau)

%%% YOUR CODE HERE
[mm,nn] = size(X_train);

%initial theta 
theta = zeros(nn,1);

%lambda 
lambda = 0.0001;

%weight w(i) = exp(-norm(x-X(i))/(2*band^2))
w = zeros(mm,1);

for ii = 1:mm	
	w(ii) = exp(-sum((transpose(x)-X_train(ii,:)).^2,2)/(2*tau*tau) );
	D(ii,ii) = -w(ii)*(1/(1+exp(-X_train(ii,:)*theta)))*(1-1/(1+exp(-X_train(ii,:)*theta)));
end 

%get hessian
H = transpose(X_train)*D*X_train - lambda * eye(nn);

Z = zeros(mm,1);
%gradient of cost function  Grad = transpose(X)*Z - lambda*theta
for ii = 1:mm	
	Z(ii) = w(ii)*(y_train(ii)-1/(1+exp(-X_train(ii,:)*theta)));
end

grad_theta = transpose(X_train)*Z - lambda*theta;

%final step 
theta = theta - inv(H)*grad_theta;

%return y

y = double(x'*theta>0);
% w_2 = exp(-sum((X_train - repmat(transpose(x),mm,1)).^2,2)/(2*tau*tau));